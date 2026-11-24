"""Actuator models: PD law, DC-motor torque-speed envelope, optional actuator net.

The real motor driver (MIT mode) evaluates ``tau = kp (q* - q) + kd (dq* - dq) + tau_ff``
on its own microcontroller at several kHz. In simulation we evaluate the same
law at the physics rate (500 Hz by default) and then pass the requested torque
through a model of what the motor can physically deliver.

DC-motor envelope
-----------------
For a permanent-magnet DC / BLDC motor with a fixed supply voltage, the
available torque falls linearly with speed when the motor is *motoring*
(torque and velocity have the same sign)::

    tau_avail(w) = tau_stall * (1 - |w| / w_noload)     (clipped to [0, tau_limit])

When braking (torque opposes velocity) the full ``tau_limit`` is available.
``tau_stall`` and ``w_noload`` are measured on the bench (``scripts/motor_sysid.py``)
or taken from the datasheet (both at the joint/output side).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def pd_torque(q_target, q, dq, kp, kd, dq_target=0.0, tau_ff=0.0):
    """MIT-mode PD law, vectorised over any broadcastable shape."""
    return kp * (q_target - q) + kd * (dq_target - dq) + tau_ff


class DCMotorModel:
    """Torque-speed envelope of a DC motor + strength scaling (for domain randomisation)."""

    def __init__(self, torque_limit: float, stall_torque: float, no_load_speed: float):
        if no_load_speed <= 0:
            raise ValueError("no_load_speed must be positive")
        self.torque_limit = float(torque_limit)
        self.stall_torque = float(stall_torque)
        self.no_load_speed = float(no_load_speed)

    @classmethod
    def from_config(cls, motor_cfg: dict) -> "DCMotorModel":
        return cls(motor_cfg["torque_limit"], motor_cfg.get("stall_torque", motor_cfg["torque_limit"]),
                   motor_cfg.get("no_load_speed", 1e6))

    def max_torque(self, tau_cmd, dq, strength=1.0):
        """Per-element available torque magnitude given commanded sign and joint speed."""
        tau_cmd = np.asarray(tau_cmd, dtype=np.float64)
        dq = np.asarray(dq, dtype=np.float64)
        motoring = np.sign(tau_cmd) * np.sign(dq) > 0
        curve = self.stall_torque * (1.0 - np.abs(dq) / self.no_load_speed)
        avail = np.where(motoring, np.clip(curve, 0.0, self.torque_limit), self.torque_limit)
        return avail * strength

    def __call__(self, tau_cmd, dq, strength=1.0):
        lim = self.max_torque(tau_cmd, dq, strength)
        return np.clip(tau_cmd, -lim, lim)


class ActuatorNet:
    """Learned actuator model (Hwangbo et al., 2019 style), optional.

    An MLP maps a short history of joint position errors and velocities to the
    torque actually produced. Train it on logged real-motor data with
    ``scripts/train_actuator_net.py``; it then replaces :class:`DCMotorModel`
    in the environment when ``actuator.type: actuator_net`` is configured.

    This is a functional but deliberately minimal implementation: the network
    is evaluated in numpy (weights exported from torch) so the simulator does
    not need torch in the inner physics loop.
    """

    def __init__(self, weights: list[tuple[np.ndarray, np.ndarray]], history: int,
                 pos_scale: float = 1.0, vel_scale: float = 0.1, torque_limit: float = np.inf):
        self.weights = weights
        self.history = history
        self.pos_scale = pos_scale
        self.vel_scale = vel_scale
        self.torque_limit = torque_limit

    @classmethod
    def load(cls, path: str | Path, torque_limit: float = np.inf) -> "ActuatorNet":
        data = np.load(path)
        n = int(data["n_layers"])
        weights = [(data[f"W{i}"], data[f"b{i}"]) for i in range(n)]
        return cls(weights, int(data["history"]), float(data["pos_scale"]), float(data["vel_scale"]), torque_limit)

    def save(self, path: str | Path) -> None:
        arrays = {"n_layers": len(self.weights), "history": self.history,
                  "pos_scale": self.pos_scale, "vel_scale": self.vel_scale}
        for i, (W, b) in enumerate(self.weights):
            arrays[f"W{i}"] = W
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)

    def features(self, pos_err_hist: np.ndarray, vel_hist: np.ndarray) -> np.ndarray:
        """``*_hist``: (..., history), newest last -> (..., 2*history)."""
        return np.concatenate([pos_err_hist * self.pos_scale, vel_hist * self.vel_scale], axis=-1)

    def __call__(self, pos_err_hist: np.ndarray, vel_hist: np.ndarray) -> np.ndarray:
        x = self.features(pos_err_hist, vel_hist)
        for i, (W, b) in enumerate(self.weights):
            x = x @ W.T + b
            if i < len(self.weights) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))  # ELU (min() avoids overflow warnings)
        return np.clip(x[..., 0], -self.torque_limit, self.torque_limit)
