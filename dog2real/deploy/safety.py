"""Safety layer between the policy and the motors.

Every 50 Hz tick the runtime asks the safety layer to

1. ``check_state``: latch a fault on e-stop, excessive tilt, joint over-speed,
   joints outside their hard limits, or lost motor communication;
2. ``filter``: turn the policy's position targets into a *safe* motor command:
   clip to joint limits (with a margin), rate-limit the target change, and
   reshape the target so that the PD torque the motor will apply,
   ``kp (q* - q) - kd dq``, never exceeds ``max_torque``.

Once a fault is latched the output switches to *damping mode* (``kp = 0``,
``kd = damping_kd``): the legs go limp but are braked, which lets the robot
sag to the ground instead of collapsing or flailing. Faults stay latched until
``reset()`` is called explicitly by the operator.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class MotorCommand:
    q_des: np.ndarray
    dq_des: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray


@dataclass
class SafetyConfig:
    lower: np.ndarray
    upper: np.ndarray
    limit_margin: float = 0.05        # rad kept away from the hard stops
    max_target_step: float = 0.15     # rad per tick (7.5 rad/s at 50 Hz)
    max_torque: float = 15.0          # Nm, PD torque budget
    max_joint_vel: float = 20.0       # rad/s, fault above this
    tilt_cutoff_rad: float = 1.0      # fault if the body tilts more than this
    damping_kd: float = 1.0           # Nm s/rad in damping mode
    hard_limit_tolerance: float = 0.15  # rad past the hard limit before faulting
    tilt_check: bool = True           # disable for the get-up task

    @classmethod
    def from_dicts(cls, robot: dict, safety: dict | None = None) -> "SafetyConfig":
        from dog2real.robot.model_builder import joint_limits

        lo, hi = joint_limits(robot)
        s = dict(safety or {})
        s.setdefault("max_torque", robot["motor"]["torque_limit"])
        s.setdefault("max_joint_vel", robot["motor"].get("velocity_limit", 20.0))
        known = {k: v for k, v in s.items() if k in cls.__dataclass_fields__}
        return cls(lower=lo, upper=hi, **known)


class SafetyLayer:
    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self.lo = cfg.lower + cfg.limit_margin
        self.hi = cfg.upper - cfg.limit_margin
        self._lock = threading.Lock()
        self.fault: str | None = None
        self.last_q_des: np.ndarray | None = None
        self.clamp_events = 0

    # ----------------------------------------------------------------- faults --
    def estop(self, reason: str = "e-stop") -> None:
        """Thread-safe: may be called from a signal handler, GPIO callback or UI thread."""
        with self._lock:
            if self.fault is None:
                self.fault = reason

    @property
    def ok(self) -> bool:
        return self.fault is None

    def reset(self) -> None:
        with self._lock:
            self.fault = None
            self.last_q_des = None

    def check_state(self, q, dq, projected_gravity, comm_ok: bool = True) -> bool:
        c = self.cfg
        q, dq, g = np.asarray(q), np.asarray(dq), np.asarray(projected_gravity)
        if not comm_ok:
            self.estop("motor communication timeout")
        elif not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq)) and np.all(np.isfinite(g))):
            self.estop("non-finite sensor reading")
        elif np.any(np.abs(dq) > c.max_joint_vel):
            self.estop(f"joint over-speed (max |dq| = {np.max(np.abs(dq)):.1f} rad/s)")
        elif np.any(q < c.lower - c.hard_limit_tolerance) or np.any(q > c.upper + c.hard_limit_tolerance):
            self.estop("joint outside hard limits")
        elif c.tilt_check:
            # projected gravity z = -cos(tilt)
            gz = g[2] / max(np.linalg.norm(g), 1e-6)
            if gz > -np.cos(c.tilt_cutoff_rad):
                self.estop(f"tilt cutoff (tilt = {np.degrees(np.arccos(np.clip(-gz, -1, 1))):.0f} deg)")
        return self.ok

    # ---------------------------------------------------------------- filter --
    def filter(self, q_des, q, dq, kp, kd) -> MotorCommand:
        q, dq = np.asarray(q, dtype=np.float64), np.asarray(dq, dtype=np.float64)
        n = q.shape[0]
        kp = np.broadcast_to(np.asarray(kp, dtype=np.float64), (n,)).copy()
        kd = np.broadcast_to(np.asarray(kd, dtype=np.float64), (n,)).copy()
        zeros = np.zeros(n)
        if not self.ok:
            self.last_q_des = q.copy()
            return MotorCommand(q.copy(), zeros, zeros.copy(), np.full(n, self.cfg.damping_kd), zeros.copy())

        raw = np.asarray(q_des, dtype=np.float64)
        if not np.all(np.isfinite(raw)):
            self.estop("non-finite policy output")
            return self.filter(raw, q, dq, kp, kd)
        tgt = np.clip(raw, self.lo, self.hi)
        prev = q if self.last_q_des is None else self.last_q_des
        step = self.cfg.max_target_step
        tgt = np.clip(tgt, prev - step, prev + step)
        # torque budget: kp (tgt - q) - kd dq must lie within +-max_torque
        tau = kp * (tgt - q) - kd * dq
        tau_c = np.clip(tau, -self.cfg.max_torque, self.cfg.max_torque)
        safe_kp = np.where(kp > 1e-9, kp, 1.0)
        tgt = np.where(kp > 1e-9, q + (tau_c + kd * dq) / safe_kp, tgt)
        tgt = np.clip(tgt, self.lo, self.hi)
        if np.any(np.abs(tgt - raw) > 1e-9):
            self.clamp_events += 1
        self.last_q_des = tgt
        return MotorCommand(tgt, zeros, kp, kd, zeros.copy())
