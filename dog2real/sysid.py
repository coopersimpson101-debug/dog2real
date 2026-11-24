"""Least-squares system identification helpers for matching the sim to a real motor.

``fit_joint_dynamics`` -- an unloaded (or known-load) joint obeys

    tau = J * ddq + b * dq + f_c * sign(dq)

where ``J`` is the reflected rotor inertia (MuJoCo ``armature``), ``b`` the
viscous damping (``damping``) and ``f_c`` the Coulomb friction
(``frictionloss``). Given logged ``(t, q, dq, tau)`` from a chirp experiment
(``scripts/motor_sysid.py``) we solve for ``[J, b, f_c]`` by linear least squares.

``fit_torque_speed`` -- from saturated-torque samples at different speeds, fit
``tau_max = tau_stall * (1 - |w| / w_noload)`` (the DC-motor envelope used in
``dog2real.sim.actuator.DCMotorModel``).
"""

from __future__ import annotations

import numpy as np


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    kernel = np.ones(k) / k
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), kernel, mode="valid")


def fit_joint_dynamics(t, dq, tau, smooth: int = 5, min_speed: float = 0.05) -> dict:
    t, dq, tau = (np.asarray(a, dtype=np.float64) for a in (t, dq, tau))
    dq_s = _smooth(dq, smooth)
    ddq = np.gradient(dq_s, t)
    mask = np.abs(dq_s) > min_speed  # sign(dq) is ill-defined near zero speed
    A = np.stack([ddq, dq_s, np.sign(dq_s)], axis=1)[mask]
    y = _smooth(tau, smooth)[mask]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    rmse = float(np.sqrt(np.mean((A @ coef - y) ** 2)))
    return {"armature": float(coef[0]), "damping": float(coef[1]), "frictionloss": float(coef[2]), "rmse": rmse}


def fit_torque_speed(speed, tau_max) -> dict:
    w = np.abs(np.asarray(speed, dtype=np.float64))
    tm = np.abs(np.asarray(tau_max, dtype=np.float64))
    A = np.stack([np.ones_like(w), w], axis=1)
    (c0, c1), *_ = np.linalg.lstsq(A, tm, rcond=None)
    stall = float(c0)
    no_load = float(-c0 / c1) if c1 < 0 else float("inf")
    return {"stall_torque": stall, "no_load_speed": no_load}


def chirp(t: np.ndarray, f0: float, f1: float, duration: float, amplitude: float) -> np.ndarray:
    """Linear frequency sweep used as the position excitation."""
    k = (f1 - f0) / duration
    return amplitude * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t ** 2))


BENCH_XML = """<mujoco model="bench_motor">
  <option timestep="{dt}"/>
  <worldbody>
    <body name="rotor">
      <joint name="j" type="hinge" axis="0 0 1" armature="{armature}" damping="{damping}" frictionloss="{frictionloss}"/>
      <geom type="cylinder" size="0.02 0.01" mass="1e-6"/>
    </body>
  </worldbody>
  <actuator><motor joint="j" ctrlrange="-{tlim} {tlim}"/></actuator>
</mujoco>"""


def simulate_bench_motor(armature: float, damping: float, frictionloss: float, kp: float, kd: float,
                         duration: float = 6.0, amplitude: float = 1.0, f0: float = 0.2, f1: float = 4.0,
                         dt: float = 0.001, torque_limit: float = 18.0, torque_noise: float = 0.0,
                         seed: int = 0) -> dict:
    """Simulate the chirp experiment of ``scripts/motor_sysid.py`` on an unloaded motor.

    Returns logged arrays like the real experiment would. Used to validate the
    identification pipeline (it should recover the parameters it was given).
    """
    import mujoco

    m = mujoco.MjModel.from_xml_string(BENCH_XML.format(dt=dt, armature=armature, damping=damping,
                                                        frictionloss=frictionloss, tlim=torque_limit))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    n = int(duration / dt)
    t = np.arange(n) * dt
    q_des = chirp(t, f0, f1, duration, amplitude)
    q, dq, tau = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
        q[k], dq[k] = d.qpos[0], d.qvel[0]
        u = np.clip(kp * (q_des[k] - d.qpos[0]) - kd * d.qvel[0], -torque_limit, torque_limit)
        tau[k] = u + rng.normal(0.0, torque_noise)  # "measured" (current-based) torque estimate
        d.ctrl[0] = u
        mujoco.mj_step(m, d)
    return {"t": t, "q_des": q_des, "q": q, "dq": dq, "tau": tau}
