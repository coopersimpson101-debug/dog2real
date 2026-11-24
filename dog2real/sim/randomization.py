"""Domain randomisation and sensor-noise models.

Each environment owns its own ``MjModel`` copy, so physical parameters can be
randomised per environment at every reset. Parameters that do not live in the
model (motor strength, PD gain error, latency) are returned as arrays and used
by the environment's actuator loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _rng_uniform(rng: np.random.Generator, lo_hi, size):
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    if hi < lo:
        raise ValueError(f"invalid range {lo_hi}")
    return rng.uniform(lo, hi, size=size)


@dataclass
class RandomizationSample:
    """Per-environment randomised parameters (all arrays have leading dim N)."""

    friction: np.ndarray
    added_base_mass: np.ndarray
    base_com_offset: np.ndarray        # (N, 3)
    link_mass_scale: np.ndarray        # (N,)
    motor_strength: np.ndarray         # (N, 12)
    kp_scale: np.ndarray               # (N, 12)
    kd_scale: np.ndarray               # (N, 12)
    latency_substeps: np.ndarray       # (N,) int
    extra: dict = field(default_factory=dict)

    def subset(self, idx) -> "RandomizationSample":
        return RandomizationSample(
            self.friction[idx], self.added_base_mass[idx], self.base_com_offset[idx],
            self.link_mass_scale[idx], self.motor_strength[idx], self.kp_scale[idx],
            self.kd_scale[idx], self.latency_substeps[idx])


class DomainRandomizer:
    """Samples and applies physical parameter randomisation.

    Config keys (all optional; ``enabled: false`` returns nominal values)::

        friction: [lo, hi]                # absolute Coulomb friction coefficient
        added_base_mass: [lo, hi]         # kg added to the base (payload, battery)
        base_com_offset: [dx, dy, dz]     # +- metres, uniform per axis
        link_mass_scale: [lo, hi]         # multiplicative on leg link masses
        motor_strength: [lo, hi]          # multiplicative on available torque, per motor
        kp_scale: [lo, hi]                # PD gain error, per motor
        kd_scale: [lo, hi]
        latency_substeps: [lo, hi]        # integer physics substeps of actuation delay
    """

    def __init__(self, cfg: dict | None, num_joints: int = 12):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.cfg = cfg
        self.nj = num_joints

    def sample(self, n: int, rng: np.random.Generator, nominal_friction: float = 1.0) -> RandomizationSample:
        c = self.cfg
        if not self.enabled:
            return RandomizationSample(
                friction=np.full(n, nominal_friction), added_base_mass=np.zeros(n),
                base_com_offset=np.zeros((n, 3)), link_mass_scale=np.ones(n),
                motor_strength=np.ones((n, self.nj)), kp_scale=np.ones((n, self.nj)),
                kd_scale=np.ones((n, self.nj)), latency_substeps=np.zeros(n, dtype=np.int64))
        com = np.asarray(c.get("base_com_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
        lat = c.get("latency_substeps", [0, 0])
        return RandomizationSample(
            friction=_rng_uniform(rng, c.get("friction", [nominal_friction] * 2), n),
            added_base_mass=_rng_uniform(rng, c.get("added_base_mass", [0.0, 0.0]), n),
            base_com_offset=rng.uniform(-1.0, 1.0, size=(n, 3)) * com,
            link_mass_scale=_rng_uniform(rng, c.get("link_mass_scale", [1.0, 1.0]), n),
            motor_strength=_rng_uniform(rng, c.get("motor_strength", [1.0, 1.0]), (n, self.nj)),
            kp_scale=_rng_uniform(rng, c.get("kp_scale", [1.0, 1.0]), (n, self.nj)),
            kd_scale=_rng_uniform(rng, c.get("kd_scale", [1.0, 1.0]), (n, self.nj)),
            latency_substeps=rng.integers(int(lat[0]), int(lat[1]) + 1, size=n),
        )


class ModelParamApplier:
    """Remembers nominal model parameters and writes randomised ones into a model copy."""

    def __init__(self, nominal_model, base_body_id: int, leg_body_ids: list[int]):
        self.friction0 = nominal_model.geom_friction.copy()
        self.mass0 = nominal_model.body_mass.copy()
        self.inertia0 = nominal_model.body_inertia.copy()
        self.ipos0 = nominal_model.body_ipos.copy()
        self.base_id = base_body_id
        self.leg_ids = list(leg_body_ids)

    def apply(self, model, s: RandomizationSample, i: int) -> None:
        model.geom_friction[:] = self.friction0
        model.geom_friction[:, 0] = s.friction[i]
        model.body_mass[:] = self.mass0
        model.body_inertia[:] = self.inertia0
        model.body_ipos[:] = self.ipos0
        # base payload: scale inertia proportionally (keeps the inertia physically valid)
        m0 = self.mass0[self.base_id]
        m = max(m0 + s.added_base_mass[i], 0.1 * m0)
        model.body_mass[self.base_id] = m
        model.body_inertia[self.base_id] = self.inertia0[self.base_id] * (m / m0)
        model.body_ipos[self.base_id] = self.ipos0[self.base_id] + s.base_com_offset[i]
        k = s.link_mass_scale[i]
        for b in self.leg_ids:
            model.body_mass[b] = self.mass0[b] * k
            model.body_inertia[b] = self.inertia0[b] * k


class IMUNoiseModel:
    """Gyro bias + white noise and attitude-estimate error for the projected gravity.

    * gyro: ``w_meas = w + b + n``,  b ~ U(-bias, bias) per episode, n ~ N(0, noise^2)
    * gravity: the onboard attitude filter is modelled as the true gravity rotated by
      a small per-episode tilt error plus white noise, then re-normalised.
    """

    def __init__(self, cfg: dict | None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.gyro_bias = float(cfg.get("gyro_bias", 0.0))
        self.gyro_noise = float(cfg.get("gyro_noise", 0.0))
        self.tilt_bias = float(cfg.get("tilt_bias", 0.0))
        self.gravity_noise = float(cfg.get("gravity_noise", 0.0))

    def sample_bias(self, n: int, rng: np.random.Generator):
        if not self.enabled:
            return np.zeros((n, 3)), np.zeros((n, 3))
        return (rng.uniform(-self.gyro_bias, self.gyro_bias, (n, 3)),
                rng.uniform(-self.tilt_bias, self.tilt_bias, (n, 3)))

    def apply(self, gyro, gravity, gyro_bias, tilt_bias, rng: np.random.Generator):
        if not self.enabled:
            return gyro, gravity
        gyro = gyro + gyro_bias + rng.normal(0.0, self.gyro_noise, gyro.shape)
        g = gravity + np.cross(tilt_bias, gravity) + rng.normal(0.0, self.gravity_noise, gravity.shape)
        g = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-6)
        return gyro, g


class ObservationNoise:
    """Additive uniform noise on joint encoders (uniform as in legged_gym)."""

    def __init__(self, cfg: dict | None):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", True))
        self.level = float(cfg.get("level", 1.0))
        self.dof_pos = float(cfg.get("dof_pos", 0.0))
        self.dof_vel = float(cfg.get("dof_vel", 0.0))

    def apply(self, dof_pos, dof_vel, rng: np.random.Generator):
        if not self.enabled:
            return dof_pos, dof_vel
        dp = dof_pos + rng.uniform(-1, 1, dof_pos.shape) * self.dof_pos * self.level
        dv = dof_vel + rng.uniform(-1, 1, dof_vel.shape) * self.dof_vel * self.level
        return dp, dv
