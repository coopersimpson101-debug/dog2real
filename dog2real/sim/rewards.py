"""Reward terms. Each term maps a batched state dict to an (N,) array.

The total reward is ``sum_i scale_i * term_i * dt`` (scales come from the YAML
config; a scale of 0 removes the term). Positive scales are rewards, negative
ones penalties. Term definitions follow Rudin et al., "Learning to Walk in
Minutes Using Massively Parallel Deep RL" (CoRL 2021) with a few additions for
the get-up task.
"""

from __future__ import annotations

import numpy as np

UP = np.array([0.0, 0.0, -1.0])


# ----------------------------------------------------------------- walking ----
def tracking_lin_vel(s):
    err = np.sum((s["commands"][:, :2] - s["base_lin_vel"][:, :2]) ** 2, axis=1)
    return np.exp(-err / s["tracking_sigma"])


def tracking_ang_vel(s):
    err = (s["commands"][:, 2] - s["base_ang_vel"][:, 2]) ** 2
    return np.exp(-err / s["tracking_sigma"])


def lin_vel_z(s):
    return s["base_lin_vel"][:, 2] ** 2


def ang_vel_xy(s):
    return np.sum(s["base_ang_vel"][:, :2] ** 2, axis=1)


def orientation(s):
    return np.sum(s["projected_gravity"][:, :2] ** 2, axis=1)


def base_height(s):
    return (s["base_height"] - s["base_height_target"]) ** 2


def torques(s):
    return np.sum(s["torques"] ** 2, axis=1)


def dof_vel(s):
    return np.sum(s["dof_vel"] ** 2, axis=1)


def dof_acc(s):
    return np.sum(((s["dof_vel"] - s["last_dof_vel"]) / s["dt"]) ** 2, axis=1)


def action_rate(s):
    return np.sum((s["actions"] - s["last_actions"]) ** 2, axis=1)


def feet_air_time(s):
    """Reward steps whose swing lasted > 0.5 s, credited at touchdown; only when moving."""
    r = np.sum((s["feet_air_time"] - 0.5) * s["first_contact"], axis=1)
    return r * (np.linalg.norm(s["commands"][:, :2], axis=1) > 0.1)


def collision(s):
    return s["collisions"].astype(np.float64)


def dof_pos_limits(s):
    below = np.clip(s["soft_lower"] - s["dof_pos"], 0.0, None)
    above = np.clip(s["dof_pos"] - s["soft_upper"], 0.0, None)
    return np.sum(below + above, axis=1)


def stand_still(s):
    still = np.linalg.norm(s["commands"][:, :2], axis=1) < 0.1
    return np.sum(np.abs(s["dof_pos"] - s["default_dof_pos"]), axis=1) * still


def termination(s):
    return (s["terminated"] & ~s["time_out"]).astype(np.float64)


# ---------------------------------------------------------------- recovery ----
def upright(s):
    """1 when the base z axis points up, 0 when upside down."""
    return 0.5 * (1.0 - s["projected_gravity"][:, 2])


def recovery_height(s):
    up = upright(s) > 0.9
    return np.exp(-((s["base_height"] - s["base_height_target"]) ** 2) / 0.005) * up


def recovery_pose(s):
    up = upright(s) > 0.9
    return np.exp(-np.sum((s["dof_pos"] - s["default_dof_pos"]) ** 2, axis=1) / 1.0) * up


REWARD_FUNCTIONS = {f.__name__: f for f in [
    tracking_lin_vel, tracking_ang_vel, lin_vel_z, ang_vel_xy, orientation, base_height,
    torques, dof_vel, dof_acc, action_rate, feet_air_time, collision, dof_pos_limits,
    stand_still, termination, upright, recovery_height, recovery_pose,
]}


class RewardManager:
    def __init__(self, scales: dict[str, float], dt: float):
        unknown = set(scales) - set(REWARD_FUNCTIONS)
        if unknown:
            raise KeyError(f"unknown reward terms: {sorted(unknown)}")
        self.scales = {k: float(v) * dt for k, v in scales.items() if float(v) != 0.0}
        self.names = list(self.scales)

    def compute(self, state: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        n = state["base_height"].shape[0]
        total = np.zeros(n)
        terms = {}
        for name, scale in self.scales.items():
            r = REWARD_FUNCTIONS[name](state) * scale
            terms[name] = r
            total += r
        return total, terms
