"""Observation layout shared by the simulator AND the real-robot runtime.

Keeping a single implementation is one of the cheapest sim-to-real wins: any
scaling / ordering mismatch between training and deployment silently breaks a
policy. Both ``dog2real.sim.env`` and ``dog2real.deploy.runtime`` call
:func:`build_actor_obs`.

Actor observation (45 dims), in order::

    projected gravity (3)        body-frame unit gravity vector, from the IMU
    base angular velocity (3)    gyro, rad/s            * ang_vel scale
    commands (3)                 vx, vy, yaw rate        * command scales
    joint pos - default (12)     rad                     * dof_pos scale
    joint vel (12)               rad/s                   * dof_vel scale
    last action (12)             raw policy output (clipped)
"""

from __future__ import annotations

import numpy as np

DEFAULT_OBS_SCALES = {
    "ang_vel": 0.25,
    "commands": [2.0, 2.0, 0.25],
    "dof_pos": 1.0,
    "dof_vel": 0.05,
    "lin_vel": 2.0,
}

ACTOR_OBS_SLICES = {
    "projected_gravity": slice(0, 3),
    "ang_vel": slice(3, 6),
    "commands": slice(6, 9),
    "dof_pos": slice(9, 21),
    "dof_vel": slice(21, 33),
    "last_action": slice(33, 45),
}
ACTOR_OBS_DIM = 45


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion(s) (w, x, y, z) -> rotation matrix (..., 3, 3) body-to-world."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def world_to_body(q: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector(s) into the body frame: R^T v."""
    R = quat_to_rotmat(q)
    return np.einsum("...ji,...j->...i", R, v_world)


def projected_gravity_from_quat(q: np.ndarray) -> np.ndarray:
    g = np.broadcast_to(np.array([0.0, 0.0, -1.0]), np.asarray(q).shape[:-1] + (3,))
    return world_to_body(q, g)


def build_actor_obs(projected_gravity, ang_vel, commands, dof_pos_rel, dof_vel, last_action,
                    scales: dict | None = None) -> np.ndarray:
    """Concatenate (and scale) the actor observation. Works batched (N, .) or single (.)."""
    s = {**DEFAULT_OBS_SCALES, **(scales or {})}
    parts = [
        np.asarray(projected_gravity, dtype=np.float64),
        np.asarray(ang_vel, dtype=np.float64) * s["ang_vel"],
        np.asarray(commands, dtype=np.float64) * np.asarray(s["commands"], dtype=np.float64),
        np.asarray(dof_pos_rel, dtype=np.float64) * s["dof_pos"],
        np.asarray(dof_vel, dtype=np.float64) * s["dof_vel"],
        np.asarray(last_action, dtype=np.float64),
    ]
    return np.concatenate(parts, axis=-1).astype(np.float32)
