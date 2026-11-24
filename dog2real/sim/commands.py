"""Velocity command sampling with a tracking-performance curriculum."""

from __future__ import annotations

import numpy as np


class CommandCurriculum:
    """Samples (vx, vy, yaw_rate) commands; widens the range as tracking improves.

    The curriculum starts from ``init_ranges`` and, whenever the mean normalised
    linear-velocity tracking reward of finished episodes exceeds ``threshold``,
    grows every range by ``step`` (per side) until ``max_ranges`` is reached.
    This mirrors the "game-inspired" curriculum of Rudin et al. (2022) applied
    to the command space only (flat terrain).
    """

    def __init__(self, cfg: dict):
        self.init = {k: np.array(v, dtype=np.float64) for k, v in cfg["init_ranges"].items()}
        self.max = {k: np.array(v, dtype=np.float64) for k, v in cfg.get("max_ranges", cfg["init_ranges"]).items()}
        self.ranges = {k: v.copy() for k, v in self.init.items()}
        self.enabled = bool(cfg.get("curriculum", True))
        self.threshold = float(cfg.get("threshold", 0.8))
        self.step = float(cfg.get("step", 0.1))
        self.zero_prob = float(cfg.get("zero_prob", 0.1))
        self.small_cmd = float(cfg.get("small_cmd", 0.1))
        self.level = 0

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        cmd = np.stack([rng.uniform(*self.ranges[k], size=n) for k in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")], axis=1)
        # a fraction of "stand still" commands; tiny planar commands are zeroed
        cmd[rng.random(n) < self.zero_prob] = 0.0
        small = np.linalg.norm(cmd[:, :2], axis=1) < self.small_cmd
        cmd[small, :2] = 0.0
        return cmd

    def update(self, tracking_score: float) -> bool:
        """Grow the ranges if ``tracking_score`` (0..1) is above threshold. Returns True if grown."""
        if not self.enabled or tracking_score < self.threshold:
            return False
        grown = False
        for k in self.ranges:
            lo = max(self.ranges[k][0] - self.step, self.max[k][0])
            hi = min(self.ranges[k][1] + self.step, self.max[k][1])
            if lo != self.ranges[k][0] or hi != self.ranges[k][1]:
                grown = True
            self.ranges[k] = np.array([lo, hi])
        if grown:
            self.level += 1
        return grown

    def state_dict(self) -> dict:
        return {"ranges": {k: v.tolist() for k, v in self.ranges.items()}, "level": self.level}

    def load_state_dict(self, d: dict) -> None:
        self.ranges = {k: np.array(v) for k, v in d["ranges"].items()}
        self.level = int(d.get("level", 0))
