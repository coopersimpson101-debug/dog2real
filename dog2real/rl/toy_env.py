"""A tiny vectorised continuous-control task used to unit-test PPO.

The agent observes its 2-D position ``x`` and moves by ``0.1 * clip(a, -1, 1)``;
reward ``-|x|^2``. A zero / random policy scores about -0.67 per step, a good
policy drives ``x`` to the origin within ~10 steps and scores > -0.15 per step.
"""

from __future__ import annotations

import numpy as np


class PointMassEnv:
    def __init__(self, num_envs: int = 32, episode_length: int = 30, seed: int = 0):
        self.num_envs = num_envs
        self.num_obs = 2
        self.num_privileged_obs = 2
        self.num_actions = 2
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)
        self.x = np.zeros((num_envs, 2))
        self.t = np.zeros(num_envs, dtype=np.int64)

    def reset(self):
        self.x = self.rng.uniform(-1, 1, (self.num_envs, 2))
        self.t[:] = 0
        return self.x.astype(np.float32), self.x.astype(np.float32)

    def step(self, actions):
        self.x = self.x + 0.1 * np.clip(actions, -1, 1)
        rew = -np.sum(self.x ** 2, axis=1)
        self.t += 1
        done = self.t >= self.episode_length
        ids = np.nonzero(done)[0]
        self.x[ids] = self.rng.uniform(-1, 1, (ids.size, 2))
        self.t[ids] = 0
        obs = self.x.astype(np.float32)
        return obs, rew.astype(np.float32), done, {"time_outs": done.copy(), "privileged_obs": obs}
