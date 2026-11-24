"""Single-environment Gymnasium wrapper around :class:`VecQuadrupedEnv`.

Useful for interop with other tooling (``gymnasium.utils.env_checker``, other
RL libraries for comparison). The from-scratch PPO in ``dog2real.rl`` uses the
batched API directly.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from dog2real.sim.env import VecQuadrupedEnv


class QuadrupedGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, cfg: dict, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode
        self._seed = int(cfg.get("seed", 0))
        self.vec = VecQuadrupedEnv(cfg, num_envs=1, seed=self._seed, auto_reset=False)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.vec.num_obs,), np.float32)
        lim = self.vec.clip_actions
        self.action_space = gym.spaces.Box(-lim, lim, (self.vec.num_actions,), np.float32)
        self.metadata["render_fps"] = int(round(1.0 / self.vec.dt))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.vec.rng = np.random.default_rng(seed)
        obs, priv = self.vec.reset()
        return obs[0], {"privileged_obs": priv[0]}

    def step(self, action):
        obs, rew, done, info = self.vec.step(np.asarray(action, dtype=np.float64)[None])
        terminated = bool(info["terminated"][0])
        truncated = bool(info["time_outs"][0]) and not terminated
        return obs[0], float(rew[0]), terminated, truncated, {"privileged_obs": info["privileged_obs"][0]}

    def render(self):
        if self.render_mode == "rgb_array":
            return self.vec.render(0)
        return None

    def close(self):
        self.vec.close()
