"""On-policy training loop: rollout collection, PPO updates, logging, checkpoints."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from dog2real.rl.logger import Logger
from dog2real.rl.networks import ActorCritic
from dog2real.rl.normalizer import EmpiricalNormalization
from dog2real.rl.ppo import PPO


class OnPolicyRunner:
    def __init__(self, env, cfg: dict, log_dir: str | Path | None = None, device: str = "cpu"):
        self.env = env
        self.cfg = cfg
        self.device = device
        pcfg = cfg.get("policy", {})
        ppo_cfg = dict(cfg.get("ppo", {}))
        self.steps_per_env = int(ppo_cfg.pop("steps_per_env", 24))
        self.asymmetric = bool(pcfg.get("asymmetric_critic", True))
        critic_dim = env.num_privileged_obs if self.asymmetric else env.num_obs
        self.policy = ActorCritic(env.num_obs, critic_dim, env.num_actions,
                                  pcfg.get("actor_hidden", [256, 128, 64]),
                                  pcfg.get("critic_hidden", [256, 128, 64]),
                                  pcfg.get("activation", "elu"), pcfg.get("init_noise_std", 1.0))
        self.obs_norm = EmpiricalNormalization(env.num_obs).to(device)
        self.critic_norm = EmpiricalNormalization(critic_dim).to(device)
        self.alg = PPO(self.policy, device=device, seed=int(cfg.get("seed", 0)), **ppo_cfg)
        self.alg.init_storage(self.steps_per_env, env.num_envs, env.num_obs, critic_dim, env.num_actions)
        self.log_dir = Path(log_dir) if log_dir else None
        rcfg = cfg.get("runner", {})
        self.logger = Logger(self.log_dir, use_tensorboard=rcfg.get("tensorboard", True))
        self.save_interval = int(rcfg.get("save_interval", 50))
        self.iteration = 0
        self.history: list[dict] = []

    def _t(self, x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def learn(self, iterations: int, verbose: bool = True) -> list[dict]:
        obs, priv = self.env.reset()
        critic = priv if self.asymmetric else obs
        n = self.env.num_envs
        ep_ret, ep_len = np.zeros(n), np.zeros(n)
        ret_buf, len_buf = deque(maxlen=100), deque(maxlen=100)
        start_it = self.iteration
        for it in range(start_it, start_it + iterations):
            t0 = time.time()
            ep_infos = []
            step_rewards = []
            self.policy.eval()
            for _ in range(self.steps_per_env):
                o, c = self._t(obs), self._t(critic)
                self.obs_norm.update(o)
                self.critic_norm.update(c)
                a = self.alg.act(self.obs_norm(o), self.critic_norm(c))
                obs, rew, done, info = self.env.step(a.cpu().numpy())
                critic = info["privileged_obs"] if self.asymmetric else obs
                self.alg.process_env_step(self._t(rew), self._t(done), self._t(info["time_outs"]))
                if "episode" in info:
                    ep_infos.append(info["episode"])
                step_rewards.append(float(np.mean(rew)))
                ep_ret += rew
                ep_len += 1
                for i in np.nonzero(done)[0]:
                    ret_buf.append(ep_ret[i])
                    len_buf.append(ep_len[i])
                ep_ret[done] = 0.0
                ep_len[done] = 0
            collect_time = time.time() - t0
            self.alg.compute_returns(self.critic_norm(self._t(critic)))
            self.policy.train()
            t1 = time.time()
            stats = self.alg.update()
            learn_time = time.time() - t1

            steps = self.steps_per_env * n
            stats["perf/fps"] = steps / (collect_time + learn_time)
            stats["perf/collect_s"] = collect_time
            stats["perf/learn_s"] = learn_time
            stats["train/mean_step_reward"] = float(np.mean(step_rewards))
            if ret_buf:
                stats["train/mean_return"] = float(np.mean(ret_buf))
                stats["train/mean_episode_length"] = float(np.mean(len_buf))
            keys = sorted({k for e in ep_infos for k in e})
            for k in keys:
                stats[f"episode/{k}"] = float(np.mean([e[k] for e in ep_infos if k in e]))
            self.iteration = it + 1
            self.history.append(stats)
            self.logger.log(self.iteration, stats)
            if verbose:
                msg = (f"it {self.iteration:5d} | fps {stats['perf/fps']:6.0f} "
                       f"| step_rew {stats['train/mean_step_reward']:+.4f}")
                if "train/mean_return" in stats:
                    msg += (f" | ep_ret {stats['train/mean_return']:+8.3f}"
                            f" | ep_len {stats['train/mean_episode_length']:6.1f}")
                msg += (f" | v_loss {stats['loss/value']:.4f} | kl {stats['policy/kl']:.4f}"
                        f" | lr {stats['policy/lr']:.1e} | std {stats['policy/noise_std']:.2f}")
                print(msg, flush=True)
            if self.log_dir and (self.iteration % self.save_interval == 0):
                self.save(self.log_dir / f"model_{self.iteration}.pt")
        if self.log_dir:
            self.save(self.log_dir / "model_last.pt")
        self.logger.close()
        return self.history

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "policy": self.policy.state_dict(),
            "obs_norm": self.obs_norm.state_dict(),
            "critic_norm": self.critic_norm.state_dict(),
            "optimizer": self.alg.optimizer.state_dict(),
            "learning_rate": self.alg.learning_rate,
            "iteration": self.iteration,
            "cfg": self.cfg,
            "dims": {"obs": self.env.num_obs, "critic": int(self.critic_norm.mean.numel()),
                     "act": self.env.num_actions},
        }
        if hasattr(self.env, "curriculum"):
            ckpt["curriculum"] = self.env.curriculum.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str | Path, load_optimizer: bool = True) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.obs_norm.load_state_dict(ckpt["obs_norm"])
        self.critic_norm.load_state_dict(ckpt["critic_norm"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(ckpt["optimizer"])
            self.alg._set_lr(ckpt.get("learning_rate", self.alg.learning_rate))
        self.iteration = int(ckpt.get("iteration", 0))
        if "curriculum" in ckpt and hasattr(self.env, "curriculum"):
            self.env.curriculum.load_state_dict(ckpt["curriculum"])
