"""Proximal Policy Optimisation (clip variant), written from scratch.

Per minibatch, with ratio ``r = pi(a|s) / pi_old(a|s)`` and normalised advantages ``A``::

    L_clip  = -E[min(r A, clip(r, 1-eps, 1+eps) A)]
    L_value =  E[max((V - R)^2, (V_old + clip(V - V_old, -eps, eps) - R)^2)]   (value clipping)
    L       =  L_clip + c_v L_value - c_e H[pi]

The learning rate is adapted from the KL divergence between the old and new
Gaussian policies (as in rsl_rl): halve-ish if KL > 2 * target, grow if KL < target / 2.
"""

from __future__ import annotations

import torch
from torch import nn

from dog2real.rl.networks import ActorCritic
from dog2real.rl.storage import RolloutStorage


def gaussian_kl(mu_old, sigma_old, mu_new, sigma_new) -> torch.Tensor:
    """KL(old || new) for diagonal Gaussians, summed over action dims -> (B,)."""
    return torch.sum(
        torch.log(sigma_new / sigma_old + 1e-8)
        + (sigma_old.pow(2) + (mu_old - mu_new).pow(2)) / (2.0 * sigma_new.pow(2))
        - 0.5,
        dim=-1,
    )


class PPO:
    def __init__(self, policy: ActorCritic, num_epochs: int = 5, num_minibatches: int = 4,
                 clip_param: float = 0.2, gamma: float = 0.99, lam: float = 0.95,
                 value_loss_coef: float = 1.0, entropy_coef: float = 0.01, learning_rate: float = 1e-3,
                 max_grad_norm: float = 1.0, value_clip: bool = True, schedule: str = "adaptive",
                 desired_kl: float = 0.01, min_lr: float = 1e-5, max_lr: float = 1e-2,
                 device: str = "cpu", seed: int = 0, **_ignored):
        self.policy = policy.to(device)
        self.device = device
        self.num_epochs, self.num_minibatches = num_epochs, num_minibatches
        self.clip_param, self.gamma, self.lam = clip_param, gamma, lam
        self.value_loss_coef, self.entropy_coef = value_loss_coef, entropy_coef
        self.max_grad_norm, self.value_clip = max_grad_norm, value_clip
        self.schedule, self.desired_kl = schedule, desired_kl
        self.min_lr, self.max_lr = min_lr, max_lr
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.storage: RolloutStorage | None = None

    def init_storage(self, num_steps, num_envs, obs_dim, critic_obs_dim, act_dim):
        self.storage = RolloutStorage(num_steps, num_envs, obs_dim, critic_obs_dim, act_dim, self.device)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor):
        a, logp, mu, sigma = self.policy.act(obs)
        v = self.policy.evaluate(critic_obs)
        self._pending = (obs, critic_obs, a, logp, mu, sigma, v)
        return a

    @torch.no_grad()
    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, time_outs: torch.Tensor):
        obs, critic_obs, a, logp, mu, sigma, v = self._pending
        # bootstrap truncated (time-out) episodes: they did not really terminate
        rewards = rewards + self.gamma * v * time_outs.float()
        self.storage.add(obs, critic_obs, a, logp, mu, sigma, rewards, dones.float(), v)

    @torch.no_grad()
    def compute_returns(self, last_critic_obs: torch.Tensor):
        last_v = self.policy.evaluate(last_critic_obs)
        self.storage.compute_returns(last_v, self.gamma, self.lam)

    def _set_lr(self, lr: float):
        self.learning_rate = lr
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def update(self) -> dict[str, float]:
        stats = {"loss/surrogate": 0.0, "loss/value": 0.0, "loss/entropy": 0.0, "loss/total": 0.0,
                 "policy/kl": 0.0, "policy/clip_frac": 0.0}
        n = 0
        for (obs, critic_obs, actions, old_logp, old_mu, old_sigma, old_v, adv, ret) in \
                self.storage.minibatches(self.num_minibatches, self.num_epochs, self.generator):
            dist = self.policy.distribution(obs)
            logp = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            value = self.policy.evaluate(critic_obs)

            with torch.no_grad():
                kl = gaussian_kl(old_mu, old_sigma, dist.mean, dist.stddev).mean()
            if self.schedule == "adaptive" and self.desired_kl > 0:
                if kl > 2.0 * self.desired_kl:
                    self._set_lr(max(self.min_lr, self.learning_rate / 1.5))
                elif 0.0 < kl < self.desired_kl / 2.0:
                    self._set_lr(min(self.max_lr, self.learning_rate * 1.5))

            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = torch.exp(logp - old_logp)
            surr = -torch.min(ratio * adv, torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv).mean()
            if self.value_clip:
                v_clipped = old_v + (value - old_v).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((value - ret).pow(2), (v_clipped - ret).pow(2)).mean()
            else:
                value_loss = (value - ret).pow(2).mean()
            loss = surr + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            stats["loss/surrogate"] += surr.item()
            stats["loss/value"] += value_loss.item()
            stats["loss/entropy"] += entropy.item()
            stats["loss/total"] += loss.item()
            stats["policy/kl"] += kl.item()
            stats["policy/clip_frac"] += ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
            n += 1
        self.storage.clear()
        stats = {k: v / max(n, 1) for k, v in stats.items()}
        stats["policy/lr"] = self.learning_rate
        stats["policy/noise_std"] = self.policy.log_std.exp().mean().item()
        return stats
