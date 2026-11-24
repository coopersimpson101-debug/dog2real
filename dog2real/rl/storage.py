"""Rollout buffer and Generalised Advantage Estimation."""

from __future__ import annotations

import torch


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                last_values: torch.Tensor, gamma: float, lam: float):
    """GAE(gamma, lambda) over a (T, N) rollout.

    ``dones[t]`` = 1 if the episode ended *after* the transition at ``t`` (so
    ``V(s_{t+1})`` must not be bootstrapped). Time-out bootstrapping is handled
    by the caller by adding ``gamma * V(s_t)`` to the reward at truncation.

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
        A_t     = delta_t + gamma * lam * (1 - d_t) * A_{t+1}
        R_t     = A_t + V(s_t)                      (value target)
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_values)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_v * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        adv[t] = gae
    return adv, adv + values


class RolloutStorage:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, critic_obs_dim: int, act_dim: int,
                 device: str = "cpu"):
        self.T, self.N = num_steps, num_envs
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        self.obs = z(num_steps, num_envs, obs_dim)
        self.critic_obs = z(num_steps, num_envs, critic_obs_dim)
        self.actions = z(num_steps, num_envs, act_dim)
        self.log_probs = z(num_steps, num_envs)
        self.mu = z(num_steps, num_envs, act_dim)
        self.sigma = z(num_steps, num_envs, act_dim)
        self.rewards = z(num_steps, num_envs)
        self.dones = z(num_steps, num_envs)
        self.values = z(num_steps, num_envs)
        self.advantages = z(num_steps, num_envs)
        self.returns = z(num_steps, num_envs)
        self.step = 0

    def add(self, obs, critic_obs, actions, log_probs, mu, sigma, rewards, dones, values):
        t = self.step
        if t >= self.T:
            raise RuntimeError("rollout storage overflow")
        self.obs[t], self.critic_obs[t], self.actions[t] = obs, critic_obs, actions
        self.log_probs[t], self.mu[t], self.sigma[t] = log_probs, mu, sigma
        self.rewards[t], self.dones[t], self.values[t] = rewards, dones, values
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        self.advantages, self.returns = compute_gae(self.rewards, self.values, self.dones, last_values, gamma, lam)

    def minibatches(self, num_minibatches: int, num_epochs: int, generator: torch.Generator | None = None):
        B = self.T * self.N
        mb = B // num_minibatches
        flat = lambda x: x.reshape(B, *x.shape[2:])  # noqa: E731
        data = [flat(x) for x in (self.obs, self.critic_obs, self.actions, self.log_probs, self.mu,
                                  self.sigma, self.values, self.advantages, self.returns)]
        for _ in range(num_epochs):
            perm = torch.randperm(B, generator=generator, device=self.obs.device)
            for i in range(num_minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                yield [x[idx] for x in data]
