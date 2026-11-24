"""Gaussian MLP actor and MLP critic."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU}


def mlp(inp: int, hidden: list[int], out: int, activation: str = "elu", out_gain: float = 1.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = inp
    for h in hidden:
        lin = nn.Linear(last, h)
        nn.init.orthogonal_(lin.weight, gain=2 ** 0.5)
        nn.init.zeros_(lin.bias)
        layers += [lin, ACTIVATIONS[activation]()]
        last = h
    head = nn.Linear(last, out)
    nn.init.orthogonal_(head.weight, gain=out_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """pi(a|o) = N(mu(o), diag(sigma^2)) with a state-independent learned log-std.

    The critic may take a different (privileged) observation -- an *asymmetric*
    actor-critic. Only the actor is deployed, so the critic is free to use
    simulator-only information such as true base velocity or friction.
    """

    def __init__(self, num_obs: int, num_critic_obs: int, num_actions: int,
                 actor_hidden=(256, 128, 64), critic_hidden=(256, 128, 64),
                 activation: str = "elu", init_noise_std: float = 1.0):
        super().__init__()
        self.actor = mlp(num_obs, list(actor_hidden), num_actions, activation, out_gain=0.01)
        self.critic = mlp(num_critic_obs, list(critic_hidden), 1, activation, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((num_actions,), float(init_noise_std)).log())

    def distribution(self, obs: torch.Tensor) -> Normal:
        mu = self.actor(obs)
        return Normal(mu, self.log_std.exp().expand_as(mu))

    def act(self, obs: torch.Tensor):
        dist = self.distribution(obs)
        a = dist.sample()
        return a, dist.log_prob(a).sum(-1), dist.mean, dist.stddev

    def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)
