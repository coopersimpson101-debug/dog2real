"""Train a learned actuator model (MLP) from logged motor data.

Input features: the last ``history`` samples of position error ``q* - q`` and
velocity ``dq``; target: measured joint torque. The trained weights are saved
in the numpy format read by :class:`dog2real.sim.actuator.ActuatorNet`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dog2real.sim.actuator import ActuatorNet


def make_dataset(q_des, q, dq, tau, history: int):
    err = np.asarray(q_des) - np.asarray(q)
    dq = np.asarray(dq)
    n = len(err) - history + 1
    E = np.stack([err[i:i + n] for i in range(history)], axis=1)
    V = np.stack([dq[i:i + n] for i in range(history)], axis=1)
    y = np.asarray(tau)[history - 1:]
    return E, V, y


def train_actuator_net(q_des, q, dq, tau, history: int = 3, hidden=(32, 32), epochs: int = 300,
                       lr: float = 3e-3, pos_scale: float = 1.0, vel_scale: float = 0.1, seed: int = 0,
                       torque_limit: float = np.inf) -> tuple[ActuatorNet, float]:
    torch.manual_seed(seed)
    E, V, y = make_dataset(q_des, q, dq, tau, history)
    X = np.concatenate([E * pos_scale, V * vel_scale], axis=1)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)[:, None]
    layers, last = [], X.shape[1]
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ELU()]
        last = h
    layers.append(nn.Linear(last, 1))
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 1024):
            idx = perm[i:i + 1024]
            loss = (net(Xt[idx]) - yt[idx]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        rmse = float((net(Xt) - yt).pow(2).mean().sqrt())
    weights = [(m.weight.detach().numpy().astype(np.float64), m.bias.detach().numpy().astype(np.float64))
               for m in net if isinstance(m, nn.Linear)]
    return ActuatorNet(weights, history, pos_scale, vel_scale, torque_limit), rmse
