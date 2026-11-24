"""Running mean / variance observation normaliser (Welford / Chan parallel update)."""

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """``y = clip((x - mean) / sqrt(var + eps), -clip, clip)`` with running statistics.

    Statistics are stored as buffers so they are saved in checkpoints and baked
    into exported TorchScript / ONNX policies.
    """

    def __init__(self, shape: int, eps: float = 1e-8, clip: float = 5.0):
        super().__init__()
        self.eps = eps
        self.clip = clip
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(0.0))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).to(self.mean.dtype)
        b_mean = x.mean(0)
        b_var = x.var(0, unbiased=False)
        b_count = float(x.shape[0])
        if b_count == 0:
            return
        tot = self.count + b_count
        delta = b_mean - self.mean
        new_mean = self.mean + delta * b_count / tot
        m_a = self.var * self.count
        m_b = b_var * b_count
        m2 = m_a + m_b + delta.pow(2) * self.count * b_count / tot
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + self.eps), -self.clip, self.clip)
