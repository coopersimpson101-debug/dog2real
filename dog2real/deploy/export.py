"""Export a trained actor (+ its observation normaliser) to TorchScript and ONNX.

The exported graph maps the *raw* 45-D observation built by
``dog2real.observation.build_actor_obs`` to the raw action, so the robot side
needs no knowledge of the normaliser statistics.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dog2real.rl.networks import ActorCritic
from dog2real.rl.normalizer import EmpiricalNormalization


class DeployPolicy(nn.Module):
    def __init__(self, normalizer: EmpiricalNormalization, actor: nn.Module, clip_actions: float):
        super().__init__()
        self.normalizer = normalizer
        self.actor = actor
        self.clip_actions = float(clip_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        a = self.actor(self.normalizer(obs))
        return torch.clamp(a, -self.clip_actions, self.clip_actions)


def load_deploy_policy(checkpoint: str | Path) -> tuple[DeployPolicy, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    dims = ckpt["dims"]
    pcfg = cfg.get("policy", {})
    ac = ActorCritic(dims["obs"], dims["critic"], dims["act"], pcfg.get("actor_hidden", [256, 128, 64]),
                     pcfg.get("critic_hidden", [256, 128, 64]), pcfg.get("activation", "elu"))
    ac.load_state_dict(ckpt["policy"])
    norm = EmpiricalNormalization(dims["obs"])
    norm.load_state_dict(ckpt["obs_norm"])
    policy = DeployPolicy(norm, ac.actor, cfg.get("env", {}).get("clip_actions", 10.0)).eval()
    return policy, cfg


def export_policy(policy: DeployPolicy, out_dir: str | Path, cfg: dict | None = None,
                  name: str = "policy") -> dict[str, Path]:
    """Write ``<name>.pt`` (TorchScript), ``<name>.onnx`` and ``<name>.json`` (metadata)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_dim = policy.normalizer.mean.numel()
    example = torch.zeros(1, obs_dim)
    paths = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # torch>=2.9 deprecates TorchScript; still the
        scripted = torch.jit.script(policy)              # simplest C++/Python-loadable format
    paths["torchscript"] = out_dir / f"{name}.pt"
    scripted.save(str(paths["torchscript"]))

    paths["onnx"] = out_dir / f"{name}.onnx"
    export_kwargs = dict(input_names=["obs"], output_names=["action"],
                         dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(policy, (example,), str(paths["onnx"]), dynamo=False, **export_kwargs)
    except TypeError:  # older torch without the dynamo kwarg
        torch.onnx.export(policy, (example,), str(paths["onnx"]), **export_kwargs)

    meta = {"obs_dim": int(obs_dim), "act_dim": int(policy.actor[-1].out_features),
            "clip_actions": policy.clip_actions}
    if cfg is not None:
        env = cfg.get("env", {})
        robot = cfg.get("robot", {})
        meta.update({
            "task": cfg.get("task", "walk"),
            "policy_dt": float(env.get("sim_dt", 0.002)) * int(env.get("decimation", 10)),
            "action_scale": float(env.get("action_scale", 0.25)),
            "obs_scales": env.get("obs_scales", {}),
            "kp": robot.get("pd", {}).get("kp"), "kd": robot.get("pd", {}).get("kd"),
        })
    paths["meta"] = out_dir / f"{name}.json"
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths


class PolicyRunner:
    """Loads an exported policy for inference: ONNX Runtime if available, else TorchScript."""

    def __init__(self, path: str | Path, backend: str = "auto"):
        path = Path(path)
        self.backend = None
        if backend in ("auto", "onnx") and path.suffix == ".onnx":
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
                self.backend = "onnx"
            except ImportError:
                if backend == "onnx":
                    raise
                path = path.with_suffix(".pt")
        if self.backend is None:
            torch.set_num_threads(1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                self.module = torch.jit.load(str(path), map_location="cpu").eval()
            self.backend = "torchscript"

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if self.backend == "onnx":
            return self.sess.run(None, {"obs": x})[0][0]
        with torch.no_grad():
            return self.module(torch.from_numpy(x)).numpy()[0]
