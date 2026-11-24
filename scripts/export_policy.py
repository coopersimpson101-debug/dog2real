"""Export a checkpoint to TorchScript + ONNX and verify both against PyTorch.

    python scripts/export_policy.py --checkpoint runs/walk_xxx/model_last.pt --out-dir exported
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dog2real.deploy.export import PolicyRunner, export_policy, load_deploy_policy  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", default="exported")
    p.add_argument("--name", default="policy")
    args = p.parse_args()

    policy, cfg = load_deploy_policy(args.checkpoint)
    paths = export_policy(policy, args.out_dir, cfg, args.name)
    obs = np.random.default_rng(0).normal(size=(64, policy.normalizer.mean.numel())).astype(np.float32)
    with torch.no_grad():
        ref = policy(torch.from_numpy(obs)).numpy()
    for key in ("torchscript", "onnx"):
        try:
            runner = PolicyRunner(paths[key], backend="onnx" if key == "onnx" else "torchscript")
        except ImportError:
            print(f"{key}: runtime not installed, skipped check")
            continue
        out = np.stack([runner(o) for o in obs])
        print(f"{key:12s} {paths[key]}  max |diff| vs torch = {np.abs(out - ref).max():.2e}")
    print(f"metadata     {paths['meta']}")


if __name__ == "__main__":
    main()
