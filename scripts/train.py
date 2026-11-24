"""Train a walking or get-up policy with the from-scratch PPO.

    python scripts/train.py --config configs/train_walk.yaml
    python scripts/train.py --config configs/train_recovery.yaml --num-envs 256
    python scripts/train.py --config configs/smoke_test.yaml          # 3-iteration smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import yaml  # noqa: E402

from dog2real.config import load_train_config  # noqa: E402
from dog2real.rl.runner import OnPolicyRunner  # noqa: E402
from dog2real.sim.env import VecQuadrupedEnv  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/train_walk.yaml")
    p.add_argument("--num-envs", type=int)
    p.add_argument("--iterations", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--log-dir")
    p.add_argument("--resume", help="checkpoint to resume from")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    args = p.parse_args()

    cfg = load_train_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.num_envs:
        cfg["env"]["num_envs"] = args.num_envs
    iterations = args.iterations or int(cfg["runner"]["iterations"])
    if args.threads:
        torch.set_num_threads(args.threads)
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)

    log_dir = Path(args.log_dir or Path(cfg["runner"].get("log_dir", "runs")) /
                   f"{cfg.get('task', 'walk')}_{time.strftime('%Y%m%d_%H%M%S')}")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    env = VecQuadrupedEnv(cfg, seed=seed)
    runner = OnPolicyRunner(env, cfg, log_dir=log_dir, device=args.device)
    if args.resume:
        runner.load(args.resume)
    print(f"task={cfg.get('task')} envs={env.num_envs} obs={env.num_obs} critic_obs={env.num_privileged_obs} "
          f"policy_dt={env.dt:.3f}s physics_dt={env.sim_dt:.4f}s -> logging to {log_dir}")
    t0 = time.time()
    runner.learn(iterations)
    print(f"done in {time.time() - t0:.1f}s; checkpoint: {log_dir / 'model_last.pt'}")


if __name__ == "__main__":
    main()
