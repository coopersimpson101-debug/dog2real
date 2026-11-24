"""Fit a learned actuator model from logged motor data (optional sim-to-real refinement).

    python scripts/train_actuator_net.py --logs "logs/sysid_*.npz" --out configs/actuator_net.npz
    python scripts/train_actuator_net.py --synthetic --out build/actuator_net_demo.npz   # pipeline demo

Each log needs arrays q_des, q, dq, tau sampled at the physics rate (e.g. from
scripts/motor_sysid.py --out). Then set in the training config:

    actuator: {type: actuator_net, path: configs/actuator_net.npz}
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from dog2real.actuator_net_training import train_actuator_net  # noqa: E402
from dog2real.sysid import simulate_bench_motor  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", nargs="*", default=[])
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    logs = []
    if args.synthetic:
        logs.append(simulate_bench_motor(0.01, 0.05, 0.1, kp=20.0, kd=0.5, dt=0.002, torque_noise=0.05))
    for pattern in args.logs:
        logs += [dict(np.load(f)) for f in glob.glob(pattern)]
    if not logs:
        sys.exit("no data: pass --logs or --synthetic")
    cat = {k: np.concatenate([lg[k] for lg in logs]) for k in ("q_des", "q", "dq", "tau")}
    net, rmse = train_actuator_net(cat["q_des"], cat["q"], cat["dq"], cat["tau"], history=args.history,
                                   epochs=args.epochs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    net.save(args.out)
    print(f"trained on {len(cat['tau'])} samples, rmse {rmse:.3f} Nm -> {args.out}")


if __name__ == "__main__":
    main()
