"""Roll out a checkpoint in simulation, print tracking metrics and (optionally) record an mp4.

    python scripts/play.py --checkpoint runs/walk_xxx/model_last.pt --video media/sim_walk.mp4 --command 0.5 0 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dog2real.deploy.export import load_deploy_policy  # noqa: E402
from dog2real.sim.env import VecQuadrupedEnv  # noqa: E402
from dog2real.video import write_video  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "YAW"))
    p.add_argument("--video")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--no-randomization", action="store_true", help="nominal physics, no noise, no pushes")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    policy, cfg = load_deploy_policy(args.checkpoint)
    if args.no_randomization:
        cfg["domain_rand"]["enabled"] = False
        cfg["imu_noise"]["enabled"] = False
        cfg["obs_noise"]["enabled"] = False
    env = VecQuadrupedEnv(cfg, num_envs=1, seed=args.seed)
    obs, _ = env.reset()
    frames, vel_err, falls, ret = [], [], 0, 0.0
    steps = int(args.seconds / env.dt)
    for _ in range(steps):
        if args.command is not None:
            env.commands[:] = args.command
            obs, _ = env.compute_observations()
        with torch.no_grad():
            act = policy(torch.as_tensor(obs)).numpy()
        obs, rew, done, info = env.step(act)
        ret += float(rew[0])
        s = env._state
        vel_err.append(np.linalg.norm(s["commands"][0, :2] - s["base_lin_vel"][0, :2]))
        falls += int(info["terminated"][0])
        if args.video:
            frames.append(env.render(0, args.width, args.height))
    print(f"return {ret:.3f} | mean |v_cmd - v| {np.mean(vel_err):.3f} m/s | falls {falls} in {args.seconds:.0f}s")
    if args.video:
        out = write_video(frames, Path(args.video), fps=int(round(1 / env.dt)))
        print(f"wrote {out}")
    env.close()


if __name__ == "__main__":
    main()
