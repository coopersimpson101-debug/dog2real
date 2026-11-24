"""Identify armature / damping / Coulomb friction of ONE motor from a chirp experiment.

Bench procedure (real): motor clamped to the bench, NOTHING attached to the output
(or only a link with known inertia), CAN wired, e-stop ready.

    python scripts/motor_sysid.py --joint FL_knee --kp 5 --kd 0.1 --amplitude 1.0 --out logs/sysid_FL_knee.npz

Validate the pipeline without hardware (simulated motor with known parameters):

    python scripts/motor_sysid.py --sim

Paste the printed values into configs/robot.yaml (joints.armature / damping / frictionloss).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from dog2real.config import load_yaml  # noqa: E402
from dog2real.sysid import chirp, fit_joint_dynamics, simulate_bench_motor  # noqa: E402


def run_hardware(args) -> dict:
    from dog2real.deploy.hal.mit_can import MitCanMotorBus, MotorLimits, MotorSpec

    dcfg = load_yaml(args.config)
    mcfg = dcfg["motors"]
    joints = [j for j in mcfg["joints"] if j["name"] == args.joint]
    if not joints:
        sys.exit(f"joint {args.joint} not in {args.config}")
    j = joints[0]
    spec = MotorSpec(j["name"], int(j["can_id"]), MotorLimits.preset(j.get("model", mcfg.get("model", "AK80-9"))),
                     float(j.get("sign", 1.0)), float(j.get("offset", 0.0)), float(j.get("gear", 1.0)))
    bus = MitCanMotorBus([spec], channel=mcfg.get("channel", "can0"), interface=mcfg.get("interface", "socketcan"),
                         bitrate=int(mcfg.get("bitrate", 1_000_000)))
    dt = 1.0 / args.rate
    n = int(args.duration * args.rate)
    log = {k: np.zeros(n) for k in ("t", "q_des", "q", "dq", "tau")}
    bus.enable()
    q0 = bus.read().q[0]
    try:
        t_start = time.perf_counter()
        for k in range(n):
            t = k * dt
            q_des = q0 + float(chirp(np.array(t), args.f0, args.f1, args.duration, args.amplitude))
            bus.command(np.array([q_des]), np.zeros(1), np.array([args.kp]), np.array([args.kd]), np.zeros(1))
            s = bus.read()
            if not s.comm_ok:
                raise RuntimeError("motor stopped replying")
            for key, v in (("t", t), ("q_des", q_des), ("q", s.q[0]), ("dq", s.dq[0]), ("tau", s.tau[0])):
                log[key][k] = v
            while time.perf_counter() < t_start + (k + 1) * dt:
                pass
    finally:
        bus.close()
    return log


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/deploy.yaml")
    p.add_argument("--joint", default="FL_knee")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--kp", type=float, default=5.0)
    p.add_argument("--kd", type=float, default=0.1)
    p.add_argument("--amplitude", type=float, default=1.0)
    p.add_argument("--f0", type=float, default=0.2)
    p.add_argument("--f1", type=float, default=4.0)
    p.add_argument("--duration", type=float, default=6.0)
    p.add_argument("--rate", type=float, default=500.0, help="Hz (hardware)")
    p.add_argument("--from-log", help="fit an existing .npz log instead of running an experiment")
    p.add_argument("--out")
    args = p.parse_args()

    if args.from_log:
        log = dict(np.load(args.from_log))
    elif args.sim:
        truth = dict(armature=0.01, damping=0.05, frictionloss=0.1)
        log = simulate_bench_motor(**truth, kp=args.kp, kd=args.kd, duration=args.duration,
                                   amplitude=args.amplitude, f0=args.f0, f1=args.f1, torque_noise=0.02)
        print("simulated motor ground truth:", truth)
    else:
        log = run_hardware(args)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, **log)
    fit = fit_joint_dynamics(log["t"], log["dq"], log["tau"])
    print(f"identified: armature={fit['armature']:.4f} kg m^2  damping={fit['damping']:.4f} Nm s/rad  "
          f"frictionloss={fit['frictionloss']:.4f} Nm  (fit rmse {fit['rmse']:.3f} Nm)")
    print("configs/robot.yaml ->\n  joints:\n"
          f"    damping: {fit['damping']:.4f}\n    armature: {fit['armature']:.4f}\n"
          f"    frictionloss: {fit['frictionloss']:.4f}")


if __name__ == "__main__":
    main()
