"""Run an exported policy on the robot -- or on MuJoCo through the exact same runtime.

Sim-in-the-loop (do this first, on your laptop or on the Pi itself):

    python scripts/run_robot.py --policy exported/policy.onnx --sim --duration 10 --video media/sitl.mp4

Real robot (robot on a stand, e-stop in hand, low max_torque in configs/deploy.yaml):

    sudo ip link set can0 up type can bitrate 1000000
    python scripts/run_robot.py --policy exported/policy.onnx --duration 20 --log logs/real_run.npz

Without ``--policy`` the runtime only performs the stand-up ramp and holds the
default pose (a good first hardware test).
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from dog2real.config import load_yaml  # noqa: E402
from dog2real.deploy.runtime import RealClock, RobotRuntime, SimClock  # noqa: E402
from dog2real.deploy.safety import SafetyConfig, SafetyLayer  # noqa: E402


class HoldPolicy:
    """Outputs zero actions = hold the default pose."""

    def __call__(self, obs):
        return np.zeros(12)


def build_hardware(dcfg: dict):
    from dog2real.deploy.hal.imu import Mpu6050IMU
    from dog2real.deploy.hal.mit_can import MitCanMotorBus

    mcfg = dcfg["motors"]
    if mcfg.get("driver", "mit_can") != "mit_can":
        raise NotImplementedError(f"motor driver {mcfg['driver']!r}: implement dog2real.deploy.hal.base.MotorBus")
    bus = MitCanMotorBus.from_config(mcfg)
    icfg = dcfg["imu"]
    if icfg.get("driver", "mpu6050") != "mpu6050":
        raise NotImplementedError(f"IMU driver {icfg['driver']!r}: implement dog2real.deploy.hal.base.IMU")
    imu = Mpu6050IMU(icfg.get("i2c_bus", 1), int(icfg.get("address", 0x68)), icfg.get("rotation"),
                     filter_gain=float(icfg.get("filter_gain", 0.02)))
    print("calibrating gyro, keep the robot still...")
    print("gyro bias:", imu.calibrate_gyro(2.0))
    return bus, imu


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/deploy.yaml")
    p.add_argument("--policy", help="exported .onnx or .pt; omitted = stand and hold")
    p.add_argument("--sim", action="store_true", help="sim-in-the-loop: drive MuJoCo instead of hardware")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "YAW"))
    p.add_argument("--log", help="save per-tick log (.npz)")
    p.add_argument("--video", help="(sim only) record an mp4 of the run")
    args = p.parse_args()

    dcfg = load_yaml(args.config)
    robot = load_yaml(dcfg.get("robot", "robot.yaml"))
    ctrl = dcfg["control"]
    if args.policy:
        meta_path = Path(args.policy).with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            for key in ("policy_dt", "action_scale"):
                if key in meta and abs(float(meta[key]) - float(ctrl[key])) > 1e-9:
                    sys.exit(f"deploy.yaml control.{key}={ctrl[key]} does not match the trained policy ({meta[key]})")
        from dog2real.deploy.export import PolicyRunner

        policy = PolicyRunner(args.policy)
        print(f"policy backend: {policy.backend}")
    else:
        policy = HoldPolicy()

    safety = SafetyLayer(SafetyConfig.from_dicts(robot, dcfg.get("safety")))
    period = float(ctrl["policy_dt"])
    if args.sim:
        from dog2real.deploy.hal.sim_backend import SimIMUView, SimRobot

        scfg = dcfg.get("sim", {})
        sim = SimRobot(robot, latency_s=float(scfg.get("latency_s", 0.004)),
                       start_pose=scfg.get("start_pose", "lying"), gyro_noise=float(scfg.get("gyro_noise", 0.0)))
        bus, imu, clock = sim, SimIMUView(sim), SimClock(period, sim)
    else:
        bus, imu = build_hardware(dcfg)
        clock = RealClock(period)
        pin = (dcfg.get("estop") or {}).get("gpio_pin")
        if pin is not None:
            from gpiozero import Button  # normally-closed button: released -> e-stop

            button = Button(int(pin))
            button.when_released = lambda: safety.estop("hardware e-stop")

    signal.signal(signal.SIGINT, lambda *_: safety.estop("SIGINT"))
    command = args.command if args.command is not None else ctrl.get("command", [0.0, 0.0, 0.0])
    rt = RobotRuntime(robot, dcfg, policy, bus, imu, clock, safety, command, log_path=args.log)

    frames = []
    if args.video and args.sim:
        orig_tick = rt.tick

        def tick_and_render():
            entry = orig_tick()
            frames.append(sim.render())
            return entry

        rt.tick = tick_and_render

    ticks = int(args.duration / period)
    rt.start()
    try:
        while rt.tick_count < ticks and safety.ok:
            rt.tick()
            clock.wait()
    finally:
        fault = safety.fault
        height_before_shutdown = sim.base_height() if args.sim else None
        rt.shutdown()
    stats = rt.timing.summary()
    print("timing:", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()}))
    print("safety fault:", fault or "none", "| target clamp events:", safety.clamp_events)
    if args.sim:
        print(f"base height before shutdown {height_before_shutdown:.3f} m (sim)")
    if frames:
        from dog2real.video import write_video

        print("wrote", write_video(frames, Path(args.video), fps=int(round(1 / period))))


if __name__ == "__main__":
    main()
