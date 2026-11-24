# Deployment on a Raspberry Pi / Jetson

## Hardware assumed

* Raspberry Pi 4/5 (64-bit OS) or Jetson Nano/Orin, powered from a separate
  regulator (not from the motor bus).
* CAN: MCP2515/MCP2518FD HAT or a USB-CAN adapter supported by socketcan
  (candleLight / CANable / gs_usb). One bus at 1 Mbit/s handles 12 motors at
  50 Hz with large margin (12 x 2 frames x 50 Hz = 1200 frames/s, about 15 % load); for 200 Hz+ split legs over two buses.
* IMU: MPU-6050/9250 on I2C (driver included), or any AHRS via `QuaternionIMU`.
* A physical **normally-closed e-stop that cuts motor power** (independent of
  software) plus optionally a GPIO line for the software e-stop (`estop.gpio_pin`).

## Software setup

```bash
sudo apt install python3-venv can-utils
python3 -m venv ~/dog2real-venv && source ~/dog2real-venv/bin/activate
pip install numpy pyyaml onnxruntime python-can smbus2 gpiozero
pip install mujoco   # only needed for --sim on the robot computer
# copy the repo (or just dog2real/, configs/, scripts/ and exported/) to the robot
sudo ip link set can0 up type can bitrate 1000000
candump can0          # check that the motors answer
```

PyTorch is **not** needed on the robot when you deploy the ONNX file
(`PolicyRunner` uses onnxruntime; inference of the 45-256-128-64-12 MLP takes
well under 1 ms on a Pi 4).

## Real-time hygiene

* Run the loop with real-time priority: `sudo chrt -f 80 python scripts/run_robot.py ...`
* Isolate a core (`isolcpus=3` in `/boot/firmware/cmdline.txt`) and pin: `taskset -c 3 ...`
* Disable CPU frequency scaling: `sudo cpupower frequency-set -g performance`
* The runtime prints `compute_mean/p99/max` and `jitter_p99`. Overruns > 0 or a
  p99 jitter above ~2 ms mean the loop is not real-time enough.

## Bring-up checklist (do not skip steps)

1. `pytest` passes on the robot computer (or at least `tests/test_mit_can.py`, `tests/test_deploy.py`).
2. One motor on the bench: `python scripts/motor_sysid.py --joint FL_knee` -- the motor follows the chirp.
3. All 12 motors, robot hanging, **no policy**: `python scripts/run_robot.py --duration 10`
   -> the legs ramp to the standing pose and hold. Check every joint moves the right way.
4. Sim-in-the-loop with the policy: `python scripts/run_robot.py --sim --policy exported/policy.onnx`.
5. Policy on hardware, robot hanging, `safety.max_torque` at ~50 % of the motor limit, zero command.
6. On the ground on a soft mat, zero command, then small `--command 0.2 0 0`.
7. Record both: `--video` for sim, a phone video for real, and fill the README table.

## Safety layer behaviour

| Trigger | Result |
|---|---|
| e-stop (GPIO, SIGINT/Ctrl-C, `safety.estop()`) | latched fault -> damping mode |
| body tilt > `tilt_cutoff_rad` (walking) | latched fault -> damping mode |
| any joint speed > `max_joint_vel` | latched fault -> damping mode |
| joint beyond hard limit + tolerance | latched fault -> damping mode |
| a motor silent for > `comm_timeout` | latched fault -> damping mode |
| NaN/inf in sensors or policy output | latched fault -> damping mode |
| target outside soft limits / too fast / PD torque > `max_torque` | target clipped (counted in `clamp_events`) |

Damping mode sends `kp = 0, kd = damping_kd`: the robot sinks slowly instead of
dropping. After `--duration` or a fault the runtime damps for 1 s and then sends
"exit motor mode" to every motor.

## Adding another motor type

Implement `dog2real.deploy.hal.base.MotorBus` (4 methods: `enable`, `disable`,
`command(q_des, dq_des, kp, kd, tau_ff)`, `read() -> JointState`). For position-only
serial bus servos (e.g. Feetech/Dynamixel) emulate the impedance law by sending
`q_des` and setting the servo's internal P/D gains once in `enable()`; then
retrain with `pd.kp/kd` matching the servo's *effective* stiffness (measure it
with a weight on a lever arm). Register the driver in `scripts/run_robot.py::build_hardware`.
