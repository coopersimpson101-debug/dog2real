# Match the simulation to your robot

A policy only transfers if the simulator is *close enough* to your robot, and
domain randomisation covers what is left. This guide goes from "generic placeholder dog" to
"my dog", in order of how much each step matters. Every number ends up in
[`configs/robot.yaml`](../configs/robot.yaml) or [`configs/deploy.yaml`](../configs/deploy.yaml).

Tools needed: kitchen/postal scale (1 g resolution), calipers or a ruler, a string
or a stand to hang the robot from, a bench clamp for one motor.

---

## 0. Conventions (read first)

* Base frame: **x forward, y left, z up**. The IMU must be aligned with it, or
  you set `imu.rotation` in `deploy.yaml`.
* Leg order **FL, FR, RL, RR**; joint order per leg **abad, hip, knee**.
* Zero pose: **legs straight down**, abad level. Positive hip swings the foot
  *backwards*; the knee is negative when bent (knee pointing backwards).
* If your robot's knees point forward on the rear legs ("><" stance), keep the
  convention and just use different `joints.*.default` and `range` values.

## 1. Geometry (10 minutes)

Measure axis-to-axis distances, not link outer dimensions:

| YAML key | What to measure |
|---|---|
| `leg.hip_offset` `[x, y]` | from the body centre to the abduction axis, front-left leg |
| `leg.abad.length` | lateral distance from the abduction axis to the hip-flexion axis |
| `leg.thigh.length` | hip-flexion axis to knee axis |
| `leg.calf.length` | knee axis to the centre of the foot sphere |
| `leg.foot.radius` | radius of the rubber foot |
| `base.size` | outer box of the body (collision only) |

Check: `python scripts/build_model.py --view` and compare with a photo of the robot
standing on the default pose.

## 2. Masses and centre of mass (20 minutes)

1. Weigh the complete robot (battery in). This is the number that matters most.
2. Weigh one leg's parts if you can (abad housing + motor, thigh with knee motor
   if it is mounted on the thigh, calf + foot). Otherwise estimate: motors are
   most of the mass, so assign each motor's mass to the link it sits on.
3. `base.mass = total - 4 * (abad + thigh + calf + foot)`.
4. CoM: balance the body (legs folded) on a pencil along x and along y. The
   offset from the geometric centre goes into `base.com_offset`. Do not worry
   about a few millimetres -- `domain_rand.base_com_offset` covers +-3 cm/2 cm.

Check: `python scripts/build_model.py` prints the total mass; it must match the scale.

## 3. Motors: limits, torque-speed, friction (1 hour, one motor is enough)

Datasheet first, then measure:

| YAML key | Source |
|---|---|
| `motor.torque_limit` | peak torque at the *joint* (after any external gearing) |
| `motor.stall_torque`, `motor.no_load_speed` | datasheet torque-speed curve at your battery voltage, joint side |
| `motor.velocity_limit` | safety limit, below no-load speed |
| `joints.armature` | reflected rotor inertia = J_rotor * gear_ratio^2 |
| `joints.damping`, `joints.frictionloss` | measured (below) |

**Chirp identification** (motor clamped, nothing on the output):

```bash
python scripts/motor_sysid.py --joint FL_knee --kp 5 --kd 0.1 --amplitude 1.0 --out logs/sysid_FL_knee.npz
```

The script sends a 0.2-4 Hz position chirp through the MIT-mode driver, logs
`q, dq, tau` and fits `tau = J ddq + b dq + f_c sign(dq)`. Paste the printed
`armature / damping / frictionloss`. Run `python scripts/motor_sysid.py --sim` to see
the procedure recover known parameters from a simulated motor.

Notes:

* `tau` comes from the motor's current-based torque estimate -- check it
  against a luggage scale on a lever arm once (`tau = F * r`), and correct
  `motor.torque_limit` if the estimate is off by a constant factor.
* **Torque-speed**: command a large constant torque with `kp = kd = 0`, record
  the steady speed, repeat at 3-4 torque levels, then use
  `dog2real.sysid.fit_torque_speed(speed, torque)`. Safety: do this with the motor clamped, never in the leg.
* Optional: record logs at several gains/amplitudes and train a learned
  actuator model: `python scripts/train_actuator_net.py --logs "logs/sysid_*.npz" --out configs/actuator_net.npz`,
  then set `actuator: {type: actuator_net, path: configs/actuator_net.npz}` in the training config.

## 4. PD gains

`pd.kp / pd.kd` in `robot.yaml` must equal `control.kp / kd` in `deploy.yaml`
-- the policy learns *through* these gains. Typical for a 7-12 kg robot: kp 20-40,
kd 0.5-1.0. Check that the robot can hold the standing pose on the stand with
`python scripts/run_robot.py` (no `--policy`: stand-up ramp + hold) before training.

## 5. Latency

Measure the time from "command sent" to "motor reacts": log a step command and
the reply stream at high rate. Typical MIT-CAN loops on a Pi are 2-8 ms. Make sure the
`domain_rand.latency_substeps` range (x 2 ms) brackets your measurement.

## 6. Joint calibration (zero offsets and signs) -- on the robot

1. Put the robot on its back (or in a jig) with the legs in the *zero pose*
   (straight down relative to the body).
2. Either store zero in each motor (`MitCanMotorBus.set_zero()`, only in the jig!)
   or read the motor positions and write them into `motors.joints[].offset`.
3. Signs: move each joint by hand in its *positive* direction and flip `sign`
   until `bus.read().q` increases. Positive directions (right-hand rule, same for
   all four legs): **abad** (+x axis) moves the foot towards the robot's *left*
   (+y); **hip** and **knee** (+y axis) move the foot *backwards*. Verify all 12
   before powering the loop.

## 7. Validate in sim-in-the-loop, then on a stand

```bash
python scripts/run_robot.py --policy exported/policy.onnx --sim --duration 10 --video media/sitl.mp4
```

Then on hardware with the robot hanging (feet just off the ground), low
`safety.max_torque` and the e-stop in hand. Compare the logged joint
trajectories (`--log logs/stand.npz`) against the sim-in-the-loop log of the same
command: large differences in tracking lag point back to step 3 or 5.
