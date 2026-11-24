# Sim-to-real techniques used in dog2real

The reality gap has three parts: **dynamics** (masses, friction, motors),
**sensing** (noise, bias, latency) and **software** (a different code path on
the robot). Each technique below targets one of them; the config keys are in
`configs/train_walk.yaml`.

## 1. One observation function for sim and robot (software gap)

`dog2real/observation.py::build_actor_obs` is called by the simulator *and* by
the real-robot runtime. The exported ONNX/TorchScript file contains the
observation normaliser, so the robot never needs the training statistics.
`scripts/run_robot.py` refuses to start if `policy_dt` / `action_scale` in
`deploy.yaml` disagree with the metadata stored next to the exported policy.

## 2. Only deployable observations for the actor

The actor sees only what the robot measures: IMU gravity direction and gyro,
joint encoders, the command and its own last action. **Base linear velocity is
not observed** (it is very hard to estimate on a DIY robot). The critic is
*asymmetric*: it also receives true base velocity, base height, foot contacts,
friction, payload, CoM offset, gain and strength multipliers -- this makes the
value function far easier to learn without leaking anything into the deployed
network.

## 3. Domain randomisation (dynamics gap)

Sampled per environment at every reset (`dog2real/sim/randomization.py`):

| Parameter | Default range | Why |
|---|---|---|
| ground friction | 0.4 - 1.25 | floors, rubber feet wear |
| added base mass | -0.5 - +1.5 kg | battery, cables, payload, mis-weighing |
| base CoM offset | +-3 / 2 / 2 cm | CoM never where CAD says |
| leg link mass scale | 0.9 - 1.1 | |
| motor strength | 0.85 - 1.1 per motor | battery sag, temperature, unit variance |
| Kp / Kd error | 0.85-1.15 / 0.8-1.25 per motor | firmware gain scaling, quantisation |
| actuation latency | 0 - 16 ms | CAN round trip, scheduling |
| pushes | +-0.5 m/s every 8 s | robustness, recovery |

Latency is implemented as a delay buffer on the position target at the physics
rate: for the first `latency_substeps` physics steps of each policy step the
motors still track the previous target -- the same thing that happens on the
real bus.

## 4. Actuator model (dynamics gap)

The PD law runs at the physics rate (500 Hz) like a motor driver, and the
requested torque passes through a **DC-motor torque-speed envelope**
(`DCMotorModel`): available torque falls linearly with speed while motoring,
saturates at the peak torque, and is scaled by the randomised motor strength.
Without this, policies learn fast, high-torque motions that a real motor
cannot deliver. Joint `armature` (reflected rotor inertia), `damping` and
`frictionloss` are identified on the bench (`scripts/motor_sysid.py`).

Optional: a learned actuator network (Hwangbo et al., *Science Robotics* 2019)
trained on logged motor data (`scripts/train_actuator_net.py`) can replace the
analytic model (`actuator.type: actuator_net`).

## 5. Sensor noise (sensing gap)

* IMU (`IMUNoiseModel`): constant per-episode gyro bias + white noise; the
  gravity estimate gets a per-episode tilt error (attitude filter bias) + noise.
* Encoders (`ObservationNoise`): uniform position noise (quantisation /
  backlash) and large velocity noise (finite-difference velocity on cheap drives).

## 6. Smoothness and energy regularisation

Penalties on torque, joint acceleration, action rate and a foot air-time reward
produce low-frequency, low-torque gaits that survive unmodelled compliance and
backlash. The safety layer additionally rate-limits targets at deployment, so
the policy must not rely on step changes.

## 7. Curriculum

The command range starts small (`commands.init_ranges`) and grows by
`commands.step` whenever the episode-mean velocity-tracking reward of finished
episodes exceeds 80 % of its maximum, up to `commands.max_ranges`.

## 8. Sim-in-the-loop validation

`scripts/run_robot.py --sim` runs the *deployment* stack (HAL, safety layer,
start-up ramp, ONNX inference, timing loop) against MuJoCo, with emulated
MIT-mode motors and command latency. Bugs in calibration-independent code are
caught here instead of on the robot.

## What is not modelled (known gaps)

* Uneven terrain (flat plane only; add a heightfield + terrain curriculum).
* Gear backlash and joint compliance beyond `armature` / `damping`.
* Battery voltage dynamics (only covered by motor strength randomisation).
* Foot slip on soft surfaces / carpet deformation.
