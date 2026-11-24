# Code architecture

```
dog2real/
├── __init__.py              joint names / ordering (single source of truth)
├── config.py                YAML loading with `extends:` inheritance
├── observation.py           actor observation layout -- shared by sim AND robot
├── robot/model_builder.py   robot.yaml -> MJCF (12-DoF quadruped), limits, default pose
├── sim/
│   ├── env.py               VecQuadrupedEnv: N x (MjModel, MjData), batched numpy
│   ├── gym_env.py           single-env gymnasium.Env wrapper
│   ├── actuator.py          PD law, DC-motor torque-speed model, ActuatorNet
│   ├── randomization.py     domain randomisation, IMU noise, encoder noise
│   ├── rewards.py           reward terms + RewardManager
│   └── commands.py          velocity commands + curriculum
├── rl/
│   ├── networks.py          Gaussian MLP actor, MLP critic (asymmetric)
│   ├── normalizer.py        running mean/var observation normalisation
│   ├── storage.py           rollout buffer, GAE
│   ├── ppo.py               clipped PPO, value clipping, entropy, KL-adaptive LR
│   ├── runner.py            rollout/update loop, CSV + TensorBoard, checkpoints
│   ├── logger.py
│   └── toy_env.py           point-mass task for PPO unit tests
├── deploy/
│   ├── export.py            TorchScript + ONNX export, PolicyRunner (onnxruntime / torchscript)
│   ├── safety.py            SafetyLayer: limits, rate limit, torque budget, e-stop, tilt cutoff
│   ├── runtime.py           RobotRuntime: 50 Hz loop, start-up ramp, timing stats, RealClock/SimClock
│   └── hal/
│       ├── base.py          MotorBus / IMU interfaces
│       ├── mit_can.py       MIT-mode CAN driver (python-can) + frame packing
│       ├── imu.py           MPU-6050 I2C driver, gravity complementary filter, quaternion adapter
│       └── sim_backend.py   SimRobot: MuJoCo behind the MotorBus/IMU interfaces
├── sysid.py                 least-squares motor identification, bench-motor simulator
├── actuator_net_training.py actuator network training
└── video.py                 mp4/gif writer
```

## Data flow during training

```
             ┌──────────── VecQuadrupedEnv (numpy, N envs) ────────────┐
 actions ──► │ q* = q0 + 0.25 a ─► [delay buffer] ─► PD @500Hz ─► DC motor │ ─► mj_step x10
 (N,12)      │      model ─► torques                                    │
             │ obs  = build_actor_obs(noisy IMU, noisy encoders, ...)   │ ─► obs (N,45)
             │ priv = clean obs + lin vel, height, contacts, DR params  │ ─► critic obs (N,60)
             │ reward = Σ scale_i * term_i * dt ; auto-reset            │ ─► rew, done, time_out
             └──────────────────────────────────────────────────────────┘
                       ▲                                    │
                       │       OnPolicyRunner / PPO         ▼
                 normalise obs ─► actor N(mu, sigma) ; critic V(priv) ─► GAE ─► clipped update
```

## Data flow on the robot

```
 MotorBus.read() ─┐
 IMU.read() ──────┼─► SafetyLayer.check_state ─► build_actor_obs ─► ONNX policy
                  │                                                   │
                  │      q* = q0 + 0.25 a ◄───────────────────────────┘
                  └────► SafetyLayer.filter (limits, rate, torque) ─► MotorBus.command (MIT: q*, kp, kd)
                                                   ▲
                              RealClock (50 Hz) or SimClock (MuJoCo lock-step)
```
