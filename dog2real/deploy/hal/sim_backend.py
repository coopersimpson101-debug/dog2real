"""Sim-in-the-loop backend: MuJoCo pretending to be the robot's hardware.

``SimRobot`` implements :class:`MotorBus` and :class:`IMU` and a lock-step
clock, so the *unchanged* real-robot runtime (safety layer, startup ramp,
policy inference, timing loop) can be exercised against the simulator before
any hardware is powered. The emulated motors run the MIT impedance law at the
physics rate through the same DC-motor model used for training, with a
configurable command latency.
"""

from __future__ import annotations

import collections

import mujoco
import numpy as np

from dog2real.deploy.hal.base import IMU, ImuState, JointState, MotorBus
from dog2real.observation import projected_gravity_from_quat
from dog2real.robot.model_builder import build_mjcf, default_joint_pos
from dog2real.sim.actuator import DCMotorModel


class SimRobot(MotorBus):
    def __init__(self, robot_cfg: dict, sim_dt: float = 0.002, latency_s: float = 0.004,
                 start_pose: str = "lying", seed: int = 0, gyro_noise: float = 0.0):
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(robot_cfg, sim_dt))
        self.data = mujoco.MjData(self.model)
        self.sim_dt = sim_dt
        self.motor = DCMotorModel.from_config(robot_cfg["motor"])
        self.rng = np.random.default_rng(seed)
        self.gyro_noise = gyro_noise
        self.gyro_adr = self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")]
        self.enabled = False
        n = 12
        self._cmd = (np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n))
        delay = max(0, int(round(latency_s / sim_dt)))
        self._queue: collections.deque = collections.deque([self._cmd] * delay, maxlen=delay + 1)
        self.time = 0.0
        self._reset_pose(robot_cfg, start_pose)
        self._renderer = None

    def _reset_pose(self, robot_cfg: dict, start_pose: str) -> None:
        d, m = self.data, self.model
        mujoco.mj_resetData(m, d)
        if start_pose == "standing":
            d.qpos[7:19] = default_joint_pos(robot_cfg)
        else:  # lying on its belly with folded legs, like a robot that was just switched on
            d.qpos[7:19] = np.tile([0.0, 1.2, -2.6], 4)
        mujoco.mj_kinematics(m, d)
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
        lowest = np.min(d.geom_xpos[geoms, 2] - m.geom_rbound[geoms])
        d.qpos[2] += -lowest + 0.005
        for _ in range(int(0.5 / self.sim_dt)):  # settle
            mujoco.mj_step(m, d)

    # ------------------------------------------------------------- MotorBus --
    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def command(self, q_des, dq_des, kp, kd, tau_ff) -> None:
        self._cmd = tuple(np.array(x, dtype=np.float64) for x in (q_des, dq_des, kp, kd, tau_ff))

    def read(self) -> JointState:
        return JointState(self.data.qpos[7:19].copy(), self.data.qvel[6:18].copy(),
                          self.data.ctrl.copy(), self.time, True)

    # ------------------------------------------------------------------ IMU --
    def read_imu(self) -> ImuState:
        quat = self.data.qpos[3:7].copy()
        gyro = self.data.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()
        if self.gyro_noise:
            gyro += self.rng.normal(0.0, self.gyro_noise, 3)
        return ImuState(projected_gravity_from_quat(quat), gyro, self.time, quat)

    # ---------------------------------------------------------------- clock --
    def advance(self, duration: float) -> None:
        """Step physics for ``duration`` seconds, emulating the motor drivers."""
        for _ in range(max(1, int(round(duration / self.sim_dt)))):
            self._queue.append(self._cmd)
            q_des, dq_des, kp, kd, tau_ff = self._queue[0]
            q, dq = self.data.qpos[7:19], self.data.qvel[6:18]
            if self.enabled:
                tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
                tau = self.motor(tau, dq)
            else:
                tau = np.zeros(12)
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)
            self.time += self.sim_dt

    def base_height(self) -> float:
        return float(self.data.qpos[2])

    def render(self, width: int = 640, height: int = 360) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera="track")
        return self._renderer.render()


class SimIMUView(IMU):
    """Exposes the IMU half of a :class:`SimRobot` through the plain ``IMU`` interface."""

    def __init__(self, robot: SimRobot):
        self.robot = robot

    def read(self) -> ImuState:
        return self.robot.read_imu()
