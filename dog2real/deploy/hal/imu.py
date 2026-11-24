"""IMU readers and a gravity-direction estimator.

The policy needs the gravity direction in the body frame and the body angular
velocity. A raw 6-axis IMU (e.g. MPU-6050 / ICM-20948 over I2C) provides
gyro + accelerometer; :class:`GravityEstimator` fuses them with a Mahony-style
complementary filter. IMUs with on-chip fusion (BNO085, many serial AHRS
modules) can feed their quaternion straight in via ``ImuState.quat``.

Mounting: the IMU axes must be aligned with the base frame (x forward, y left,
z up), or set ``imu.rotation`` in ``configs/deploy.yaml`` (3x3, sensor -> body).
"""

from __future__ import annotations

import time

import numpy as np

from dog2real.deploy.hal.base import IMU, ImuState
from dog2real.observation import projected_gravity_from_quat


class GravityEstimator:
    """Complementary filter on the gravity unit vector in the body frame.

    Prediction: a gravity vector fixed in the world appears to rotate in the
    body frame as ``g_dot = -omega x g``. Correction: when the accelerometer
    magnitude is close to 1 g, pull ``g`` towards ``-acc / |acc|`` (the
    accelerometer measures specific force, which points *up* at rest).
    """

    def __init__(self, gain: float = 0.02, accel_gate: float = 0.15):
        self.gain = gain
        self.accel_gate = accel_gate
        self.g = np.array([0.0, 0.0, -1.0])

    def reset(self, acc: np.ndarray | None = None) -> None:
        if acc is not None and np.linalg.norm(acc) > 1e-3:
            self.g = -np.asarray(acc, dtype=np.float64) / np.linalg.norm(acc)
        else:
            self.g = np.array([0.0, 0.0, -1.0])

    def update(self, gyro: np.ndarray, acc: np.ndarray, dt: float) -> np.ndarray:
        gyro = np.asarray(gyro, dtype=np.float64)
        acc = np.asarray(acc, dtype=np.float64)
        g = self.g - np.cross(gyro, self.g) * dt
        a_norm = np.linalg.norm(acc)
        if a_norm > 1e-3 and abs(a_norm / 9.81 - 1.0) < self.accel_gate:
            g = (1.0 - self.gain) * g + self.gain * (-acc / a_norm)
        self.g = g / np.linalg.norm(g)
        return self.g.copy()


class Mpu6050IMU(IMU):
    """MPU-6050 / MPU-9250 (gyro+accel part) over I2C using ``smbus2`` (Raspberry Pi, Jetson).

    Call ``read()`` at a steady rate (the runtime does, at 50 Hz); for better
    estimates run :meth:`poll` in a faster background thread.
    """

    PWR_MGMT_1, GYRO_CONFIG, ACCEL_CONFIG, ACCEL_XOUT_H = 0x6B, 0x1B, 0x1C, 0x3B
    GYRO_SCALE = np.radians(1.0 / 65.5)   # +-500 deg/s
    ACC_SCALE = 9.81 / 8192.0             # +-4 g

    def __init__(self, bus: int = 1, address: int = 0x68, rotation=None, gyro_bias=None, filter_gain: float = 0.02):
        from smbus2 import SMBus  # imported lazily: only needed on the robot

        self.i2c = SMBus(bus)
        self.addr = address
        self.R = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
        self.gyro_bias = np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, dtype=np.float64)
        self.i2c.write_byte_data(self.addr, self.PWR_MGMT_1, 0x00)
        self.i2c.write_byte_data(self.addr, self.GYRO_CONFIG, 0x08)
        self.i2c.write_byte_data(self.addr, self.ACCEL_CONFIG, 0x08)
        self.est = GravityEstimator(filter_gain)
        self._last = None
        _, acc = self._raw()
        self.est.reset(acc)

    def _raw(self):
        b = self.i2c.read_i2c_block_data(self.addr, self.ACCEL_XOUT_H, 14)
        v = np.array([int.from_bytes(bytes(b[i:i + 2]), "big", signed=True) for i in range(0, 14, 2)], dtype=np.float64)
        acc = self.R @ (v[0:3] * self.ACC_SCALE)
        gyro = self.R @ (v[4:7] * self.GYRO_SCALE) - self.gyro_bias
        return gyro, acc

    def calibrate_gyro(self, seconds: float = 2.0) -> np.ndarray:
        """Robot must be still. Averages the gyro to estimate its bias."""
        self.gyro_bias = np.zeros(3)
        samples = []
        t_end = time.monotonic() + seconds
        while time.monotonic() < t_end:
            samples.append(self._raw()[0])
            time.sleep(0.005)
        self.gyro_bias = np.mean(samples, axis=0)
        return self.gyro_bias

    def poll(self) -> ImuState:
        now = time.monotonic()
        gyro, acc = self._raw()
        dt = 0.0 if self._last is None else now - self._last
        self._last = now
        g = self.est.update(gyro, acc, dt)
        self._state = ImuState(g, gyro, now)
        return self._state

    def read(self) -> ImuState:
        return self.poll()

    def close(self) -> None:
        self.i2c.close()


class QuaternionIMU(IMU):
    """Adapter for any AHRS that outputs a quaternion + gyro (BNO085, serial AHRS modules).

    ``source`` is a callable returning ``(quat_wxyz, gyro_xyz)`` in the body frame.
    """

    def __init__(self, source):
        self.source = source

    def read(self) -> ImuState:
        quat, gyro = self.source()
        quat = np.asarray(quat, dtype=np.float64)
        return ImuState(projected_gravity_from_quat(quat), np.asarray(gyro, dtype=np.float64), time.monotonic(), quat)
