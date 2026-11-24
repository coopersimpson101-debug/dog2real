"""Hardware abstraction layer interfaces.

The runtime only talks to these two interfaces, so swapping a CAN BLDC bus for
serial bus servos, or real hardware for the MuJoCo sim-in-the-loop backend,
requires no change to the control loop.

All quantities are in the *joint* frame used in simulation: radians, rad/s,
Nm, ordered FL, FR, RL, RR x (abad, hip, knee). Drivers are responsible for
the motor-to-joint mapping (direction sign, zero offset, external gear ratio).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class JointState:
    q: np.ndarray          # (12,) rad
    dq: np.ndarray         # (12,) rad/s
    tau: np.ndarray        # (12,) Nm (estimated from motor current)
    timestamp: float       # seconds (monotonic)
    comm_ok: bool = True   # False if any motor has not replied recently


@dataclass
class ImuState:
    projected_gravity: np.ndarray  # (3,) unit vector, gravity direction in the body frame
    gyro: np.ndarray               # (3,) rad/s, body frame
    timestamp: float
    quat: np.ndarray | None = None  # (4,) w, x, y, z if the IMU provides it


class MotorBus(ABC):
    """A set of 12 position-controlled joints with an impedance (MIT-style) interface."""

    @abstractmethod
    def enable(self) -> None: ...

    @abstractmethod
    def disable(self) -> None: ...

    @abstractmethod
    def command(self, q_des: np.ndarray, dq_des: np.ndarray, kp: np.ndarray, kd: np.ndarray,
                tau_ff: np.ndarray) -> None:
        """Send one impedance command per joint: tau = kp (q_des - q) + kd (dq_des - dq) + tau_ff."""

    @abstractmethod
    def read(self) -> JointState:
        """Latest joint state (non-blocking; returns cached replies)."""

    def close(self) -> None:
        self.disable()


class IMU(ABC):
    @abstractmethod
    def read(self) -> ImuState: ...

    def close(self) -> None:
        pass
