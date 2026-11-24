"""MIT-mode CAN driver for BLDC actuators (Mini Cheetah protocol) over python-can.

The "MIT mode" protocol (Ben Katz' Mini Cheetah motor controller, also used by
CubeMars AK-series in MIT mode, many MyActuator / Steadywin / Tmotor clones and
ODrive firmware forks) packs a full impedance command into one 8-byte frame:

  command (host -> motor, arbitration id = motor CAN id)
    p_des 16 bit | v_des 12 bit | kp 12 bit | kd 12 bit | t_ff 12 bit
  reply (motor -> host)
    byte0 = motor id | p 16 bit | v 12 bit | t 12 bit

Each float is linearly mapped from ``[min, max]`` onto an unsigned integer.
The ranges (``P_MAX``, ``V_MAX``, ``T_MAX``, ...) are firmware constants and
DIFFER BETWEEN MOTOR MODELS -- check your motor's manual and set them in
``configs/deploy.yaml``. Special frames: ``FF..FC`` enter motor mode,
``FF..FD`` exit motor mode, ``FF..FE`` set the current position as zero.

Tested here only against python-can's ``virtual`` interface with a simulated
motor (``tests/test_mit_can.py``). Validate on your bench with one motor, low
gains and the robot on a stand before running the full loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dog2real.deploy.hal.base import JointState, MotorBus

ENTER_MOTOR_MODE = bytes([0xFF] * 7 + [0xFC])
EXIT_MOTOR_MODE = bytes([0xFF] * 7 + [0xFD])
SET_ZERO = bytes([0xFF] * 7 + [0xFE])

# Firmware limits of a few common motors (output side). VERIFY for your firmware.
MOTOR_PRESETS = {
    "AK80-9": dict(p_max=12.5, v_max=50.0, t_max=18.0, kp_max=500.0, kd_max=5.0),
    "AK70-10": dict(p_max=12.5, v_max=50.0, t_max=25.0, kp_max=500.0, kd_max=5.0),
    "AK60-6": dict(p_max=12.5, v_max=45.0, t_max=15.0, kp_max=500.0, kd_max=5.0),
    "AK10-9": dict(p_max=12.5, v_max=50.0, t_max=65.0, kp_max=500.0, kd_max=5.0),
    "minicheetah": dict(p_max=12.5, v_max=65.0, t_max=18.0, kp_max=500.0, kd_max=5.0),
}


def float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    x = min(max(float(x), x_min), x_max)
    return int((x - x_min) * ((1 << bits) - 1) / (x_max - x_min))


def uint_to_float(x: int, x_min: float, x_max: float, bits: int) -> float:
    return float(x) * (x_max - x_min) / ((1 << bits) - 1) + x_min


@dataclass
class MotorLimits:
    p_max: float = 12.5
    v_max: float = 50.0
    t_max: float = 18.0
    kp_max: float = 500.0
    kd_max: float = 5.0

    @classmethod
    def preset(cls, name: str, **overrides) -> "MotorLimits":
        return cls(**{**MOTOR_PRESETS[name], **overrides})


def pack_command(p: float, v: float, kp: float, kd: float, t: float, lim: MotorLimits) -> bytes:
    pi = float_to_uint(p, -lim.p_max, lim.p_max, 16)
    vi = float_to_uint(v, -lim.v_max, lim.v_max, 12)
    kpi = float_to_uint(kp, 0.0, lim.kp_max, 12)
    kdi = float_to_uint(kd, 0.0, lim.kd_max, 12)
    ti = float_to_uint(t, -lim.t_max, lim.t_max, 12)
    return bytes([
        pi >> 8, pi & 0xFF,
        vi >> 4, ((vi & 0xF) << 4) | (kpi >> 8),
        kpi & 0xFF,
        kdi >> 4, ((kdi & 0xF) << 4) | (ti >> 8),
        ti & 0xFF,
    ])


def unpack_command(data: bytes, lim: MotorLimits) -> tuple[float, float, float, float, float]:
    """Inverse of :func:`pack_command` (used by the motor simulator in tests)."""
    pi = (data[0] << 8) | data[1]
    vi = (data[2] << 4) | (data[3] >> 4)
    kpi = ((data[3] & 0xF) << 8) | data[4]
    kdi = (data[5] << 4) | (data[6] >> 4)
    ti = ((data[6] & 0xF) << 8) | data[7]
    return (uint_to_float(pi, -lim.p_max, lim.p_max, 16), uint_to_float(vi, -lim.v_max, lim.v_max, 12),
            uint_to_float(kpi, 0.0, lim.kp_max, 12), uint_to_float(kdi, 0.0, lim.kd_max, 12),
            uint_to_float(ti, -lim.t_max, lim.t_max, 12))


def pack_reply(motor_id: int, p: float, v: float, t: float, lim: MotorLimits) -> bytes:
    pi = float_to_uint(p, -lim.p_max, lim.p_max, 16)
    vi = float_to_uint(v, -lim.v_max, lim.v_max, 12)
    ti = float_to_uint(t, -lim.t_max, lim.t_max, 12)
    return bytes([motor_id & 0xFF, pi >> 8, pi & 0xFF, vi >> 4, ((vi & 0xF) << 4) | (ti >> 8), ti & 0xFF])


def unpack_reply(data: bytes, lim: MotorLimits) -> tuple[int, float, float, float]:
    motor_id = data[0]
    pi = (data[1] << 8) | data[2]
    vi = (data[3] << 4) | (data[4] >> 4)
    ti = ((data[4] & 0xF) << 8) | data[5]
    return (motor_id, uint_to_float(pi, -lim.p_max, lim.p_max, 16),
            uint_to_float(vi, -lim.v_max, lim.v_max, 12), uint_to_float(ti, -lim.t_max, lim.t_max, 12))


@dataclass
class MotorSpec:
    """One joint. Joint-space value = sign * motor_value / gear + offset."""

    name: str
    can_id: int
    limits: MotorLimits = field(default_factory=MotorLimits)
    sign: float = 1.0
    offset: float = 0.0     # rad, joint angle when the motor reads zero
    gear: float = 1.0       # extra reduction between motor output and joint (belt / linkage)

    def joint_to_motor(self, q, dq, kp, kd, tau):
        s, g = self.sign, self.gear
        return (s * (q - self.offset) * g, s * dq * g, kp / g**2, kd / g**2, s * tau / g)

    def motor_to_joint(self, p, v, t):
        s, g = self.sign, self.gear
        return s * p / g + self.offset, s * v / g, s * t * g


class MitCanMotorBus(MotorBus):
    """12 MIT-mode motors on one or more CAN buses.

    ``command`` sends one frame per motor and then drains replies for up to
    ``reply_timeout`` seconds. ``read`` returns the cached latest state; a motor
    that has not replied for ``comm_timeout`` seconds marks ``comm_ok = False``,
    which the safety layer turns into damping mode.
    """

    def __init__(self, motors: list[MotorSpec], channel: str = "can0", interface: str = "socketcan",
                 bitrate: int = 1_000_000, reply_timeout: float = 0.002, comm_timeout: float = 0.05,
                 bus=None):
        import can  # python-can

        self.can = can
        self.motors = motors
        self.by_id = {m.can_id: k for k, m in enumerate(motors)}
        self.bus = bus if bus is not None else can.Bus(channel=channel, interface=interface, bitrate=bitrate)
        self.reply_timeout = reply_timeout
        self.comm_timeout = comm_timeout
        n = len(motors)
        self.q, self.dq, self.tau = np.zeros(n), np.zeros(n), np.zeros(n)
        self.last_reply = np.full(n, -np.inf)
        self.enabled = False

    def _send(self, can_id: int, data: bytes) -> None:
        self.bus.send(self.can.Message(arbitration_id=can_id, data=data, is_extended_id=False))

    def _drain(self, timeout: float, expected: int | None = None) -> int:
        got = 0
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            msg = self.bus.recv(timeout=max(remaining, 0.0))
            if msg is None:
                break
            if len(msg.data) >= 6:
                mid = msg.data[0]
                k = self.by_id.get(mid)
                if k is not None:
                    spec = self.motors[k]
                    _, p, v, t = unpack_reply(bytes(msg.data), spec.limits)
                    self.q[k], self.dq[k], self.tau[k] = spec.motor_to_joint(p, v, t)
                    self.last_reply[k] = time.monotonic()
                    got += 1
            if (expected is not None and got >= expected) or remaining <= 0:
                break
        return got

    def enable(self) -> None:
        for m in self.motors:
            self._send(m.can_id, ENTER_MOTOR_MODE)
        self._drain(0.05, expected=len(self.motors))
        self.enabled = True

    def disable(self) -> None:
        for m in self.motors:
            self._send(m.can_id, EXIT_MOTOR_MODE)
        self._drain(0.02)
        self.enabled = False

    def set_zero(self) -> None:
        """Store the current position as zero in every motor. Only with the robot in its calibration jig!"""
        for m in self.motors:
            self._send(m.can_id, SET_ZERO)
        self._drain(0.05)

    def command(self, q_des, dq_des, kp, kd, tau_ff) -> None:
        for k, m in enumerate(self.motors):
            p, v, kpm, kdm, t = m.joint_to_motor(q_des[k], dq_des[k], kp[k], kd[k], tau_ff[k])
            self._send(m.can_id, pack_command(p, v, kpm, kdm, t, m.limits))
        self._drain(self.reply_timeout, expected=len(self.motors))

    def read(self) -> JointState:
        now = time.monotonic()
        comm_ok = bool(np.all(now - self.last_reply < self.comm_timeout))
        return JointState(self.q.copy(), self.dq.copy(), self.tau.copy(), now, comm_ok)

    def close(self) -> None:
        try:
            self.disable()
        finally:
            self.bus.shutdown()

    @classmethod
    def from_config(cls, cfg: dict, bus=None) -> "MitCanMotorBus":
        """Build from the ``motors:`` section of ``configs/deploy.yaml``."""
        from dog2real import JOINT_NAMES

        default_model = cfg.get("model", "AK80-9")
        specs = []
        by_name = {j["name"]: j for j in cfg["joints"]}
        for name in JOINT_NAMES:
            j = by_name[name]
            lim = MotorLimits.preset(j.get("model", default_model), **j.get("limits", {}))
            specs.append(MotorSpec(name, int(j["can_id"]), lim, float(j.get("sign", 1.0)),
                                   float(j.get("offset", 0.0)), float(j.get("gear", 1.0))))
        return cls(specs, channel=cfg.get("channel", "can0"), interface=cfg.get("interface", "socketcan"),
                   bitrate=int(cfg.get("bitrate", 1_000_000)), reply_timeout=float(cfg.get("reply_timeout", 0.002)),
                   comm_timeout=float(cfg.get("comm_timeout", 0.05)), bus=bus)
