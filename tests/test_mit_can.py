"""MIT-mode CAN protocol tests, including a full loop over python-can's virtual bus."""

import threading
import time

import numpy as np
import pytest

from dog2real import JOINT_NAMES
from dog2real.deploy.hal.mit_can import (ENTER_MOTOR_MODE, EXIT_MOTOR_MODE, MitCanMotorBus, MotorLimits, MotorSpec,
                                         float_to_uint, pack_command, pack_reply, uint_to_float, unpack_command,
                                         unpack_reply)

can = pytest.importorskip("can")


def test_float_uint_roundtrip_and_saturation():
    for bits in (12, 16):
        for x in np.linspace(-12.5, 12.5, 37):
            y = uint_to_float(float_to_uint(x, -12.5, 12.5, bits), -12.5, 12.5, bits)
            assert abs(y - x) <= 25.0 / ((1 << bits) - 1) + 1e-12
    assert float_to_uint(99.0, -1, 1, 12) == 4095 and float_to_uint(-99.0, -1, 1, 12) == 0


def test_command_frame_layout_and_roundtrip():
    lim = MotorLimits.preset("AK80-9")
    frame = pack_command(1.0, -2.0, 30.0, 1.2, 3.0, lim)
    assert len(frame) == 8
    p, v, kp, kd, t = unpack_command(frame, lim)
    assert p == pytest.approx(1.0, abs=4e-4) and v == pytest.approx(-2.0, abs=0.03)
    assert kp == pytest.approx(30.0, abs=0.13) and kd == pytest.approx(1.2, abs=2e-3) and t == pytest.approx(3.0, abs=0.01)
    # zero command maps to mid-scale for symmetric ranges
    z = pack_command(0, 0, 0, 0, 0, lim)
    assert (z[0] << 8 | z[1]) == 32767


def test_reply_roundtrip():
    lim = MotorLimits.preset("AK70-10")
    mid, p, v, t = unpack_reply(pack_reply(7, -0.7, 4.0, -5.0, lim), lim)
    assert mid == 7 and p == pytest.approx(-0.7, abs=4e-4) and v == pytest.approx(4.0, abs=0.03)
    assert t == pytest.approx(-5.0, abs=0.02)


def test_joint_motor_mapping_inverse():
    spec = MotorSpec("x", 1, sign=-1.0, offset=0.3, gear=1.5)
    p, v, kp, kd, t = spec.joint_to_motor(0.8, 2.0, 20.0, 0.5, 3.0)
    q, dq, tau = spec.motor_to_joint(p, v, t)
    assert (q, dq, tau) == pytest.approx((0.8, 2.0, 3.0))
    # impedance must be equivalent at the joint: kp_m * g^2 == kp_joint
    assert kp * 1.5 ** 2 == pytest.approx(20.0)


class FakeMotors(threading.Thread):
    """Simulated MIT-mode motors on a virtual CAN channel: integrate a 1-DoF rotor under the impedance law."""

    def __init__(self, channel, ids, lim):
        super().__init__(daemon=True)
        self.bus = can.Bus(interface="virtual", channel=channel)
        self.lim = lim
        self.state = {i: [0.0, 0.0] for i in ids}  # q, dq
        self.enabled = {i: False for i in ids}
        self.running = True
        self.last_cmd = {}

    def run(self):
        while self.running:
            msg = self.bus.recv(timeout=0.01)
            if msg is None:
                continue
            mid, data = msg.arbitration_id, bytes(msg.data)
            if mid not in self.state:
                continue
            if data == ENTER_MOTOR_MODE:
                self.enabled[mid] = True
            elif data == EXIT_MOTOR_MODE:
                self.enabled[mid] = False
            elif self.enabled[mid]:
                p, v, kp, kd, t = unpack_command(data, self.lim)
                self.last_cmd[mid] = (p, v, kp, kd, t)
                q, dq = self.state[mid]
                for _ in range(10):  # 10 x 1 ms of rotor dynamics (J = 0.01)
                    tau = kp * (p - q) + kd * (v - dq) + t
                    dq += tau / 0.01 * 0.001 - 5.0 * dq * 0.001
                    q += dq * 0.001
                self.state[mid] = [q, dq]
            q, dq = self.state[mid]
            self.bus.send(can.Message(arbitration_id=0, data=pack_reply(mid, q, dq, 0.0, self.lim), is_extended_id=False))

    def stop(self):
        self.running = False
        self.join(timeout=1)
        self.bus.shutdown()


def test_full_loop_over_virtual_can_bus():
    lim = MotorLimits.preset("AK80-9")
    channel = f"dog2real_test_{time.monotonic_ns()}"
    ids = list(range(1, 13))
    motors = FakeMotors(channel, ids, lim)
    motors.start()
    host_bus = can.Bus(interface="virtual", channel=channel)
    specs = [MotorSpec(n, i, lim, sign=-1.0 if k % 2 else 1.0) for k, (n, i) in enumerate(zip(JOINT_NAMES, ids))]
    bus = MitCanMotorBus(specs, bus=host_bus, reply_timeout=0.05, comm_timeout=0.5)
    try:
        bus.enable()
        assert bus.read().comm_ok
        target = np.linspace(-0.5, 0.5, 12)
        for _ in range(60):
            bus.command(target, np.zeros(12), np.full(12, 5.0), np.full(12, 0.3), np.zeros(12))
        st = bus.read()
        assert st.comm_ok
        np.testing.assert_allclose(st.q, target, atol=0.02)  # motors converged to the joint-space targets
        # sign mapping: odd joints are mirrored in motor space
        p_motor_j1 = motors.last_cmd[2][0]
        assert p_motor_j1 == pytest.approx(-target[1], abs=1e-3)
    finally:
        bus.disable()
        motors.stop()
        host_bus.shutdown()
    assert not any(motors.enabled.values())


def test_comm_timeout_detected():
    lim = MotorLimits()
    host_bus = can.Bus(interface="virtual", channel=f"dog2real_silent_{time.monotonic_ns()}")
    bus = MitCanMotorBus([MotorSpec("FL_abad", 1, lim)], bus=host_bus, reply_timeout=0.001, comm_timeout=0.01)
    bus.command(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1))
    assert not bus.read().comm_ok  # nobody answered
    host_bus.shutdown()


def test_from_config_reads_deploy_yaml():
    from dog2real.config import load_yaml

    cfg = load_yaml("configs/deploy.yaml")["motors"]
    host_bus = can.Bus(interface="virtual", channel=f"dog2real_cfg_{time.monotonic_ns()}")
    bus = MitCanMotorBus.from_config(cfg, bus=host_bus)
    assert [m.name for m in bus.motors] == list(JOINT_NAMES)
    assert len({m.can_id for m in bus.motors}) == 12
    host_bus.shutdown()
