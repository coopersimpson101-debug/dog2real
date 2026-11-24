import threading
import time

import numpy as np
import pytest
import torch

from dog2real.config import load_yaml
from dog2real.deploy.export import DeployPolicy, PolicyRunner, export_policy
from dog2real.deploy.hal.imu import GravityEstimator
from dog2real.deploy.safety import SafetyConfig, SafetyLayer
from dog2real.observation import build_actor_obs
from dog2real.rl.networks import ActorCritic
from dog2real.rl.normalizer import EmpiricalNormalization
from dog2real.robot.model_builder import default_joint_pos


@pytest.fixture(scope="module")
def robot():
    return load_yaml("configs/robot.yaml")


# ---------------------------------------------------------------- export ----
def _random_policy():
    torch.manual_seed(0)
    ac = ActorCritic(45, 60, 12, [64, 32], [64, 32])
    norm = EmpiricalNormalization(45)
    norm.update(torch.randn(500, 45) * 3 + 1)
    return DeployPolicy(norm, ac.actor, clip_actions=10.0).eval()


def test_policy_export_roundtrip_torchscript_and_onnx(tmp_path):
    policy = _random_policy()
    paths = export_policy(policy, tmp_path, cfg={"env": {"sim_dt": 0.002, "decimation": 10}})
    obs = np.random.default_rng(1).normal(size=(20, 45)).astype(np.float32) * 3
    with torch.no_grad():
        ref = policy(torch.from_numpy(obs)).numpy()
    ts = PolicyRunner(paths["torchscript"], backend="torchscript")
    np.testing.assert_allclose(np.stack([ts(o) for o in obs]), ref, atol=1e-5)
    pytest.importorskip("onnxruntime")
    ox = PolicyRunner(paths["onnx"], backend="onnx")
    assert ox.backend == "onnx"
    np.testing.assert_allclose(np.stack([ox(o) for o in obs]), ref, atol=1e-5)
    meta = (tmp_path / "policy.json").read_text()
    assert '"policy_dt": 0.02' in meta


# ---------------------------------------------------------------- safety ----
def _safety(robot, **kw):
    return SafetyLayer(SafetyConfig.from_dicts(robot, {"max_torque": 10.0, "max_target_step": 0.2, **kw}))


def test_safety_clamps_targets_to_joint_limits(robot):
    s = _safety(robot, max_target_step=100.0, max_torque=1e6)
    q = default_joint_pos(robot)
    cmd = s.filter(q + 50.0, q, np.zeros(12), 20.0, 0.5)
    assert np.all(cmd.q_des <= s.cfg.upper - s.cfg.limit_margin + 1e-12)
    cmd = s.filter(q - 50.0, q, np.zeros(12), 20.0, 0.5)
    assert np.all(cmd.q_des >= s.cfg.lower + s.cfg.limit_margin - 1e-12)
    assert s.clamp_events == 2


def test_safety_rate_limits_and_bounds_pd_torque(robot):
    s = _safety(robot, max_target_step=0.2, max_torque=1e6)
    q = default_joint_pos(robot)
    cmd = s.filter(q + 0.5, q, np.zeros(12), 20.0, 0.5)
    np.testing.assert_allclose(cmd.q_des - q, np.minimum(0.2, s.hi - q), atol=1e-12)

    s = _safety(robot, max_target_step=10.0, max_torque=3.0)
    dq = np.random.default_rng(0).normal(size=12) * 2
    cmd = s.filter(q + 0.4, q, dq, 20.0, 0.5)
    tau = cmd.kp * (cmd.q_des - q) - cmd.kd * dq
    assert np.all(np.abs(tau) <= 3.0 + 1e-9)


def test_safety_tilt_cutoff_and_damping_mode(robot):
    s = _safety(robot, tilt_cutoff_rad=0.8)
    q = default_joint_pos(robot)
    assert s.check_state(q, np.zeros(12), [0, 0, -1])
    tilted = [np.sin(1.0), 0.0, -np.cos(1.0)]  # 57 deg
    assert not s.check_state(q, np.zeros(12), tilted)
    assert "tilt" in s.fault
    cmd = s.filter(q + 0.3, q, np.zeros(12), 20.0, 0.5)
    assert np.all(cmd.kp == 0.0) and np.all(cmd.kd == s.cfg.damping_kd)
    np.testing.assert_allclose(cmd.q_des, q)
    # latched until explicit reset
    assert not s.check_state(q, np.zeros(12), [0, 0, -1])
    s.reset()
    assert s.check_state(q, np.zeros(12), [0, 0, -1])


def test_safety_estop_overspeed_comm_and_nan(robot):
    q = default_joint_pos(robot)
    s = _safety(robot)
    threading.Thread(target=s.estop, args=("button",)).start()
    time.sleep(0.05)
    assert not s.ok and s.fault == "button"
    s = _safety(robot)
    assert not s.check_state(q, np.full(12, 100.0), [0, 0, -1]) and "over-speed" in s.fault
    s = _safety(robot)
    assert not s.check_state(q, np.zeros(12), [0, 0, -1], comm_ok=False) and "communication" in s.fault
    s = _safety(robot)
    cmd = s.filter(np.full(12, np.nan), q, np.zeros(12), 20.0, 0.5)
    assert not s.ok and np.all(cmd.kp == 0) and np.all(np.isfinite(cmd.q_des))
    s = _safety(robot, tilt_check=False)
    assert s.check_state(q, np.zeros(12), [0, 0, 1])  # upside down allowed for the get-up policy


# ------------------------------------------------------------------- IMU ----
def test_gravity_estimator_tracks_rotation():
    est = GravityEstimator(gain=0.0)  # pure gyro integration
    dt, w = 0.002, np.array([0.5, 0.0, 0.0])  # roll at 0.5 rad/s
    for _ in range(1000):  # 2 s -> 1 rad
        g = est.update(w, np.zeros(3), dt)
    np.testing.assert_allclose(g, [0.0, -np.sin(1.0), -np.cos(1.0)], atol=1e-3)
    est = GravityEstimator(gain=0.05)
    for _ in range(500):  # accelerometer corrects a wrong initial estimate
        g = est.update(np.zeros(3), np.array([0.0, 0.0, 9.81]), dt)
    np.testing.assert_allclose(g, [0, 0, -1], atol=1e-3)


# --------------------------------------------------------- sim in the loop --
def test_runtime_sim_in_the_loop_stands_up_and_holds(robot):
    from dog2real.deploy.hal.sim_backend import SimIMUView, SimRobot
    from dog2real.deploy.runtime import RobotRuntime, SimClock

    dcfg = load_yaml("configs/deploy.yaml")
    sim = SimRobot(robot, latency_s=0.004, start_pose="lying")
    h0 = sim.base_height()
    safety = SafetyLayer(SafetyConfig.from_dicts(robot, dcfg["safety"]))
    calls = []

    def policy(obs):
        calls.append(obs)
        return np.zeros(12)

    rt = RobotRuntime(robot, dcfg, policy, sim, SimIMUView(sim), SimClock(0.02, sim), safety)
    rt.start()
    for _ in range(250):  # 3 s ramp + 2 s policy
        rt.tick()
        rt.clock.wait()
    assert safety.ok, safety.fault
    assert sim.base_height() > h0 + 0.1  # stood up
    assert len(calls) > 0 and calls[-1].shape == (45,)
    stats = rt.timing.summary()
    assert stats["ticks"] == 250 and stats["compute_mean_ms"] < 20.0
    rt.shutdown(damping_s=0.2)
    assert not sim.enabled


def test_runtime_observation_matches_env_layout(robot):
    """The runtime and the env must build bit-identical observations from the same state."""
    rng = np.random.default_rng(0)
    g, w, c = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
    q, dq, a = rng.normal(size=12), rng.normal(size=12), rng.normal(size=12)
    single = build_actor_obs(g, w, c, q, dq, a)
    batched = build_actor_obs(g[None], w[None], c[None], q[None], dq[None], a[None])
    np.testing.assert_array_equal(single, batched[0])
    np.testing.assert_allclose(single[3:6], w * 0.25, rtol=1e-6)
    np.testing.assert_allclose(single[21:33], dq * 0.05, rtol=1e-6)


# ----------------------------------------------------------------- sysid ----
def test_sysid_recovers_simulated_motor_parameters():
    from dog2real.sysid import fit_joint_dynamics, fit_torque_speed, simulate_bench_motor

    log = simulate_bench_motor(armature=0.02, damping=0.08, frictionloss=0.15, kp=5.0, kd=0.1, torque_noise=0.01)
    fit = fit_joint_dynamics(log["t"], log["dq"], log["tau"])
    assert fit["armature"] == pytest.approx(0.02, rel=0.15)
    assert fit["damping"] == pytest.approx(0.08, rel=0.25)
    assert fit["frictionloss"] == pytest.approx(0.15, rel=0.25)
    w = np.linspace(0, 25, 20)
    ts = fit_torque_speed(w, 22.0 * (1 - w / 30.0))
    assert ts["stall_torque"] == pytest.approx(22.0) and ts["no_load_speed"] == pytest.approx(30.0)
