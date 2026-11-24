import copy

import numpy as np
import pytest

from dog2real.config import load_train_config
from dog2real.sim.actuator import ActuatorNet, DCMotorModel, pd_torque
from dog2real.sim.env import VecQuadrupedEnv
from dog2real.sim.randomization import DomainRandomizer, IMUNoiseModel, ObservationNoise
from dog2real.sim.rewards import REWARD_FUNCTIONS, RewardManager


@pytest.fixture(scope="module")
def cfg():
    return load_train_config("configs/train_walk.yaml")


def test_randomization_samples_within_ranges(cfg):
    dr = cfg["domain_rand"]
    s = DomainRandomizer(dr).sample(2000, np.random.default_rng(0))
    for key in ("friction", "added_base_mass", "link_mass_scale", "motor_strength", "kp_scale", "kd_scale"):
        v = getattr(s, key)
        lo, hi = dr[key]
        assert v.min() >= lo and v.max() <= hi, key
        assert v.max() - v.min() > 0.8 * (hi - lo), f"{key} does not cover its range"
    assert np.all(np.abs(s.base_com_offset) <= np.asarray(dr["base_com_offset"]))
    lat = s.latency_substeps
    assert lat.dtype.kind == "i" and lat.min() == dr["latency_substeps"][0] and lat.max() == dr["latency_substeps"][1]


def test_randomization_disabled_is_nominal():
    s = DomainRandomizer({"enabled": False}).sample(5, np.random.default_rng(0), nominal_friction=0.8)
    assert np.all(s.friction == 0.8) and np.all(s.motor_strength == 1.0) and np.all(s.latency_substeps == 0)


def test_randomization_is_applied_to_each_env_model(cfg):
    env = VecQuadrupedEnv(cfg, num_envs=6, seed=0)
    env.reset()
    for i in range(6):
        m = env.models[i]
        assert np.allclose(m.geom_friction[:, 0], env.rand.friction[i])
        assert m.body_mass[env.base_body] == pytest.approx(env.applier.mass0[env.base_body] + env.rand.added_base_mass[i])
        np.testing.assert_allclose(m.body_ipos[env.base_body], env.applier.ipos0[env.base_body] + env.rand.base_com_offset[i])
    assert len(set(np.round(env.rand.friction, 6))) == 6  # every env differs


def test_latency_delays_actuation(cfg):
    c = copy.deepcopy(cfg)
    c["domain_rand"] = {"enabled": True, "latency_substeps": [10, 10]}  # one full policy step of delay
    c["imu_noise"]["enabled"] = False
    c["obs_noise"]["enabled"] = False
    env = VecQuadrupedEnv(c, num_envs=1, seed=0)
    env.reset()
    q_before = env.datas[0].qpos[7:19].copy()
    env.step(np.full((1, 12), 2.0))  # big action, but delayed by a whole step
    # first step still tracked the previous (reset) target -> small motion only
    moved_first = np.abs(env.datas[0].qpos[7:19] - q_before).max()
    env.step(np.full((1, 12), 2.0))
    moved_second = np.abs(env.datas[0].qpos[7:19] - q_before).max()
    assert moved_second > 3 * moved_first


def test_dc_motor_torque_speed_curve():
    m = DCMotorModel(torque_limit=18.0, stall_torque=22.0, no_load_speed=30.0)
    assert m(100.0, 0.0) == pytest.approx(18.0)           # clamp at peak torque
    assert m(100.0, 15.0) == pytest.approx(11.0)          # 22 * (1 - 15/30), motoring
    assert m(100.0, 40.0) == pytest.approx(0.0)           # beyond no-load speed
    assert m(-100.0, 15.0) == pytest.approx(-18.0)        # braking: full torque available
    assert m(5.0, 10.0) == pytest.approx(5.0)             # within envelope: untouched
    assert m(100.0, 0.0, strength=0.5) == pytest.approx(9.0)
    np.testing.assert_allclose(pd_torque(1.0, 0.5, 2.0, kp=20.0, kd=0.5), 9.0)


def test_actuator_net_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    w = [(rng.normal(size=(8, 6)), rng.normal(size=8)), (rng.normal(size=(1, 8)), rng.normal(size=1))]
    net = ActuatorNet(w, history=3, torque_limit=5.0)
    net.save(tmp_path / "a.npz")
    net2 = ActuatorNet.load(tmp_path / "a.npz", torque_limit=5.0)
    e, v = rng.normal(size=(4, 12, 3)), rng.normal(size=(4, 12, 3))
    out = net2(e, v)
    assert out.shape == (4, 12) and np.all(np.abs(out) <= 5.0)
    np.testing.assert_allclose(out, net(e, v))


def test_actuator_net_training_fits_pd_motor():
    from dog2real.actuator_net_training import train_actuator_net
    from dog2real.sysid import simulate_bench_motor

    log = simulate_bench_motor(0.01, 0.05, 0.1, kp=20.0, kd=0.5, dt=0.002, duration=4.0)
    net, rmse = train_actuator_net(log["q_des"], log["q"], log["dq"], log["tau"], history=2, epochs=150)
    assert rmse < 0.25 * np.std(log["tau"])


def test_imu_and_observation_noise_statistics():
    rng = np.random.default_rng(0)
    imu = IMUNoiseModel({"gyro_bias": 0.1, "gyro_noise": 0.05, "tilt_bias": 0.05, "gravity_noise": 0.01})
    gb, tb = imu.sample_bias(5000, rng)
    assert np.all(np.abs(gb) <= 0.1) and np.all(np.abs(tb) <= 0.05)
    gyro = np.zeros((5000, 3))
    grav = np.tile([0.0, 0.0, -1.0], (5000, 1))
    g_meas, grav_meas = imu.apply(gyro, grav, np.zeros_like(gb), np.zeros_like(tb), rng)
    assert np.std(g_meas) == pytest.approx(0.05, rel=0.1)
    np.testing.assert_allclose(np.linalg.norm(grav_meas, axis=1), 1.0, atol=1e-9)
    on = ObservationNoise({"dof_pos": 0.01, "dof_vel": 1.0})
    p, v = on.apply(np.zeros((1000, 12)), np.zeros((1000, 12)), rng)
    assert np.abs(p).max() <= 0.01 and np.abs(v).max() <= 1.0


def test_push_perturbation_changes_base_velocity(cfg):
    c = copy.deepcopy(cfg)
    c["domain_rand"]["push"] = {"enabled": True, "interval_s": 0.02, "max_vel_xy": 1.0}
    env = VecQuadrupedEnv(c, num_envs=4, seed=0)
    env.reset()
    env.step(np.zeros((4, 12)))
    vel = np.array([d.qvel[0:2] for d in env.datas])
    assert np.abs(vel).max() > 0.05


def test_reward_terms_are_finite_on_random_state():
    rng = np.random.default_rng(0)
    n = 16
    s = {
        "commands": rng.normal(size=(n, 3)), "base_lin_vel": rng.normal(size=(n, 3)),
        "base_ang_vel": rng.normal(size=(n, 3)), "projected_gravity": rng.normal(size=(n, 3)),
        "base_height": rng.uniform(0, 0.4, n), "base_height_target": 0.3, "tracking_sigma": 0.25,
        "torques": rng.normal(size=(n, 12)) * 10, "dof_vel": rng.normal(size=(n, 12)),
        "last_dof_vel": rng.normal(size=(n, 12)), "dt": 0.02, "actions": rng.normal(size=(n, 12)),
        "last_actions": rng.normal(size=(n, 12)), "dof_pos": rng.normal(size=(n, 12)),
        "default_dof_pos": np.zeros(12), "soft_lower": -np.ones(12), "soft_upper": np.ones(12),
        "feet_air_time": rng.uniform(0, 1, (n, 4)), "first_contact": rng.random((n, 4)) > 0.5,
        "collisions": rng.integers(0, 3, n), "terminated": rng.random(n) > 0.5, "time_out": rng.random(n) > 0.5,
    }
    for name, f in REWARD_FUNCTIONS.items():
        r = f(s)
        assert r.shape == (n,) and np.all(np.isfinite(r)), name
    rm = RewardManager({"tracking_lin_vel": 1.0, "torques": -1e-4}, dt=0.02)
    total, terms = rm.compute(s)
    np.testing.assert_allclose(total, terms["tracking_lin_vel"] + terms["torques"])
    with pytest.raises(KeyError):
        RewardManager({"does_not_exist": 1.0}, 0.02)


def test_tracking_reward_is_max_when_tracking_perfectly():
    from dog2real.sim.rewards import tracking_lin_vel

    s = {"commands": np.array([[0.5, 0.1, 0.0]]), "base_lin_vel": np.array([[0.5, 0.1, 0.3]]), "tracking_sigma": 0.25}
    assert tracking_lin_vel(s)[0] == pytest.approx(1.0)
