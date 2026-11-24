import copy

import mujoco
import numpy as np
import pytest

from dog2real import JOINT_NAMES
from dog2real.config import load_train_config, load_yaml
from dog2real.observation import ACTOR_OBS_DIM, projected_gravity_from_quat, quat_to_rotmat
from dog2real.robot.model_builder import build_mjcf, default_joint_pos, load_model
from dog2real.sim.env import VecQuadrupedEnv, random_quaternions


@pytest.fixture(scope="module")
def robot():
    return load_yaml("configs/robot.yaml")


@pytest.fixture(scope="module")
def walk_cfg():
    return load_train_config("configs/train_walk.yaml")


def test_model_loads_and_has_12_dof(robot):
    m = load_model(robot)
    assert (m.nq, m.nv, m.nu) == (19, 18, 12)
    for name in JOINT_NAMES:
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
    expected = robot["base"]["mass"] + 4 * sum(robot["leg"][k]["mass"] for k in ("abad", "thigh", "calf", "foot"))
    assert mujoco.mj_getTotalmass(m) == pytest.approx(expected, rel=1e-6)


def test_yaml_parameters_change_the_model(robot):
    r2 = copy.deepcopy(robot)
    r2["leg"]["thigh"]["length"] = 0.25
    r2["base"]["mass"] += 1.0
    m1, m2 = load_model(robot), load_model(r2)
    assert mujoco.mj_getTotalmass(m2) == pytest.approx(mujoco.mj_getTotalmass(m1) + 1.0)
    calf = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, "FL_calf")
    assert m2.body_pos[calf][2] == pytest.approx(-0.25)


def test_robot_stands_under_pd_in_default_pose(robot):
    m = load_model(robot)
    d = mujoco.MjData(m)
    q0 = default_joint_pos(robot)
    d.qpos[7:19] = q0
    mujoco.mj_kinematics(m, d)
    d.qpos[2] = 0.33
    for _ in range(1000):  # 2 s of PD hold
        d.ctrl[:] = np.clip(robot["pd"]["kp"] * (q0 - d.qpos[7:19]) - robot["pd"]["kd"] * d.qvel[6:18], -18, 18)
        mujoco.mj_step(m, d)
    assert 0.2 < d.qpos[2] < 0.34
    assert projected_gravity_from_quat(d.qpos[3:7])[2] < -0.95


def test_quaternion_helpers():
    rng = np.random.default_rng(0)
    q = random_quaternions(100, rng)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-9)
    R = quat_to_rotmat(q)
    np.testing.assert_allclose(R @ np.transpose(R, (0, 2, 1)), np.broadcast_to(np.eye(3), R.shape), atol=1e-9)
    ref = np.empty(9)
    mujoco.mju_quat2Mat(ref, q[0])
    np.testing.assert_allclose(R[0].ravel(), ref, atol=1e-9)


@pytest.mark.parametrize("cfg_name", ["train_walk.yaml", "train_recovery.yaml"])
def test_env_reset_step_shapes_and_finite_rewards(cfg_name):
    cfg = load_train_config(cfg_name)
    env = VecQuadrupedEnv(cfg, num_envs=4, seed=0)
    obs, priv = env.reset()
    assert obs.shape == (4, ACTOR_OBS_DIM) and obs.dtype == np.float32
    assert priv.shape == (4, env.num_privileged_obs)
    rng = np.random.default_rng(0)
    for _ in range(60):
        obs, rew, done, info = env.step(rng.normal(size=(4, 12)))
        assert obs.shape == (4, 45) and rew.shape == (4,) and done.shape == (4,)
        assert np.all(np.isfinite(obs)) and np.all(np.isfinite(rew)) and np.all(np.isfinite(info["privileged_obs"]))
        assert info["time_outs"].dtype == bool


def test_upright_standing_has_expected_observation(walk_cfg):
    cfg = copy.deepcopy(walk_cfg)
    cfg["domain_rand"]["enabled"] = False
    cfg["imu_noise"]["enabled"] = False
    cfg["obs_noise"]["enabled"] = False
    env = VecQuadrupedEnv(cfg, num_envs=2, seed=0)
    env.reset()
    for _ in range(20):
        obs, rew, done, _ = env.step(np.zeros((2, 12)))
    np.testing.assert_allclose(obs[:, 0:3], [[0, 0, -1]] * 2, atol=0.1)  # projected gravity
    assert not done.any()
    assert np.all(env._state["feet_contact"])


def test_walk_terminates_when_flipped(walk_cfg):
    env = VecQuadrupedEnv(walk_cfg, num_envs=2, seed=0)
    env.reset()
    d = env.datas[0]
    d.qpos[3:7] = [0.0, 1.0, 0.0, 0.0]  # upside down
    d.qpos[2] = 0.4
    mujoco.mj_forward(env.models[0], d)
    _, _, done, info = env.step(np.zeros((2, 12)))
    assert done[0] and info["terminated"][0] and not info["time_outs"][0]
    assert env.episode_length[0] == 0  # auto-reset happened


def test_time_out_is_flagged(walk_cfg):
    cfg = copy.deepcopy(walk_cfg)
    cfg["env"]["episode_length_s"] = 0.1  # 5 policy steps
    cfg["domain_rand"]["push"]["enabled"] = False
    env = VecQuadrupedEnv(cfg, num_envs=2, seed=0)
    env.reset()
    for k in range(5):
        _, _, done, info = env.step(np.zeros((2, 12)))
    assert done.all() and info["time_outs"].all()
    assert "episode" in info


def test_recovery_reset_starts_fallen_and_has_zero_command():
    cfg = load_train_config("train_recovery.yaml")
    env = VecQuadrupedEnv(cfg, num_envs=8, seed=3)
    obs, _ = env.reset()
    assert np.all(env.commands == 0.0)
    gz = obs[:, 2]
    assert np.mean(gz > -0.9) >= 0.5  # most envs start not upright
    for _ in range(5):
        _, _, done, _ = env.step(np.zeros((8, 12)))
        assert not done.any()  # get-up task never terminates early


def test_command_curriculum_grows_ranges():
    from dog2real.sim.commands import CommandCurriculum

    c = CommandCurriculum({"init_ranges": {"lin_vel_x": [-0.2, 0.2], "lin_vel_y": [0, 0], "ang_vel_yaw": [0, 0]},
                           "max_ranges": {"lin_vel_x": [-0.3, 1.0], "lin_vel_y": [0, 0], "ang_vel_yaw": [0, 0]},
                           "threshold": 0.8, "step": 0.5})
    assert not c.update(0.5)
    assert c.update(0.9)
    np.testing.assert_allclose(c.ranges["lin_vel_x"], [-0.3, 0.7])
    c.update(0.9)
    np.testing.assert_allclose(c.ranges["lin_vel_x"], [-0.3, 1.0])
    assert not c.update(0.9)  # saturated
    cmds = c.sample(1000, np.random.default_rng(0))
    assert cmds[:, 0].min() >= -0.3 and cmds[:, 0].max() <= 1.0


def test_gymnasium_wrapper(walk_cfg):
    from gymnasium.utils.env_checker import check_env

    from dog2real.sim.gym_env import QuadrupedGymEnv

    env = QuadrupedGymEnv(copy.deepcopy(walk_cfg))
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=1)
    obs, r, term, trunc, _ = env.step(env.action_space.sample())
    assert obs.shape == (45,) and isinstance(r, float)


def test_mjcf_is_deterministic(robot):
    assert build_mjcf(robot) == build_mjcf(robot)
