"""Vectorised MuJoCo quadruped environment (walking and get-up tasks).

``VecQuadrupedEnv`` steps ``num_envs`` independent ``(MjModel, MjData)`` pairs.
Everything that is not the physics step itself -- PD control, the actuator
model, observations, rewards, resets -- is batched in numpy across envs.

Timing: physics at ``sim_dt`` (default 2 ms = 500 Hz); the policy acts every
``decimation`` physics steps (default 10 -> 50 Hz). Actions are joint-position
offsets: ``q* = q_default + action_scale * a``. The PD law + DC-motor model run
at the physics rate, like the motor driver on the real robot.

API (rsl_rl-like, auto-reset)::

    obs, priv_obs = env.reset()
    obs, reward, done, info = env.step(actions)   # info["time_outs"], info["privileged_obs"]
"""

from __future__ import annotations

import mujoco
import numpy as np

from dog2real import JOINT_NAMES, LEG_NAMES, NUM_JOINTS
from dog2real.observation import ACTOR_OBS_DIM, build_actor_obs, quat_to_rotmat
from dog2real.robot.model_builder import build_mjcf, default_joint_pos, joint_limits
from dog2real.sim.actuator import ActuatorNet, DCMotorModel, pd_torque
from dog2real.sim.commands import CommandCurriculum
from dog2real.sim.randomization import (DomainRandomizer, IMUNoiseModel, ModelParamApplier,
                                        ObservationNoise)
from dog2real.sim.rewards import RewardManager

PRIVILEGED_EXTRA_DIM = 3 + 1 + 4 + 1 + 1 + 3 + 1 + 1


def random_quaternions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed unit quaternions (w, x, y, z) (Shoemake 1992)."""
    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    a, b = np.sqrt(1 - u1), np.sqrt(u1)
    return np.stack([a * np.sin(2 * np.pi * u2), a * np.cos(2 * np.pi * u2),
                     b * np.sin(2 * np.pi * u3), b * np.cos(2 * np.pi * u3)], axis=1)[:, [3, 0, 1, 2]]


def yaw_quaternions(yaw: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], axis=1)


class VecQuadrupedEnv:
    def __init__(self, cfg: dict, num_envs: int | None = None, seed: int = 0, auto_reset: bool = True):
        self.cfg = cfg
        self.robot_cfg = cfg["robot"]
        ecfg = cfg["env"]
        self.task = cfg.get("task", "walk")
        if self.task not in ("walk", "recovery"):
            raise ValueError(f"unknown task {self.task!r}")
        self.num_envs = int(num_envs or ecfg.get("num_envs", 16))
        self.auto_reset = auto_reset
        self.rng = np.random.default_rng(seed)

        self.sim_dt = float(ecfg.get("sim_dt", 0.002))
        self.decimation = int(ecfg.get("decimation", 10))
        self.dt = self.sim_dt * self.decimation
        self.max_episode_length = int(round(float(ecfg.get("episode_length_s", 20.0)) / self.dt))
        self.action_scale = float(ecfg.get("action_scale", 0.25))
        self.clip_actions = float(ecfg.get("clip_actions", 10.0))
        self.base_height_target = float(ecfg.get("base_height_target", 0.30))
        self.min_base_height = float(ecfg.get("min_base_height", 0.12))
        self.max_tilt_cos = np.cos(float(ecfg.get("max_tilt_rad", 1.0)))
        self.tracking_sigma = float(ecfg.get("tracking_sigma", 0.25))
        self.resample_steps = int(round(float(ecfg.get("command_resample_s", 10.0)) / self.dt))
        self.obs_scales = ecfg.get("obs_scales", {})
        self.recovery_stand_prob = float(ecfg.get("recovery_stand_prob", 0.1))
        self.settle_substeps = int(ecfg.get("recovery_settle_substeps", 250))

        # ---- models -------------------------------------------------------------
        self.xml = build_mjcf(self.robot_cfg, self.sim_dt)
        self.models = [mujoco.MjModel.from_xml_string(self.xml) for _ in range(self.num_envs)]
        self.datas = [mujoco.MjData(m) for m in self.models]
        m0 = self.models[0]
        self._check_joint_layout(m0)
        gid = lambda n: mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        bid = lambda n: mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.floor_geom = gid("floor")
        self.base_geom = gid("base_geom")
        self.foot_geoms = np.array([gid(f"{l}_foot") for l in LEG_NAMES])
        self.penalised_geoms = set(gid(f"{l}_{p}_geom") for l in LEG_NAMES for p in ("thigh", "calf", "hip"))
        self.base_body = bid("base")
        leg_bodies = [bid(f"{l}_{p}") for l in LEG_NAMES for p in ("hip", "thigh", "calf")]
        self.robot_geoms = np.array([g for g in range(m0.ngeom) if g != self.floor_geom])
        self.gyro_adr = m0.sensor_adr[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")]
        self.foot_radius = float(self.robot_cfg["leg"]["foot"]["radius"])

        self.default_dof_pos = default_joint_pos(self.robot_cfg)
        self.dof_lower, self.dof_upper = joint_limits(self.robot_cfg)
        mid, half = (self.dof_lower + self.dof_upper) / 2, (self.dof_upper - self.dof_lower) / 2
        soft = float(ecfg.get("soft_dof_pos_limit", 0.9))
        self.soft_lower, self.soft_upper = mid - soft * half, mid + soft * half

        pd = self.robot_cfg["pd"]
        self.kp = np.full(NUM_JOINTS, float(pd["kp"]))
        self.kd = np.full(NUM_JOINTS, float(pd["kd"]))
        self.motor = DCMotorModel.from_config(self.robot_cfg["motor"])
        act_cfg = cfg.get("actuator", {}) or {}
        self.actuator_net = None
        if act_cfg.get("type", "dc_motor") == "actuator_net":
            self.actuator_net = ActuatorNet.load(act_cfg["path"], self.motor.torque_limit)

        # ---- sim-to-real components ------------------------------------------
        self.randomizer = DomainRandomizer(cfg.get("domain_rand"), NUM_JOINTS)
        self.applier = ModelParamApplier(m0, self.base_body, leg_bodies)
        self.imu_noise = IMUNoiseModel(cfg.get("imu_noise"))
        self.obs_noise = ObservationNoise(cfg.get("obs_noise"))
        push = (cfg.get("domain_rand") or {}).get("push", {}) or {}
        self.push_enabled = bool(push.get("enabled", False)) and self.randomizer.enabled
        self.push_interval = max(1, int(round(float(push.get("interval_s", 8.0)) / self.dt)))
        self.push_max_vel = float(push.get("max_vel_xy", 0.5))
        self.rand = self.randomizer.sample(self.num_envs, self.rng)
        lat_hi = int(self.rand.latency_substeps.max(initial=0))
        if lat_hi > self.decimation:
            raise ValueError("latency_substeps upper bound must be <= decimation")

        # ---- task -------------------------------------------------------------
        self.reward_manager = RewardManager(cfg["rewards"], self.dt)
        self.commands_cfg = cfg.get("commands", {"init_ranges": {"lin_vel_x": [0, 0], "lin_vel_y": [0, 0],
                                                                  "ang_vel_yaw": [0, 0]}})
        self.curriculum = CommandCurriculum(self.commands_cfg)
        self._curriculum_scores: list[float] = []

        self.num_obs = ACTOR_OBS_DIM
        self.num_privileged_obs = ACTOR_OBS_DIM + PRIVILEGED_EXTRA_DIM
        self.num_actions = NUM_JOINTS

        n = self.num_envs
        self.actions = np.zeros((n, NUM_JOINTS))
        self.last_actions = np.zeros((n, NUM_JOINTS))
        self.prev_targets = np.tile(self.default_dof_pos, (n, 1))
        self.torques = np.zeros((n, NUM_JOINTS))
        self.last_dof_vel = np.zeros((n, NUM_JOINTS))
        self.commands = np.zeros((n, 3))
        self.episode_length = np.zeros(n, dtype=np.int64)
        self.feet_air_time = np.zeros((n, 4))
        self.last_contacts = np.zeros((n, 4), dtype=bool)
        self.gyro_bias = np.zeros((n, 3))
        self.tilt_bias = np.zeros((n, 3))
        self.episode_sums = {k: np.zeros(n) for k in self.reward_manager.names}
        hist = self.actuator_net.history if self.actuator_net else 1
        self.act_pos_err_hist = np.zeros((n, NUM_JOINTS, hist))
        self.act_vel_hist = np.zeros((n, NUM_JOINTS, hist))
        self._renderer = None
        self._state = None

    # ------------------------------------------------------------------ utils --
    @staticmethod
    def _check_joint_layout(m) -> None:
        for k, name in enumerate(JOINT_NAMES):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0 or m.jnt_qposadr[j] != 7 + k or m.jnt_dofadr[j] != 6 + k:
                raise ValueError(f"joint {name} not found at the expected qpos/qvel address")
            if m.actuator_trnid[k, 0] != j:
                raise ValueError(f"actuator {k} does not drive joint {name}")

    def _gather(self):
        n = self.num_envs
        q = np.empty((n, NUM_JOINTS))
        qd = np.empty((n, NUM_JOINTS))
        for i, d in enumerate(self.datas):
            q[i] = d.qpos[7:19]
            qd[i] = d.qvel[6:18]
        return q, qd

    def _gather_base(self):
        n = self.num_envs
        pos, quat, vel, gyro = np.empty((n, 3)), np.empty((n, 4)), np.empty((n, 3)), np.empty((n, 3))
        for i, d in enumerate(self.datas):
            pos[i] = d.qpos[0:3]
            quat[i] = d.qpos[3:7]
            vel[i] = d.qvel[0:3]
            gyro[i] = d.sensordata[self.gyro_adr:self.gyro_adr + 3]
        R = quat_to_rotmat(quat)
        lin_vel_b = np.einsum("nji,nj->ni", R, vel)
        proj_grav = -R[:, 2, :]  # R^T @ (0, 0, -1)
        return pos, quat, lin_vel_b, gyro, proj_grav

    def _contacts(self):
        n = self.num_envs
        feet = np.zeros((n, 4), dtype=bool)
        coll = np.zeros(n, dtype=np.int64)
        base = np.zeros(n, dtype=bool)
        foot_index = {int(g): k for k, g in enumerate(self.foot_geoms)}
        for i, d in enumerate(self.datas):
            for c in d.contact[:d.ncon]:
                g1, g2 = int(c.geom1), int(c.geom2)
                if g1 == self.floor_geom:
                    other = g2
                elif g2 == self.floor_geom:
                    other = g1
                else:
                    continue
                if other in foot_index:
                    feet[i, foot_index[other]] = True
                elif other == self.base_geom:
                    base[i] = True
                    coll[i] += 1
                elif other in self.penalised_geoms:
                    coll[i] += 1
        return feet, coll, base

    # ------------------------------------------------------------------ reset --
    def reset(self):
        self.reset_idx(np.arange(self.num_envs))
        obs, priv = self.compute_observations()
        return obs, priv

    def reset_idx(self, env_ids) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64)
        if env_ids.size == 0:
            return
        k = env_ids.size
        sample = self.randomizer.sample(k, self.rng)
        for j, i in enumerate(env_ids):
            for name in ("friction", "added_base_mass", "base_com_offset", "link_mass_scale",
                         "motor_strength", "kp_scale", "kd_scale", "latency_substeps"):
                getattr(self.rand, name)[i] = getattr(sample, name)[j]
            self.applier.apply(self.models[i], sample, j)

        fallen = np.zeros(k, dtype=bool)
        if self.task == "recovery":
            fallen = self.rng.random(k) >= self.recovery_stand_prob
        quats = yaw_quaternions(self.rng.uniform(-np.pi, np.pi, k))
        quats[fallen] = random_quaternions(int(fallen.sum()), self.rng)
        for j, i in enumerate(env_ids):
            m, d = self.models[i], self.datas[i]
            mujoco.mj_resetData(m, d)
            d.qpos[3:7] = quats[j]
            if fallen[j]:
                d.qpos[7:19] = self.rng.uniform(self.dof_lower, self.dof_upper)
            else:
                d.qpos[7:19] = self.default_dof_pos + self.rng.uniform(-0.1, 0.1, NUM_JOINTS)
                d.qvel[0:2] = self.rng.uniform(-0.3, 0.3, 2) if self.task == "walk" else 0.0
            self._place_on_ground(m, d, exact_feet=not fallen[j])
            if fallen[j]:
                # drop and let it come to rest passively, motors unpowered
                d.ctrl[:] = 0.0
                for _ in range(self.settle_substeps):
                    mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            self.prev_targets[i] = d.qpos[7:19]

        self.commands[env_ids] = 0.0 if self.task == "recovery" else self.curriculum.sample(k, self.rng)
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self.episode_length[env_ids] = 0
        self.feet_air_time[env_ids] = 0.0
        self.last_contacts[env_ids] = False
        self.act_pos_err_hist[env_ids] = 0.0
        self.act_vel_hist[env_ids] = 0.0
        gb, tb = self.imu_noise.sample_bias(k, self.rng)
        self.gyro_bias[env_ids], self.tilt_bias[env_ids] = gb, tb
        q, qd = self._gather()
        self.last_dof_vel[env_ids] = qd[env_ids]
        for key in self.episode_sums:
            self.episode_sums[key][env_ids] = 0.0

    def _place_on_ground(self, m, d, exact_feet: bool) -> None:
        mujoco.mj_kinematics(m, d)
        if exact_feet:
            lowest = np.min(d.geom_xpos[self.foot_geoms, 2]) - self.foot_radius
        else:
            lowest = np.min(d.geom_xpos[self.robot_geoms, 2] - m.geom_rbound[self.robot_geoms])
        d.qpos[2] += -lowest + (0.002 if exact_feet else 0.01)

    # ------------------------------------------------------------------- step --
    def step(self, actions):
        actions = np.clip(np.asarray(actions, dtype=np.float64), -self.clip_actions, self.clip_actions)
        self.last_actions[:] = self.actions
        self.actions[:] = actions
        targets = self.default_dof_pos + self.action_scale * actions
        kp = self.kp * self.rand.kp_scale
        kd = self.kd * self.rand.kd_scale
        latency = self.rand.latency_substeps[:, None]
        for sub in range(self.decimation):
            tgt = np.where(sub < latency, self.prev_targets, targets)
            q, qd = self._gather()
            if self.actuator_net is None:
                tau = pd_torque(tgt, q, qd, kp, kd)
                tau = self.motor(tau, qd, self.rand.motor_strength)
            else:
                self.act_pos_err_hist = np.roll(self.act_pos_err_hist, -1, axis=2)
                self.act_vel_hist = np.roll(self.act_vel_hist, -1, axis=2)
                self.act_pos_err_hist[..., -1] = tgt - q
                self.act_vel_hist[..., -1] = qd
                tau = self.actuator_net(self.act_pos_err_hist, self.act_vel_hist) * self.rand.motor_strength
            self.torques[:] = tau
            for i in range(self.num_envs):
                self.datas[i].ctrl[:] = tau[i]
                mujoco.mj_step(self.models[i], self.datas[i])
        self.prev_targets[:] = targets
        return self._post_physics_step()

    def _post_physics_step(self):
        self.episode_length += 1
        pos, quat, lin_vel_b, gyro, proj_grav = self._gather_base()
        q, qd = self._gather()
        feet, coll, base_contact = self._contacts()

        contact_filt = feet | self.last_contacts
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        self.feet_air_time += self.dt
        air_time_for_reward = self.feet_air_time.copy()
        self.feet_air_time *= ~contact_filt
        self.last_contacts = feet

        time_out = self.episode_length >= self.max_episode_length
        if self.task == "walk":
            terminated = base_contact | (proj_grav[:, 2] > -self.max_tilt_cos) | (pos[:, 2] < self.min_base_height)
        else:
            terminated = np.zeros(self.num_envs, dtype=bool)
        done = terminated | time_out

        state = {
            "commands": self.commands, "base_lin_vel": lin_vel_b, "base_ang_vel": gyro,
            "projected_gravity": proj_grav, "base_height": pos[:, 2],
            "base_height_target": self.base_height_target, "tracking_sigma": self.tracking_sigma,
            "torques": self.torques, "dof_vel": qd, "last_dof_vel": self.last_dof_vel, "dt": self.dt,
            "actions": self.actions, "last_actions": self.last_actions, "dof_pos": q,
            "default_dof_pos": self.default_dof_pos, "soft_lower": self.soft_lower,
            "soft_upper": self.soft_upper, "feet_air_time": air_time_for_reward,
            "first_contact": first_contact, "collisions": coll, "terminated": terminated,
            "time_out": time_out, "feet_contact": feet,
        }
        reward, terms = self.reward_manager.compute(state)
        for k, v in terms.items():
            self.episode_sums[k] += v
        self._state = state
        self.last_dof_vel[:] = qd

        info = {"time_outs": time_out.copy()}
        done_ids = np.nonzero(done)[0]
        if done_ids.size:
            info["episode"] = {f"rew_{k}": float(np.mean(v[done_ids])) / (self.max_episode_length * self.dt)
                               for k, v in self.episode_sums.items()}
            info["episode"]["length_s"] = float(np.mean(self.episode_length[done_ids])) * self.dt
            if self.task == "walk" and "tracking_lin_vel" in self.episode_sums:
                # normalised by the FULL episode length, so early falls count as poor tracking
                # (scoring only time-outs would be biased towards easy, near-zero commands)
                # Scores are pooled over at least num_envs finished episodes before deciding, so a
                # handful of easy stand-still episodes cannot widen the range on their own.
                scale = self.reward_manager.scales["tracking_lin_vel"]
                self._curriculum_scores.extend(
                    (self.episode_sums["tracking_lin_vel"][done_ids] / (self.max_episode_length * scale)).tolist())
                if len(self._curriculum_scores) >= max(self.num_envs, 32):
                    score = float(np.mean(self._curriculum_scores))
                    self._curriculum_scores = []
                    info["episode"]["curriculum_score"] = score
                    if self.curriculum.update(score):
                        info["curriculum_grown"] = True
            info["episode"]["cmd_vx_max"] = float(self.curriculum.ranges["lin_vel_x"][1])

        # perturbations and command resampling (only for envs that continue)
        alive = ~done
        if self.push_enabled:
            push_ids = np.nonzero(alive & (self.episode_length % self.push_interval == 0))[0]
            for i in push_ids:
                self.datas[i].qvel[0:2] += self.rng.uniform(-self.push_max_vel, self.push_max_vel, 2)
        if self.task == "walk":
            res_ids = np.nonzero(alive & (self.episode_length % self.resample_steps == 0))[0]
            if res_ids.size:
                self.commands[res_ids] = self.curriculum.sample(res_ids.size, self.rng)

        if self.auto_reset and done_ids.size:
            self.reset_idx(done_ids)
        obs, priv = self.compute_observations()
        info["privileged_obs"] = priv
        info["terminated"] = terminated
        return obs, reward.astype(np.float32), done, info

    # ------------------------------------------------------------ observations --
    def compute_observations(self):
        pos, quat, lin_vel_b, gyro, proj_grav = self._gather_base()
        q, qd = self._gather()
        dof_rel = q - self.default_dof_pos
        clean = build_actor_obs(proj_grav, gyro, self.commands, dof_rel, qd, self.actions, self.obs_scales)
        n_gyro, n_grav = self.imu_noise.apply(gyro, proj_grav, self.gyro_bias, self.tilt_bias, self.rng)
        n_pos, n_vel = self.obs_noise.apply(dof_rel, qd, self.rng)
        obs = build_actor_obs(n_grav, n_gyro, self.commands, n_pos, n_vel, self.actions, self.obs_scales)
        feet = self.last_contacts.astype(np.float64)
        r = self.rand
        priv = np.concatenate([
            clean, lin_vel_b * 2.0, pos[:, 2:3], feet, r.friction[:, None], r.added_base_mass[:, None],
            r.base_com_offset * 10.0, r.kp_scale.mean(1, keepdims=True), r.motor_strength.mean(1, keepdims=True),
        ], axis=1).astype(np.float32)
        return obs, priv

    # -------------------------------------------------------------- rendering --
    def render(self, env_id: int = 0, width: int = 640, height: int = 360, camera: str = "track") -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.models[env_id], height=height, width=width)
        self._renderer.update_scene(self.datas[env_id], camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
