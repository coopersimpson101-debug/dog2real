"""Real-robot runtime: fixed-rate 50 Hz policy loop with a safety layer.

Designed for a Raspberry Pi 4/5 or Jetson running Linux with a CAN adapter
(MCP2515 HAT, USB-CAN such as candleLight / CANable via socketcan). The same
class drives MuJoCo in sim-in-the-loop mode (``SimRobot`` + ``SimClock``).

Tick (every ``policy_dt``)::

    read joints + IMU  ->  safety.check_state  ->  build obs (shared code with sim)
      ->  policy  ->  q* = q_default + action_scale * a  ->  safety.filter  ->  motors

Phases: ``ramp`` (smoothly move from the current pose to the standing pose,
only for the walking policy) -> ``policy`` -> ``damping`` (after any fault or
on shutdown).

For best timing on Linux run with ``chrt -f 80`` (SCHED_FIFO) and pin to an
isolated core (``taskset``); see docs/deployment.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dog2real.deploy.hal.base import IMU, MotorBus
from dog2real.deploy.safety import SafetyLayer
from dog2real.observation import build_actor_obs
from dog2real.robot.model_builder import default_joint_pos


class RealClock:
    """Absolute-deadline scheduler: coarse ``sleep`` then a short busy-wait for precision."""

    def __init__(self, period: float, spin_s: float = 0.0008):
        self.period = period
        self.spin_s = spin_s
        self.next = None

    def now(self) -> float:
        return time.perf_counter()

    def wait(self) -> None:
        if self.next is None:
            self.next = time.perf_counter() + self.period
        remaining = self.next - time.perf_counter()
        if remaining > self.spin_s:
            time.sleep(remaining - self.spin_s)
        while time.perf_counter() < self.next:
            pass
        self.next += self.period
        # if we fell more than one period behind, re-anchor instead of bursting
        if time.perf_counter() > self.next:
            self.next = time.perf_counter() + self.period


class SimClock:
    """Lock-step clock: ``wait`` advances the simulator by one policy period."""

    lockstep = True  # wall-clock period statistics are meaningless in lock-step mode

    def __init__(self, period: float, sim_robot):
        self.period = period
        self.sim = sim_robot

    def now(self) -> float:
        return self.sim.time

    def wait(self) -> None:
        self.sim.advance(self.period)


@dataclass
class TimingStats:
    period: float
    compute: list = field(default_factory=list)
    intervals: list = field(default_factory=list)
    overruns: int = 0

    def add(self, compute_s: float, interval_s: float | None) -> None:
        self.compute.append(compute_s)
        if interval_s is not None:
            self.intervals.append(interval_s)
        if compute_s > self.period:
            self.overruns += 1

    def summary(self) -> dict:
        c = np.asarray(self.compute) * 1e3
        i = np.asarray(self.intervals) * 1e3
        out = {"ticks": len(c), "overruns": self.overruns}
        if c.size:
            out.update(compute_mean_ms=float(c.mean()), compute_p99_ms=float(np.percentile(c, 99)),
                       compute_max_ms=float(c.max()))
        if i.size:
            out.update(period_mean_ms=float(i.mean()), period_std_ms=float(i.std()),
                       jitter_p99_ms=float(np.percentile(np.abs(i - self.period * 1e3), 99)))
        return out


class RobotRuntime:
    def __init__(self, robot_cfg: dict, deploy_cfg: dict, policy, bus: MotorBus, imu: IMU, clock,
                 safety: SafetyLayer, command=(0.0, 0.0, 0.0), log_path: str | Path | None = None):
        self.robot_cfg = robot_cfg
        c = deploy_cfg.get("control", {})
        self.policy = policy
        self.bus, self.imu, self.clock, self.safety = bus, imu, clock, safety
        self.policy_dt = float(c.get("policy_dt", 0.02))
        self.action_scale = float(c.get("action_scale", 0.25))
        self.clip_actions = float(c.get("clip_actions", 10.0))
        self.obs_scales = c.get("obs_scales", {})
        self.kp = np.full(12, float(c.get("kp", robot_cfg["pd"]["kp"])))
        self.kd = np.full(12, float(c.get("kd", robot_cfg["pd"]["kd"])))
        self.ramp_s = float(c.get("ramp_s", 2.0))
        self.startup = c.get("startup", "ramp")  # "ramp" (walking) or "policy" (get-up)
        self.default_q = default_joint_pos(robot_cfg)
        self.command = np.asarray(command, dtype=np.float64)
        self.last_action = np.zeros(12)
        self.phase = "init"
        self.timing = TimingStats(self.policy_dt)
        self.log_path = Path(log_path) if log_path else None
        self.log: dict[str, list] = {k: [] for k in ("t", "q", "dq", "q_des", "action", "gravity", "gyro", "phase")}
        self.tick_count = 0
        self._ramp_start_q = None
        self._ramp_ticks = max(1, int(round(self.ramp_s / self.policy_dt)))
        self._last_tick_t = None

    def start(self) -> None:
        self.bus.enable()
        js = self.bus.read()
        self._ramp_start_q = js.q.copy()
        self.phase = "ramp" if self.startup == "ramp" else "policy"

    def tick(self) -> dict:
        t0 = time.perf_counter()
        js = self.bus.read()
        imu = self.imu.read()
        self.safety.check_state(js.q, js.dq, imu.projected_gravity, js.comm_ok)
        action = np.zeros(12)
        if not self.safety.ok:
            self.phase = "damping"
            q_des = js.q
        elif self.phase == "ramp":
            alpha = min(1.0, (self.tick_count + 1) / self._ramp_ticks)
            alpha = 0.5 - 0.5 * np.cos(np.pi * alpha)  # smooth start / stop
            q_des = (1 - alpha) * self._ramp_start_q + alpha * self.default_q
            if self.tick_count + 1 >= self._ramp_ticks:
                self.phase = "policy"
        else:
            obs = build_actor_obs(imu.projected_gravity, imu.gyro, self.command, js.q - self.default_q,
                                  js.dq, self.last_action, self.obs_scales)
            action = np.clip(np.asarray(self.policy(obs), dtype=np.float64), -self.clip_actions, self.clip_actions)
            self.last_action = action
            q_des = self.default_q + self.action_scale * action
        cmd = self.safety.filter(q_des, js.q, js.dq, self.kp, self.kd)
        self.bus.command(cmd.q_des, cmd.dq_des, cmd.kp, cmd.kd, cmd.tau_ff)
        compute = time.perf_counter() - t0
        interval = None
        if self._last_tick_t is not None and not getattr(self.clock, "lockstep", False):
            interval = t0 - self._last_tick_t
        self._last_tick_t = t0
        self.timing.add(compute, interval)
        self.tick_count += 1
        entry = {"t": self.clock.now(), "q": js.q, "dq": js.dq, "q_des": cmd.q_des, "action": action,
                 "gravity": imu.projected_gravity, "gyro": imu.gyro, "phase": self.phase}
        if self.log_path:
            for k, v in entry.items():
                self.log[k].append(v)
        return entry

    def run(self, duration_s: float | None = None, max_ticks: int | None = None) -> dict:
        if self.phase == "init":
            self.start()
        n = max_ticks if max_ticks is not None else (int(duration_s / self.policy_dt) if duration_s else None)
        try:
            while n is None or self.tick_count < n:
                self.tick()
                self.clock.wait()
        except KeyboardInterrupt:
            self.safety.estop("keyboard interrupt")
        finally:
            self.shutdown()
        return self.timing.summary()

    def shutdown(self, damping_s: float = 1.0) -> None:
        """Brake the joints for ``damping_s`` then disable the motors."""
        self.safety.estop(self.safety.fault or "shutdown")
        for _ in range(int(damping_s / self.policy_dt)):
            js = self.bus.read()
            cmd = self.safety.filter(js.q, js.q, js.dq, self.kp, self.kd)
            self.bus.command(cmd.q_des, cmd.dq_des, cmd.kp, cmd.kd, cmd.tau_ff)
            self.clock.wait()
        self.bus.disable()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.log_path, **{k: np.asarray(v) for k, v in self.log.items()})
