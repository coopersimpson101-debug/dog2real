"""Generate an MJCF model of a generic 12-DoF quadruped from ``robot.yaml``.

Kinematic convention (matches most open-source quadrupeds):

* base frame: x forward, y left, z up; IMU at the base origin.
* each leg: ``abad`` (rotation about +x) -> ``hip`` (about +y) -> ``knee`` (about +y).
* at zero joint angles the thigh and calf point straight down (-z);
  positive hip angle swings the foot backwards, negative knee folds the calf
  forward-under, so a "knee-backwards" stance is hip > 0, knee < 0.

Actuators are raw ``motor`` (torque) actuators. PD control and the DC-motor
torque-speed model are applied in Python at the physics rate (see
``dog2real.sim.actuator``) so that exactly the same control law can be mirrored
by the real-robot runtime.

If you have a URDF / CAD export of your robot, see ``docs/urdf_import.md``.
The environment only relies on the joint, body and geom *names* produced here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dog2real import JOINT_TYPES, LEG_NAMES

LEG_SIGNS = {"FL": (1, 1), "FR": (1, -1), "RL": (-1, 1), "RR": (-1, -1)}


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def _rod_inertia(mass: float, length: float, radius: float) -> np.ndarray:
    """Diagonal inertia of a solid cylinder whose axis is the local z axis."""
    ixx = mass * (3 * radius**2 + length**2) / 12.0
    izz = mass * radius**2 / 2.0
    return np.array([ixx, ixx, izz])


def _box_inertia(mass: float, size) -> np.ndarray:
    lx, ly, lz = size
    return mass / 12.0 * np.array([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])


def build_mjcf(robot: dict[str, Any], sim_dt: float = 0.002, floor_friction: float = 1.0) -> str:
    """Return an MJCF XML string for the robot described by ``robot``."""
    b = robot["base"]
    leg = robot["leg"]
    jnt = robot["joints"]
    mot = robot["motor"]
    contact = robot.get("contact", {})
    foot_mu = contact.get("foot_friction", 0.9)
    body_mu = contact.get("body_friction", 0.6)

    size = np.asarray(b["size"], dtype=float)
    half = size / 2.0
    hx, hy = leg["hip_offset"]
    l_abad = leg["abad"]["length"]
    l1 = leg["thigh"]["length"]
    l2 = leg["calf"]["length"]
    r_foot = leg["foot"]["radius"]
    tlim = float(mot["torque_limit"])

    # Spawn high enough that the default pose does not intersect the floor;
    # the environment re-computes the exact height at reset.
    z0 = l1 + l2 + r_foot + 0.05

    base_inertia = _box_inertia(b["mass"], size)
    abad_inertia = _rod_inertia(leg["abad"]["mass"], l_abad, leg["abad"]["radius"])
    thigh_inertia = _rod_inertia(leg["thigh"]["mass"], l1, leg["thigh"]["radius"])
    calf_mass = leg["calf"]["mass"] + leg["foot"]["mass"]
    calf_inertia = _rod_inertia(calf_mass, l2, leg["calf"]["radius"])

    legs_xml = []
    actuators_xml = []
    for name in LEG_NAMES:
        sx, sy = LEG_SIGNS[name]
        r = jnt["abad"]["range"]
        legs_xml.append(f"""
      <body name="{name}_hip" pos="{_fmt([sx * hx, sy * hy, 0.0])}">
        <joint name="{name}_abad" axis="1 0 0" range="{_fmt(r)}"/>
        <inertial pos="0 {sy * l_abad / 2:.6g} 0" mass="{leg['abad']['mass']}" diaginertia="{_fmt(abad_inertia[[0, 2, 1]])}"/>
        <geom name="{name}_hip_geom" class="body" type="cylinder" size="{leg['abad']['radius']:.6g} {l_abad / 2:.6g}"
              pos="0 {sy * l_abad / 2:.6g} 0" zaxis="0 1 0" rgba="0.25 0.25 0.3 1"/>
        <body name="{name}_thigh" pos="0 {sy * l_abad:.6g} 0">
          <joint name="{name}_hip" axis="0 1 0" range="{_fmt(jnt['hip']['range'])}"/>
          <inertial pos="0 0 {-l1 / 2:.6g}" mass="{leg['thigh']['mass']}" diaginertia="{_fmt(thigh_inertia)}"/>
          <geom name="{name}_thigh_geom" class="body" type="capsule" fromto="0 0 0 0 0 {-l1:.6g}" size="{leg['thigh']['radius']:.6g}" rgba="0.85 0.45 0.1 1"/>
          <body name="{name}_calf" pos="0 0 {-l1:.6g}">
            <joint name="{name}_knee" axis="0 1 0" range="{_fmt(jnt['knee']['range'])}"/>
            <inertial pos="0 0 {-l2 / 2:.6g}" mass="{calf_mass:.6g}" diaginertia="{_fmt(calf_inertia)}"/>
            <geom name="{name}_calf_geom" class="body" type="capsule" fromto="0 0 0 0 0 {-(l2 - r_foot):.6g}" size="{leg['calf']['radius']:.6g}" rgba="0.2 0.2 0.22 1"/>
            <geom name="{name}_foot" class="foot" type="sphere" pos="0 0 {-l2:.6g}" size="{r_foot:.6g}"/>
            <site name="{name}_foot_site" pos="0 0 {-l2:.6g}" size="0.005"/>
          </body>
        </body>
      </body>""")
        for jt in JOINT_TYPES:
            actuators_xml.append(
                f'    <motor name="{name}_{jt}_motor" joint="{name}_{jt}" ctrlrange="{-tlim:.6g} {tlim:.6g}"/>'
            )

    xml = f"""<mujoco model="{robot.get('name', 'dog2real')}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{sim_dt}" integrator="implicitfast" cone="pyramidal"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
  </visual>
  <default>
    <joint damping="{jnt.get('damping', 0.0)}" armature="{jnt.get('armature', 0.0)}" frictionloss="{jnt.get('frictionloss', 0.0)}"/>
    <!-- robot geoms collide with the world (conaffinity 1) but not with each other -->
    <default class="body">
      <geom contype="1" conaffinity="0" condim="3" friction="{body_mu} 0.005 0.0001"/>
    </default>
    <default class="foot">
      <geom contype="1" conaffinity="0" condim="3" friction="{foot_mu} 0.005 0.0001" rgba="0.1 0.1 0.1 1"/>
    </default>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.85 0.88" rgb2="0.62 0.66 0.7" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.6 0.75 0.9" rgb2="0.95 0.97 1" width="256" height="256"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 4" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="1" condim="3" friction="{floor_friction} 0.005 0.0001"/>
    <body name="base" pos="0 0 {z0:.6g}">
      <freejoint name="root"/>
      <inertial pos="{_fmt(b.get('com_offset', [0, 0, 0]))}" mass="{b['mass']}" diaginertia="{_fmt(base_inertia)}"/>
      <geom name="base_geom" class="body" type="box" size="{_fmt(half)}" rgba="0.9 0.9 0.92 1"/>
      <site name="imu" pos="0 0 0" size="0.01"/>
      <camera name="track" mode="trackcom" pos="0.25 -1.05 0.35" xyaxes="1 0.2 0 -0.06 0.3 1"/>{''.join(legs_xml)}
    </body>
  </worldbody>
  <actuator>
{chr(10).join(actuators_xml)}
  </actuator>
  <sensor>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
    <gyro name="imu_gyro" site="imu"/>
    <accelerometer name="imu_acc" site="imu"/>
  </sensor>
</mujoco>
"""
    return xml


def load_model(robot: dict[str, Any], sim_dt: float = 0.002):
    """Build and compile the MuJoCo model."""
    import mujoco

    return mujoco.MjModel.from_xml_string(build_mjcf(robot, sim_dt))


def default_joint_pos(robot: dict[str, Any]) -> np.ndarray:
    """Standing pose, ordered FL, FR, RL, RR x (abad, hip, knee)."""
    j = robot["joints"]
    return np.array([j[t]["default"] for _ in LEG_NAMES for t in JOINT_TYPES], dtype=np.float64)


def joint_limits(robot: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    j = robot["joints"]
    lo = np.array([j[t]["range"][0] for _ in LEG_NAMES for t in JOINT_TYPES], dtype=np.float64)
    hi = np.array([j[t]["range"][1] for _ in LEG_NAMES for t in JOINT_TYPES], dtype=np.float64)
    return lo, hi
