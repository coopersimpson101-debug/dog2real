# Replacing the generated model with your URDF / CAD export

The procedural model from `dog2real/robot/model_builder.py` is deliberately
simple (boxes and capsules with rod inertias). Once you have a CAD model, an
exported URDF/MJCF gives better inertias and collision shapes.

## What the environment relies on

The env does **not** depend on the geometry, only on names and ordering:

| Element | Required name(s) |
|---|---|
| floating base body | `base`, with a free joint as the **first** joint (qpos[0:7]) |
| 12 hinge joints, in this order right after the free joint | `FL_abad FL_hip FL_knee FR_abad ... RR_knee` |
| 12 `motor` actuators, same order | any name; `ctrlrange` = torque limit |
| foot collision geoms | `FL_foot`, `FR_foot`, `RL_foot`, `RR_foot` |
| base collision geom | `base_geom` (contact = termination for walking) |
| penalised collision geoms | `<leg>_thigh_geom`, `<leg>_calf_geom`, `<leg>_hip_geom` |
| world floor | `floor` |
| IMU site + sensors | site `imu`; sensors `imu_gyro`, `imu_quat`, `imu_acc` |
| leg bodies (mass randomisation) | `<leg>_hip`, `<leg>_thigh`, `<leg>_calf` |

`VecQuadrupedEnv._check_joint_layout` raises an error if the layout differs.

## Steps

1. Export URDF from your CAD tool (Onshape: `onshape-to-robot`; Fusion 360:
   `fusion2urdf`; SolidWorks: `sw_urdf_exporter`).
2. Convert to MJCF: `python -c "import mujoco; m = mujoco.MjModel.from_xml_path('robot.urdf'); mujoco.mj_saveLastXML('robot.xml', m)"`
   (add `<mujoco><compiler discardvisual="false"/></mujoco>` inside the URDF to keep meshes).
3. Edit `robot.xml`: add the free joint on the base, rename joints/geoms to the
   table above, add `<motor>` actuators, the floor, the IMU site and sensors.
   Copy the `<default>` block from the generated model so the robot geoms collide
   with the floor (`conaffinity`) but not with each other.
4. Replace meshes used for **collision** by primitives (capsules for links,
   spheres for feet): mesh-mesh contacts are slow and noisy.
5. Make `build_mjcf` return your file: simplest is a small change in
   `model_builder.build_mjcf` such as `if robot.get("mjcf_path"): return Path(robot["mjcf_path"]).read_text()`
   and set `mjcf_path:` in `robot.yaml`. Keep `joints`, `motor`, `pd` in the YAML --
   the env, safety layer and runtime still read limits and gains from there.
6. Run `pytest tests/test_model_and_env.py` -- the layout and standing tests
   must still pass.
