"""Generate the MJCF for configs/robot.yaml, print a summary, optionally open the viewer.

    python scripts/build_model.py --out build/dog2real.xml
    python scripts/build_model.py --view          # interactive MuJoCo viewer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from dog2real import LEG_NAMES  # noqa: E402
from dog2real.config import load_yaml  # noqa: E402
from dog2real.robot.model_builder import build_mjcf, default_joint_pos  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="configs/robot.yaml")
    p.add_argument("--out")
    p.add_argument("--view", action="store_true")
    args = p.parse_args()

    robot = load_yaml(args.robot)
    xml = build_mjcf(robot)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qpos[7:19] = default_joint_pos(robot)
    mujoco.mj_kinematics(m, d)
    feet = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_foot") for leg in LEG_NAMES]
    stand_h = d.qpos[2] - (np.min(d.geom_xpos[feet, 2]) - robot["leg"]["foot"]["radius"])
    print(f"model {robot.get('name')}: nq={m.nq} nv={m.nv} nu={m.nu} bodies={m.nbody}")
    print(f"total mass {mujoco.mj_getTotalmass(m):.2f} kg; base height in default pose {stand_h:.3f} m")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(xml, encoding="utf-8")
        print(f"wrote {args.out}")
    if args.view:
        from mujoco import viewer

        d.qpos[2] = stand_h + 0.05
        viewer.launch(m, d)


if __name__ == "__main__":
    main()
