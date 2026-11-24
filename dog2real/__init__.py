"""dog2real: sim-to-real reinforcement learning for DIY 12-DoF quadrupeds."""

__version__ = "0.1.0"

LEG_NAMES = ("FL", "FR", "RL", "RR")
JOINT_TYPES = ("abad", "hip", "knee")
JOINT_NAMES = tuple(f"{leg}_{j}" for leg in LEG_NAMES for j in JOINT_TYPES)
NUM_JOINTS = len(JOINT_NAMES)
