"""
Base data infrastructure.
"""
from typing import List, Tuple, Literal, Optional

from enum import Enum
from dataclasses import dataclass, field

from data_infra.configs.normalization import *


# ========================== Data Types ========================== #

class DataType(Enum):
    POINT_CLOUD = "point_cloud"
    IMAGE = "image"
    DEPTH = "depth"
    POINT = "point"
    POSE = "pose"
    WRENCH = "wrench"
    FORCE = "force"
    TORQUE = "torque"
    TWIST = "twist"
    LINEAR_VELOCITY = "linear_velocity"
    ANGULAR_VELOCITY = "angular_velocity"
    JOINT = "joint"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE = "joint_torque"
    VECTOR = "vector"

    @staticmethod
    def from_str(s: str):
        return DataType(s)


SpatialDataTypes = [
    DataType.POINT_CLOUD,
    DataType.POINT,
    DataType.POSE,
    DataType.WRENCH,
    DataType.FORCE,
    DataType.TORQUE,
    DataType.TWIST,
    DataType.LINEAR_VELOCITY,
    DataType.ANGULAR_VELOCITY,
]


# ========================== Base Data Configs ========================== #

@dataclass(kw_only = True)
class BaseDataConfig:
    """ Base Data Config """
    type: DataType
    frame: Optional[str] = None
    aug_groups: List[str] = field(default_factory = list)


# ========================== Common Configs ========================== #

@dataclass(kw_only = True)
class PrefixConfig:
    color: str = "color"            # color/[camera_name]
    depth: str = "depth"            # depth/[camera_name]
    extrinsic: str = "extrinsic"    # extrinsic/[camera_name]
    intrinsic: str = "intrinsic"    # intrinsic/[camera_name]
    robot: str = "robot"            # robot/[robot_name]
    gripper: str = "gripper"        # gripper/[gripper_name]
    hand: str = "hand"              # hand/[hand_name]
