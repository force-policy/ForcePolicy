"""
Data source infrastructure.

Data source configurations should only includes all necessary information to construct/fetch the data.
Construction is limited to the point cloud construction from depth.
"""
from typing import List, Union, Tuple, Literal, Optional

from enum import Enum
from dataclasses import dataclass, field

from data_infra.configs.base import *
from data_infra.configs.point_mask import *

from utils.transforms.rotation import RotationType


@dataclass(kw_only = True)
class BaseSourceConfig(BaseDataConfig):
    """ Base Source Data Config """
    input_key: str


@dataclass(kw_only = True)
class ImageSourceConfig(BaseSourceConfig):
    """ Image Source Data Config """
    type: DataType = field(default = DataType.IMAGE, init = False)
    input_key: str = field(default = None, init = False)
    camera_name: str


@dataclass(kw_only = True)
class DepthSourceConfig(BaseSourceConfig):
    """ Depth Source Configuration. """
    type: DataType = field(default = DataType.DEPTH, init = False)
    input_key: str = field(default = None, init = False)
    camera_name: str


@dataclass(kw_only = True)
class Depth2PointSourceConfig(DepthSourceConfig):
    """ Depth to Point Source Configuration. """
    # construct point cloud with image
    image_source_config: Optional[ImageSourceConfig] = None
    size: Optional[Tuple[int, int]] = None

    # hole filling on depth image
    fill_hole: bool = False
    fill_hole_kernel_size: Tuple[int, int] = field(default_factory = lambda: (60, 64))

    # whether to flatten to B * N * (3 + feat)
    flatten: bool = True
    camera_mask_config: Optional[Union[PointMaskConfig, GroupPointMaskConfig, List[PointMaskConfig]]] = None  # Point mask in camera frame
    world_mask_config: Optional[Union[PointMaskConfig, GroupPointMaskConfig, List[PointMaskConfig]]] = None  # Point mask in base frame

    # pooling if any
    pooling_size: Optional[Tuple[int, int]] = None
    pooling_interp_mode: str = "area"


@dataclass(kw_only = True)
class PointSourceConfig(BaseSourceConfig):
    """ Point Source Data Config """
    type: DataType = field(default = DataType.POINT, init = False)


@dataclass(kw_only = True)
class PoseSourceConfig(BaseSourceConfig):
    """ Pose Source Data Config """
    type: DataType = field(default = DataType.POSE, init = False)
    rotation_rep: RotationType
    convention: Optional[str] = None


@dataclass(kw_only = True)
class WrenchSourceConfig(BaseSourceConfig):
    """ Wrench Source Data Config """
    type: DataType = field(default = DataType.WRENCH, init = False)


@dataclass(kw_only = True)
class ForceSourceConfig(BaseSourceConfig):
    """ Force Source Data Config """
    type: DataType = field(default = DataType.FORCE, init = False)


@dataclass(kw_only = True)
class TorqueSourceConfig(BaseSourceConfig):
    """ Torque Source Data Config """
    type: DataType = field(default = DataType.TORQUE, init = False)


@dataclass(kw_only = True)
class TwistSourceConfig(BaseSourceConfig):
    """ Twist Source Data Config """
    type: DataType = field(default = DataType.TWIST, init = False)


@dataclass(kw_only = True)
class LinearVelocitySourceConfig(BaseSourceConfig):
    """ Linear Velocity Source Data Config """
    type: DataType = field(default = DataType.LINEAR_VELOCITY, init = False)


@dataclass(kw_only = True)
class AngularVelocitySourceConfig(BaseSourceConfig):
    """ Angular Velocity Source Data Config """
    type: DataType = field(default = DataType.ANGULAR_VELOCITY, init = False)


@dataclass(kw_only = True)
class JointSourceConfig(BaseSourceConfig):
    """ Joint Source Data Config """
    type: DataType = field(default = DataType.JOINT, init = False)


@dataclass(kw_only = True)
class JointVelocitySourceConfig(BaseSourceConfig):
    """ Joint Velocity Source Data Config """
    type: DataType = field(default = DataType.JOINT_VELOCITY, init = False)


@dataclass(kw_only = True)
class JointTorqueSourceConfig(BaseSourceConfig):
    """ Joint Torque Source Data Config """
    type: DataType = field(default = DataType.JOINT_TORQUE, init = False)

@dataclass(kw_only = True)
class VectorSourceConfig(BaseSourceConfig):
    """ Vector Source Data Config """
    type: DataType = field(default = DataType.VECTOR, init = False)
