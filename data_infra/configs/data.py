"""
Data target infrastructure.

Data target configurations should only includes all other information to construct the data input for the policy.
"""
from typing import List, Union, Tuple, Literal, Optional

from enum import Enum
from dataclasses import dataclass, field

from data_infra.configs.base import *
from data_infra.configs.data_source import *
from data_infra.configs.normalization import *

from utils.transforms.rotation import RotationType


@dataclass(kw_only = True)
class BaseTargetConfig(BaseDataConfig):
    """ Base Target Data Config """
    src: Union[BaseSourceConfig, List[BaseSourceConfig]]
    norm_config: BaseNormalizationConfig = EmptyNormalizationConfig()
    relative_key: Optional[str] = None


@dataclass(kw_only = True)
class ImageConfig(BaseTargetConfig):
    """ Image Data Config """
    type: DataType = field(default = DataType.IMAGE, init = False)
    src: ImageSourceConfig
    size: Optional[Tuple[int, int]] = None
    interp_mode: Optional[str] = None
    aug_groups: List[str] = field(default_factory = lambda: [], init = False)


@dataclass(kw_only = True)
class DepthConfig(BaseTargetConfig):
    """ Depth Configuration. """
    type: DataType = field(default = DataType.DEPTH, init = False)
    src: DepthSourceConfig
    size: Optional[Tuple[int, int]] = None
    interp_mode: Optional[str] = None
    aug_groups: List[str] = field(default_factory = lambda: [], init = False)


@dataclass(kw_only = True)
class PointConfig(BaseTargetConfig):
    """ Coordinate Configuration. """
    type: DataType = field(default = DataType.POINT, init = False)
    src: Union[Depth2PointSourceConfig, PointSourceConfig]


@dataclass(kw_only = True)
class PoseConfig(BaseTargetConfig):
    """ Pose Data Config """
    type: DataType = field(default = DataType.POSE, init = False)
    src: PoseSourceConfig
    rotation_rep: RotationType
    convention: Optional[str] = None


@dataclass(kw_only = True)
class WrenchConfig(BaseTargetConfig):
    """ Wrench Data Config """
    type: DataType = field(default = DataType.WRENCH, init = False)
    src: WrenchSourceConfig
    rotation_only: bool = True
    
    
@dataclass(kw_only = True)
class ForceConfig(BaseTargetConfig):
    """ Force Data Config """
    type: DataType = field(default = DataType.FORCE, init = False)
    src: ForceSourceConfig


@dataclass(kw_only = True)
class TorqueConfig(BaseTargetConfig):
    """ Torque Data Config """
    type: DataType = field(default = DataType.TORQUE, init = False)
    src: TorqueSourceConfig


@dataclass(kw_only = True)
class TwistConfig(BaseTargetConfig):
    """ Twist Data Config """
    type: DataType = field(default = DataType.TWIST, init = False)
    src: TwistSourceConfig
    rotation_only: bool = True


@dataclass(kw_only = True)
class LinearVelocityConfig(BaseTargetConfig):
    """ Linear Velocity Data Config """
    type: DataType = field(default = DataType.LINEAR_VELOCITY, init = False)
    src: LinearVelocitySourceConfig


@dataclass(kw_only = True)
class AngularVelocityConfig(BaseTargetConfig):
    """ Angular Velocity Data Config """
    type: DataType = field(default = DataType.ANGULAR_VELOCITY, init = False)
    src: AngularVelocitySourceConfig


@dataclass(kw_only = True)
class JointConfig(BaseTargetConfig):
    """ Joint Data Config """
    type: DataType = field(default = DataType.JOINT, init = False)
    src: JointSourceConfig
    aug_groups: List[str] = field(default_factory = lambda: [], init = False)


@dataclass(kw_only = True)
class JointVelocityConfig(BaseTargetConfig):
    """ Joint Velocity Data Config """
    type: DataType = field(default = DataType.JOINT_VELOCITY, init = False)
    src: JointVelocitySourceConfig
    aug_groups: List[str] = field(default_factory = lambda: [], init = False)


@dataclass(kw_only = True)
class JointTorqueConfig(BaseTargetConfig):
    """ Joint Torque Data Config """
    type: DataType = field(default = DataType.JOINT_TORQUE, init = False)
    src: JointTorqueSourceConfig
    aug_groups: List[str] = field(default_factory = lambda: [], init = False)


@dataclass(kw_only = True)
class VectorConfig(BaseTargetConfig):
    """ Vector Data Config """
    type: DataType = field(default = DataType.VECTOR, init = False)
    src: VectorSourceConfig
    aug_groups: List[str] = field(default_factory = lambda: [], init = False)


@dataclass(kw_only = True)
class PointCloudConfig(BaseTargetConfig):
    """
    Point Cloud Configuration.
    """
    type: DataType = field(init = False, default = DataType.POINT_CLOUD)
    src: List[Depth2PointSourceConfig]

    voxelization: bool = True
    voxel_size: float = 0.005
    
    fixed_number: bool = False
    num_points: int = 1024
    sampling_method: str = 'fps'
    
    backend: Literal['MinkowskiEngine'] = 'MinkowskiEngine'    
