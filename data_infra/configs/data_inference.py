"""
Inference data infrastructure.

Data target configurations should only includes all other information to construct the data input for the policy.
"""
from typing import List, Union, Tuple, Literal, Optional

from enum import Enum
from dataclasses import dataclass, field

from data_infra.configs.base import *
from data_infra.configs.normalization import *

from utils.transforms.rotation import RotationType


@dataclass(kw_only = True)
class BaseInferenceConfig(BaseDataConfig):
    """ Base Inference Data Config """
    src_key: str
    norm_config: BaseNormalizationConfig = EmptyNormalizationConfig()
    relative_key: Optional[str] = None
    aug_groups: List[str] = field(default_factory = list, init = False)


@dataclass(kw_only = True)
class PointInferenceConfig(BaseInferenceConfig):
    """ Point Inference Data Config """
    type: DataType = field(default = DataType.POINT, init = False)
    src_frame: str


@dataclass(kw_only = True)
class PoseInferenceConfig(BaseInferenceConfig):
    """ Pose Inference Data Config """
    type: DataType = field(default = DataType.POSE, init = False)
    src_frame: str
    src_rotation_rep: RotationType
    src_convention: Optional[str] = None
    rotation_rep: RotationType
    convention: Optional[str] = None


@dataclass(kw_only = True)
class WrenchInferenceConfig(BaseInferenceConfig):
    """ Wrench Inference Data Config """
    type: DataType = field(default = DataType.WRENCH, init = False)
    src_frame: str
    rotation_only: bool = True
    
    
@dataclass(kw_only = True)
class ForceInferenceConfig(BaseInferenceConfig):
    """ Force Inference Data Config """
    type: DataType = field(default = DataType.FORCE, init = False)
    src_frame: str


@dataclass(kw_only = True)
class TorqueInferenceConfig(BaseInferenceConfig):
    """ Torque Inference Data Config """
    type: DataType = field(default = DataType.TORQUE, init = False)
    src_frame: str


@dataclass(kw_only = True)
class TwistInferenceConfig(BaseInferenceConfig):
    """ Twist Inference Data Config """
    type: DataType = field(default = DataType.TWIST, init = False)
    src_frame: str
    rotation_only: bool = True


@dataclass(kw_only = True)
class LinearVelocityInferenceConfig(BaseInferenceConfig):
    """ Linear Velocity Inference Data Config """
    type: DataType = field(default = DataType.LINEAR_VELOCITY, init = False)
    src_frame: str


@dataclass(kw_only = True)
class AngularVelocityInferenceConfig(BaseInferenceConfig):
    """ Angular Velocity Inference Data Config """
    type: DataType = field(default = DataType.ANGULAR_VELOCITY, init = False)
    src_frame: str


@dataclass(kw_only = True)
class JointInferenceConfig(BaseInferenceConfig):
    """ Joint Inference Data Config """
    type: DataType = field(default = DataType.JOINT, init = False)


@dataclass(kw_only = True)
class JointVelocityInferenceConfig(BaseInferenceConfig):
    """ Joint Velocity Inference Data Config """
    type: DataType = field(default = DataType.JOINT_VELOCITY, init = False)


@dataclass(kw_only = True)
class JointTorqueInferenceConfig(BaseInferenceConfig):
    """ Joint Torque Inference Data Config """
    type: DataType = field(default = DataType.JOINT_TORQUE, init = False)


@dataclass(kw_only = True)
class VectorInferenceConfig(BaseInferenceConfig):
    """ Vector Inference Data Config """
    type: DataType = field(default = DataType.VECTOR, init = False)
