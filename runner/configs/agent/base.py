from typing import Any, Dict, List, Union
from dataclasses import dataclass, field

import numpy as np

from data_infra.configs.base import PrefixConfig


@dataclass(kw_only = True)
class PlatformConfig:
    """ Platform Config """
    camera_names: List[str] = field(default_factory = list)
    robot_names: List[str] = field(default_factory = list)
    gripper_names: List[str] = field(default_factory = list)
    hand_names: List[str] = field(default_factory = list)


@dataclass(kw_only = True)
class BaseAgentConfig:
    platform_config: PlatformConfig
    prefix_config: PrefixConfig = field(default_factory = PrefixConfig)
    intrinsics: Dict[str, np.ndarray] = field(default_factory = dict)
    extrinsics: Dict[str, np.ndarray] = field(default_factory = dict)
    robot_poses: Dict[str, np.ndarray] = field(default_factory = dict)
    
    agent_action_providers: Dict[str, str] = field(default_factory = dict)


@dataclass(kw_only = True)
class AgentObsKeysConfig:
    color_keys: List[str] = field(default_factory = list)
    depth_keys: List[str] = field(default_factory = list)
    intrinsic_keys: List[str] = field(default_factory = list)
    extrinsic_keys: List[str] = field(default_factory = list)
    robot_pose_keys: List[str] = field(default_factory = list)
    lowdim_provider_keys: List[str] = field(default_factory = list)
    