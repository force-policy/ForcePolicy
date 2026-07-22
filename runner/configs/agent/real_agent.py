from typing import Any, Dict, List, Union
from dataclasses import dataclass, field

import numpy as np
from easyrobot.arm.base import ArmBase
from easyrobot.hand.base import HandBase
from easyrobot.camera.base import RGBDCameraBase
from easyrobot.gripper.base import GripperBase
from easyrobot.utils.shared_memory import SharedMemoryManager, DictSharedMemoryManager

from runner.configs.agent.base import BaseAgentConfig
from runner.configs.lowdim_provider import LowdimRecorderConfig, LowdimObservationConfig
from runner.configs.filter import PoseOneEuroFilterConfig, OneEuroFilterConfig



@dataclass(kw_only = True)
class RealAgentConfig(BaseAgentConfig):
    robots: Dict[str, ArmBase] = field(default_factory = dict)
    grippers: Dict[str, GripperBase] = field(default_factory = dict)
    hands: Dict[str, HandBase] = field(default_factory = dict)
    cameras: Dict[str, Union[RGBDCameraBase, DictSharedMemoryManager]] = field(default_factory = dict)

    lowdim_recorder_configs: Dict[str, LowdimRecorderConfig] = field(default_factory = dict)
    lowdim_observation_configs: Dict[str, LowdimObservationConfig] = field(default_factory = dict)

    colors: List[str] = field(default_factory = list)
    depths: List[str] = field(default_factory = list)
    
    camera_shm_names: Dict[str, str] = field(default_factory = dict)  # camera_name -> shm_name mapping
    robot_shm_names: Dict[str, str] = field(default_factory = dict)  # robot_name -> shm_name mapping

    force_control_robot_names: List[str] = field(default_factory = list)
    torque_control_robot_names: List[str] = field(default_factory = list)

    pose_action_filter_configs: Dict[str, PoseOneEuroFilterConfig] = field(default_factory = dict)
    vector_action_filter_configs: Dict[str, OneEuroFilterConfig] = field(default_factory = dict)
