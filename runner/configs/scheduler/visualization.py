from typing import Any, Dict, List, Literal, Optional, Union
from dataclasses import dataclass, field

from data_infra.configs.point_mask import *


@dataclass(kw_only = True)
class VisualizationConfig:
    mode: Literal["2d", "3d"] = "3d"
    
    # Camera selection
    camera_names: List[str] = field(default_factory = list)
    
    # Observation keys
    obs_pose_keys: List[str] = field(default_factory = list)
    obs_ft_keys: List[str] = field(default_factory = list)
    action_pose_keys: List[str] = field(default_factory = list)
    action_ft_keys: List[str] = field(default_factory = list)

    # 3D specific masks
    world_mask_config: Optional[Union[PointMaskConfig, GroupPointMaskConfig, List[PointMaskConfig]]] = None
