"""
Dataset Configs
"""
from typing import List, Optional, Tuple, Literal, Dict
from dataclasses import dataclass, field

import numpy as np

from data_infra.configs.base import *


# TODO: support vision horizon
@dataclass(kw_only = True)
class VisionConfig:
    main_cameras: List[str]             # camera_name(main): camera_id
    aux_cameras: Dict[str, str] = field(default_factory = dict)    
                                        # camera_name: camera_id

    # list of camera names, specifying which is needed
    # default key is: camera_name/[color/depth/extrinsic/intrinsic]
    colors: List[str] = field(default_factory = list)
    depths: List[str] = field(default_factory = list)
    extrinsics: List[str] = field(default_factory = list)
    intrinsics: List[str] = field(default_factory = list)


@dataclass(kw_only = True)
class LowdimConfig:
    lowdim_name: str
    field: str
    length: int
    freq: int
    data_freq: int
    direction: Literal[1, -1]
    remove_first: bool


@dataclass(kw_only = True)
class ObservationConfig:
    vision: Optional[VisionConfig] = None
    lowdim: Dict[str, LowdimConfig] = field(default_factory = dict)


@dataclass(kw_only = True)
class ActionConfig:
    lowdim: Dict[str, LowdimConfig] = field(default_factory = dict)
    

@dataclass(kw_only = True)
class DatasetConfig:
    type: Literal["lowdim", "vision"] = "vision"
    data_path: str
    obs: ObservationConfig
    action: ActionConfig
    pivot_key: Optional[str] = None
    repeat_dataset: Optional[int] = None

    prefix_config: PrefixConfig = field(default_factory = PrefixConfig)
    robot_poses: Dict[str, np.ndarray] = field(default_factory = dict)