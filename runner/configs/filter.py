from typing import Optional, List, Union
from dataclasses import dataclass, field

import numpy as np

from utils.transforms.rotation import RotationType


@dataclass(kw_only = False)
class OneEuroFilterConfig:
    mincutoff: Union[float, np.ndarray, List[float]] = 1.0
    beta: Union[float, np.ndarray, List[float]] = 0.0
    dcutoff: Union[float, np.ndarray, List[float]] = 1.0


@dataclass(kw_only = False)
class RotationOneEuroFilterConfig:
    mincutoff: Union[float, np.ndarray, List[float]] = 1.0
    beta: Union[float, np.ndarray, List[float]] = 0.0
    dcutoff: Union[float, np.ndarray, List[float]] = 1.0
    rotation_rep: RotationType = RotationType.QUATERNION
    convention: Optional[str] = None


@dataclass(kw_only = False)
class PoseOneEuroFilterConfig:
    trans: OneEuroFilterConfig = field(default_factory = OneEuroFilterConfig)
    rot: RotationOneEuroFilterConfig = field(default_factory = RotationOneEuroFilterConfig)
