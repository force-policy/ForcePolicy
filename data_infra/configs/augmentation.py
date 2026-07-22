"""
Augmentation configurations.
"""

from typing import List, Tuple, Literal, Optional, Union
from enum import Enum
from dataclasses import dataclass, field

import numpy as np


class AugmentationType(Enum):
    RANDOM_TRANSFORM = "random_transform"
    COLOR_JITTER = "color_jitter"


@dataclass(kw_only = True)
class BaseAugmentationConfig:
    """ Base Augmentation Config """
    type: AugmentationType


@dataclass(kw_only = True)
class RandomTransformAugmentationConfig(BaseAugmentationConfig):
    """ 
    Random Rigid Transform Config (Rotation + Translation).
    Rotation is applied as R = R_z * R_y * R_x
    """
    type: AugmentationType = field(default = AugmentationType.RANDOM_TRANSFORM, init = False)

    # Translation range in meters [min, max] for each axis
    trans_x_range: Tuple[float, float] = (0.0, 0.0)
    trans_y_range: Tuple[float, float] = (0.0, 0.0)
    trans_z_range: Tuple[float, float] = (0.0, 0.0)
    
    # Rotation range in degrees [min, max] for each axis
    rot_x_range: Tuple[float, float] = (0.0, 0.0)
    rot_y_range: Tuple[float, float] = (0.0, 0.0)
    rot_z_range: Tuple[float, float] = (0.0, 0.0)


@dataclass(kw_only = True)
class ColorJitterAugmentationConfig(BaseAugmentationConfig):
    """ Color Jitter Config (Gaussian Noise) """
    type: AugmentationType = field(default = AugmentationType.COLOR_JITTER, init = False)
    prob: float
    brightness: float
    contrast: float
    saturation: float
    hue: float
