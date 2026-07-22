"""
Normalization configurations.
"""
from typing import List, Tuple, Union, Literal, Optional

import torch
import numpy as np

from dataclasses import dataclass, field


@dataclass(kw_only = True)
class BaseNormalizationConfig:
    """ Base Normalization Config """
    type: Literal["none", "linear", "gaussian"]
    channel_last: bool = True


@dataclass(kw_only = True)
class EmptyNormalizationConfig(BaseNormalizationConfig):
    type: Literal["none", "linear", "gaussian"] = field(default = "none", init = False)


@dataclass(kw_only = True)
class LinearNormalizationConfig(BaseNormalizationConfig):
    """ Min-Max Normalization """
    type: Literal["none", "linear", "gaussian"] = field(default = "linear", init = False)
    min_value: Union[np.ndarray, torch.Tensor, List[float], float]
    max_value: Union[np.ndarray, torch.Tensor, List[float], float]


@dataclass(kw_only = True)
class GaussianNormalizationConfig(BaseNormalizationConfig):
    """ Z-Score Normalization """
    type: Literal["none", "linear", "gaussian"] = field(default = "gaussian", init = False)
    mean_value: Union[np.ndarray, torch.Tensor, List[float], float]
    std_value: Union[np.ndarray, torch.Tensor, List[float], float]
