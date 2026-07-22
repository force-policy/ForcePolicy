"""
Normalization operations.
"""

from typing import List, Union

import torch
import numpy as np

from data_infra.configs import (
    BaseNormalizationConfig,
    EmptyNormalizationConfig,
    LinearNormalizationConfig,
    GaussianNormalizationConfig
)


def _cast_to_reference(
    reference: Union[np.ndarray, torch.Tensor],
    param: Union[np.ndarray, torch.Tensor, List[float], float],
    channel_last: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Casts 'param' to exactly match the type, device, and dtype of 'reference'.
    
    Logic:
    1. If reference is Tensor -> param becomes Tensor (on same GPU/dtype).
    2. If reference is Numpy -> param becomes Numpy.
    """
    if isinstance(reference, torch.Tensor):
        if isinstance(param, torch.Tensor):
            param = param.to(device = reference.device, dtype = reference.dtype)
        else:
            param = torch.tensor(param, device = reference.device, dtype = reference.dtype)

    elif isinstance(reference, np.ndarray):
        if isinstance(param, torch.Tensor):
            param = param.detach().cpu().numpy().astype(reference.dtype)
        else:
            param = np.array(param, dtype = reference.dtype)
            
    else:
        param = np.array(param)

    if not channel_last:
        param = param.reshape(param.shape + (1, ) * (len(reference.shape) - 2))

    return param


def normalize(
    value: Union[np.ndarray, torch.Tensor, List[float]],
    norm_config: BaseNormalizationConfig,
    eps: float = 1e-8
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies normalization. 
    Strictly preserves input type (Tensor->Tensor, Numpy->Numpy).
    """
    if isinstance(value, (list, tuple)):
        value = np.array(value)

    if norm_config.type == "none":
        return value

    elif norm_config.type == "linear":
        if not isinstance(norm_config, LinearNormalizationConfig):
             raise ValueError("Config type mismatch: expected LinearNormalizationConfig")
        min_v = _cast_to_reference(value, norm_config.min_value, channel_last = norm_config.channel_last)
        max_v = _cast_to_reference(value, norm_config.max_value, channel_last = norm_config.channel_last)
        return (value - min_v) / (max_v - min_v + eps) * 2 - 1
        # return (value - min_v) / (max_v - min_v + eps)

    elif norm_config.type == "gaussian":
        if not isinstance(norm_config, GaussianNormalizationConfig):
             raise ValueError("Config type mismatch: expected GaussianNormalizationConfig")
        mean_v = _cast_to_reference(value, norm_config.mean_value, channel_last = norm_config.channel_last)
        std_v = _cast_to_reference(value, norm_config.std_value, channel_last = norm_config.channel_last)
        return (value - mean_v) / (std_v + eps)
    
    else:
        raise ValueError(f"Unsupported normalization type: {norm_config.type}")


def unnormalize(
    value: Union[np.ndarray, torch.Tensor, List[float]],
    norm_config: BaseNormalizationConfig
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies inverse normalization.
    Strictly preserves input type (Tensor->Tensor, Numpy->Numpy).
    """
    if isinstance(value, (list, tuple)):
        value = np.array(value)

    if norm_config.type == "none":
        return value

    elif norm_config.type == "linear":
        if not isinstance(norm_config, LinearNormalizationConfig):
             raise ValueError("Config type mismatch: expected LinearNormalizationConfig")
        min_v = _cast_to_reference(value, norm_config.min_value, channel_last = norm_config.channel_last)
        max_v = _cast_to_reference(value, norm_config.max_value, channel_last = norm_config.channel_last)
        return (value + 1) / 2 * (max_v - min_v) + min_v
        # return value * (max_v - min_v) + min_v

    elif norm_config.type == "gaussian":
        if not isinstance(norm_config, GaussianNormalizationConfig):
             raise ValueError("Config type mismatch: expected GaussianNormalizationConfig")
        mean_v = _cast_to_reference(value, norm_config.mean_value, channel_last = norm_config.channel_last)
        std_v = _cast_to_reference(value, norm_config.std_value, channel_last = norm_config.channel_last)
        return value * std_v + mean_v

    else:
        raise ValueError(f"Unsupported normalization type: {norm_config.type}")