"""
Generic transformation utilities.
"""
from typing import Union, Tuple

import torch
import numpy as np


def _prepare_input(
    x: Union[np.ndarray, torch.Tensor]
) -> Tuple[torch.Tensor, bool]:
    """
    Performance-aware Input Standardization.
    
    Returns:
        tensor_x: The input converted to a Torch Tensor (Zero-Copy if possible).
        is_numpy: Boolean flag indicating if original input was Numpy.
    """
    if isinstance(x, torch.Tensor):
        return x, False
    tensor_x = torch.as_tensor(x)
    if not tensor_x.is_floating_point():
        tensor_x = tensor_x.float()
    return tensor_x, True


def _to_output(
    tensor_x: torch.Tensor, 
    is_numpy: bool
) -> Union[np.ndarray, torch.Tensor]:
    """
    Restore output to original type.
    """
    return tensor_x.detach().cpu().numpy() if is_numpy else tensor_x


def _smart_clip(
    val: Union[float, np.ndarray, torch.Tensor], 
    min_val: float, 
    max_val: float
):
    """
    Clip value according to array type.
    """
    if isinstance(val, torch.Tensor):
        return torch.clamp(val, min_val, max_val)
    elif isinstance(val, np.ndarray):
        return np.clip(val, min_val, max_val)
    else:
        return max(min_val, min(val, max_val))


def skew(
    p: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert 3D vectors to 3x3 skew-symmetric matrices.
    Returns tensor consistent with input type (Numpy or Tensor).
    
    Args:
        p: (..., 3) vector
    Returns:
        (..., 3, 3) skew matrix
    """
    x, is_numpy = _prepare_input(p)
    batch_shape = x.shape[:-1]
    res = torch.zeros(batch_shape + (3, 3), dtype = x.dtype, device = x.device)
    res[..., 0, 1] = -x[..., 2]
    res[..., 0, 2] =  x[..., 1]
    res[..., 1, 0] =  x[..., 2]
    res[..., 1, 2] = -x[..., 0]
    res[..., 2, 0] = -x[..., 1]
    res[..., 2, 1] =  x[..., 0]
    return _to_output(x, is_numpy)


def _broadcast_mat_to_input(
    mat: torch.Tensor, 
    input_tensor: torch.Tensor, 
    is_pose_input: bool = False
) -> torch.Tensor:
    """
    Reshapes transformation matrix to broadcast against arbitrary middle dimensions in input.
    """
    input_geom_rank = 2 if is_pose_input else 1
    input_batch_rank = input_tensor.ndim - input_geom_rank
    mat_batch_rank = mat.ndim - 2
    gap = input_batch_rank - mat_batch_rank

    if gap > 0:
        new_shape = mat.shape[:mat_batch_rank] + (1, ) * gap + mat.shape[mat_batch_rank:]
        return mat.view(new_shape)
    else:
        return mat
