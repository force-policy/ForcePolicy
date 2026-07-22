"""
Transformation Matrix Generator.
"""
from typing import Union, Optional

import torch
import numpy as np

from utils.transforms.generic import _prepare_input, _to_output


def rot_mat_x(
    angle: Union[float, np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    3x3 rotation matrix around X-axis.
    Args:
        angle: Scalar or (N,) tensor/array of angles in radians.
    Returns:
        (3, 3) or (N, 3, 3) matrix.
    """
    a, is_numpy = _prepare_input(angle)
    c, s = torch.cos(a), torch.sin(a)
    mat = torch.zeros(a.shape + (3, 3), dtype = a.dtype, device = a.device)
    mat[..., 0, 0] = 1.0
    mat[..., 1, 1] = c
    mat[..., 1, 2] = -s
    mat[..., 2, 1] = s
    mat[..., 2, 2] = c
    return _to_output(mat, is_numpy)


def rot_mat_y(
    angle: Union[float, np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    3x3 rotation matrix around Y-axis.
    """
    a, is_numpy = _prepare_input(angle)
    c, s = torch.cos(a), torch.sin(a)
    mat = torch.zeros(a.shape + (3, 3), dtype = a.dtype, device = a.device)
    mat[..., 0, 0] = c
    mat[..., 0, 2] = s
    mat[..., 1, 1] = 1.0
    mat[..., 2, 0] = -s
    mat[..., 2, 2] = c
    return _to_output(mat, is_numpy)


def rot_mat_z(
    angle: Union[float, np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    3x3 rotation matrix around Z-axis.
    """
    a, is_numpy = _prepare_input(angle)
    c, s = torch.cos(a), torch.sin(a)
    mat = torch.zeros(a.shape + (3, 3), dtype = a.dtype, device = a.device)
    mat[..., 0, 0] = c
    mat[..., 0, 1] = -s
    mat[..., 1, 0] = s
    mat[..., 1, 1] = c
    mat[..., 2, 2] = 1.0
    return _to_output(mat, is_numpy)


def rot_mat(
    angles: Union[np.ndarray, torch.Tensor], 
    convention: Optional[str] = "XYZ"
) -> Union[np.ndarray, torch.Tensor]:
    """
    Composite 3x3 rotation matrix from euler angles.
    Supports arbitrary axes and Extrinsic/Intrinsic conventions.

    Args:
        angles: (..., 3) Tensor/Array. 
                The values correspond to the order in 'convention'.
                e.g., if convention="XZY", angles=[angle_x, angle_z, angle_y].
        convention: String of 3 characters (e.g., "XYZ", "zyx").
                    - UPPER CASE ("XYZ"): Extrinsic (Fixed Frame). 
                      Result = R_z @ R_y @ R_x
                    - lower case ("xyz"): Intrinsic (Moving Frame).
                      Result = R_x @ R_y @ R_z
    """
    a, is_numpy = _prepare_input(angles)
    
    if a.shape[-1] != 3:
        raise ValueError(f"Expected last dim to be 3, got {a.shape[-1]}")
    if len(convention) != 3:
        raise ValueError("Convention must be a 3-character string (e.g., 'XYZ')")

    axis_map = {
        'x': rot_mat_x, 'X': rot_mat_x,
        'y': rot_mat_y, 'Y': rot_mat_y,
        'z': rot_mat_z, 'Z': rot_mat_z
    }

    mats = []
    for i, char in enumerate(convention):
        if char not in axis_map:
            raise ValueError(f"Invalid axis character: {char}")
        angle = a[..., i]
        mats.append(axis_map[char](angle))

    if convention.isupper():
        mat = mats[2] @ mats[1] @ mats[0]
    else:
        mat = mats[0] @ mats[1] @ mats[2]
    
    return _to_output(mat, is_numpy)


def trans_mat(
    offsets: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    4x4 translation matrix.
    Args:
        offsets: [x, y, z] list, scalar array, or batch tensor.
    """
    t, is_numpy = _prepare_input(offsets)
    if t.shape[-1] != 3:
        raise ValueError("Offsets must have last dimension 3")
    eye = torch.eye(4, dtype = t.dtype, device = t.device)
    if len(t.shape) > 1:
         res = eye.expand(t.shape[:-1] + (4, 4)).clone()
    else:
         res = eye.clone()
    res[..., :3, 3] = t
    return _to_output(res, is_numpy)


def rot_trans_mat(
    offsets: Union[np.ndarray, torch.Tensor], 
    angles: Union[np.ndarray, torch.Tensor], 
    convention: Optional[str] = "XYZ"
) -> Union[np.ndarray, torch.Tensor]:
    """
    4x4 transformation: Rotate then Translate.
    """
    t, is_numpy_t = _prepare_input(offsets)
    a, is_numpy_a = _prepare_input(angles)
    
    if t.device != a.device:
        if t.device.type == 'cpu' and a.device.type != 'cpu':
            t = t.to(a.device)
        elif a.device.type == 'cpu' and t.device.type != 'cpu':
            a = a.to(t.device)
    
    R = rot_mat(a, convention = convention) 
    
    batch_shape = R.shape[:-2]
    res = torch.zeros(batch_shape + (4, 4), dtype = t.dtype, device = t.device)
    
    res[..., :3, :3] = R
    res[..., :3, 3] = t
    res[..., 3, 3] = 1.0
    
    return _to_output(res, is_numpy_t or is_numpy_a)


def trans_rot_mat(
    offsets: Union[np.ndarray, torch.Tensor], 
    angles: Union[np.ndarray, torch.Tensor], 
    convention: Optional[str] = "XYZ"
) -> Union[np.ndarray, torch.Tensor]:
    """
    4x4 transformation: Translate then Rotate.
    """
    t, is_numpy_t = _prepare_input(offsets)
    a, is_numpy_a = _prepare_input(angles)
    
    if t.device != a.device:
        if t.device.type == 'cpu' and a.device.type != 'cpu':
            t = t.to(a.device)
        elif a.device.type == 'cpu' and t.device.type != 'cpu':
            a = a.to(t.device)

    R = rot_mat(a, convention = convention)
    
    t_rotated = (R @ t.unsqueeze(-1)).squeeze(-1)
    
    batch_shape = R.shape[:-2]
    res = torch.zeros(batch_shape + (4, 4), dtype = t.dtype, device = t.device)
    
    res[..., :3, :3] = R
    res[..., :3, 3] = t_rotated
    res[..., 3, 3] = 1.0
    
    return _to_output(res, is_numpy_t or is_numpy_a)
