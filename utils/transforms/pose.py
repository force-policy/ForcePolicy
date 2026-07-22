"""
Pose representation transformations based on pytorch.
"""
from typing import Union, Optional

import torch
import numpy as np

from utils.transforms.generic import skew, _prepare_input, _to_output
from utils.transforms.rotation import RotationType, rotation_transform


def xyz_rot_transform(
    xyz_rot: Union[np.ndarray, torch.Tensor],
    from_rep: RotationType, 
    to_rep: RotationType, 
    from_convention: Optional[str] = None, 
    to_convention: Optional[str] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform a 6D pose (XYZ + Rotation) into another equivalent representation.
    Supports both numpy.ndarray and torch.Tensor inputs.
    
    Args:
        xyz_rot: Input pose. 
                 Shape (..., 3 + rot_dim) or (..., 4, 4) for matrix.
    """
    if from_rep == to_rep and from_convention == to_convention:
        return xyz_rot
    
    x, is_numpy = _prepare_input(xyz_rot)

    if from_rep == RotationType.MATRIX:
        assert x.shape[-1] == 4 and x.shape[-2] == 4, "Input must be (..., 4, 4) for matrix rep"
        xyz = x[..., :3, 3]      # (..., 3)
        rot = x[..., :3, :3]     # (..., 3, 3)
    else:
        expected_dim = 3 + from_rep.dim()
        assert x.shape[-1] == expected_dim, f"Expected last dim {expected_dim} for {from_rep}"
        xyz = x[..., :3]         # (..., 3)
        rot = x[..., 3:]         # (..., D)

    new_rot = rotation_transform(
        rot,
        from_rep = from_rep,
        to_rep = to_rep,
        from_convention = from_convention,
        to_convention = to_convention
    )
    
    if to_rep != RotationType.MATRIX:
        ret = torch.cat([xyz, new_rot], dim=-1)
    else:
        batch_shape = xyz.shape[:-1]
        ret = torch.zeros(batch_shape + (4, 4), dtype = x.dtype, device = x.device)
        
        ret[..., :3, :3] = new_rot
        ret[..., :3, 3] = xyz
        ret[..., 3, 3] = 1.0 
    
    return _to_output(ret, is_numpy)


def xyz_rot_to_mat(
    xyz_rot: Union[np.ndarray, torch.Tensor], 
    rotation_rep: RotationType, 
    convention: Optional[str] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform an xyz_rot representation under any rotation form to an unified 4x4 pose representation.
    """
    return xyz_rot_transform(
        xyz_rot,
        from_rep = rotation_rep,
        to_rep = RotationType.MATRIX,
        from_convention = convention
    )


def mat_to_xyz_rot(
    mat: Union[np.ndarray, torch.Tensor], 
    rotation_rep: RotationType, 
    convention: Optional[str] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform an unified 4x4 pose representation to an xyz_rot representation under any rotation form.
    """
    return xyz_rot_transform(
        mat,
        from_rep = RotationType.MATRIX,
        to_rep = rotation_rep,
        to_convention = convention
    )


def pose_to_adjoint(
    xyz_rot: Union[np.ndarray, torch.Tensor], 
    rotation_rep: RotationType = RotationType.MATRIX, 
    convention = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert a pose to its 6x6 Adjoint matrix representation.
    Adjoint(T) = [[R, [p] x R], [0, R]]
    """
    x, is_numpy = _prepare_input(xyz_rot)

    mat_4x4 = xyz_rot_to_mat(
        x,
        rotation_rep = rotation_rep,
        convention = convention
    )

    R = mat_4x4[..., :3, :3] # (..., 3, 3)
    p = mat_4x4[..., :3, 3]  # (..., 3)
    batch_shape = mat_4x4.shape[:-2]
    adj = torch.zeros(batch_shape + (6, 6), dtype = x.dtype, device = x.device)
    skew_p = skew(p)
    adj[..., :3, :3] = R
    adj[..., 3:, 3:] = R
    adj[..., :3, 3:] = skew_p @ R 

    return _to_output(adj, is_numpy)


def transform_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Transformation multiplication.
    """
    A, is_numpy_A = _prepare_input(A)
    B, is_numpy_B = _prepare_input(B)

    diff = A.dim() - B.dim()

    if diff > 0:
        new_shape = B.shape[:-2] + (1,) * diff + (4, 4)
        B = B.view(new_shape)
    elif diff < 0:
        new_shape = A.shape[:-2] + (1,) * abs(diff) + (4, 4)
        A = A.view(new_shape)

    return _to_output(A @ B, is_numpy_A or is_numpy_B)
