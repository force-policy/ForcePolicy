"""
Projection based on pytorch.
"""
from typing import List, Union, Optional

import sys
import torch
import numpy as np
import utils.transforms.rotation as rot_utils

from utils.transforms.rotation import RotationType
from utils.transforms.pose import mat_to_xyz_rot, xyz_rot_to_mat
from utils.transforms.generic import _prepare_input, _to_output, _broadcast_mat_to_input


def apply_mat_to_pose(
    pose: Union[np.ndarray, torch.Tensor], 
    mat: Union[np.ndarray, torch.Tensor], 
    rotation_rep: RotationType, 
    convention: Optional[str] = None,
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply transformation matrices to poses.

    Args:
        pose: (..., 6/7/9/12/13) or (..., 4, 4) pose.
        mat: (..., 4, 4) transformation matrices.
        rotation_rep, convention: specify the pose representation.

    Returns:
        The transformed poses.
    """
    p, is_numpy = _prepare_input(pose)
    m = torch.as_tensor(mat, device = p.device, dtype = p.dtype)
    pose_mat = xyz_rot_to_mat(p, rotation_rep = rotation_rep, convention = convention)
    m = _broadcast_mat_to_input(m, pose_mat, is_pose_input = True)
    res_pose_mat = m @ pose_mat
    res_pose = mat_to_xyz_rot(res_pose_mat, rotation_rep = rotation_rep, convention = convention)
    return _to_output(res_pose, is_numpy)


def apply_mat_to_point(
    point: Union[np.ndarray, torch.Tensor],
    mat: Union[np.ndarray, torch.Tensor], 
    coord_first: Optional[bool] = False,
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply transformation matrices to points.

    Args:
        point: (..., N, 3) or (..., 3, N) points.
        mat: (..., 4, 4) transformation matrices.
        coord_first: If True, input points are (..., 3, N). Output will be (3, ...).

    Returns:
        The transformed points.
    """
    p, is_numpy = _prepare_input(point)
    m = torch.as_tensor(mat, device = p.device, dtype = p.dtype)

    if coord_first:
        p = p.transpose(-1, -2)

    m = _broadcast_mat_to_input(m, p, is_pose_input = False)
    R = m[..., :3, :3]
    t = m[..., :3, 3]
    res = (p.unsqueeze(-2) @ R.transpose(-1, -2)).squeeze(-2) + t

    if coord_first:
        res = res.transpose(-1, -2)

    return _to_output(res, is_numpy)


def apply_mat_to_wrench(
    wrench: Union[np.ndarray, torch.Tensor],
    mat: Union[np.ndarray, torch.Tensor],
    rotation_only: bool = False,
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform Wrench (Force, Torque) using transformation matrix.
    
    Args:
        wrench: (..., 6) [fx, fy, fz, tx, ty, tz]
        mat: (..., 4, 4) Transformation matrix
        rotation_only: If True, only rotates vectors. If False, adds lever-arm effect.
    """
    w, is_numpy = _prepare_input(wrench)
    m = torch.as_tensor(mat, device = w.device, dtype = w.dtype)
    
    if w.shape[-1] != 6:
        raise ValueError(f"Wrench must have last dimension 6, got {w.shape[-1]}")

    m = _broadcast_mat_to_input(m, w, is_pose_input = False)
    f = w[..., :3] 
    tau = w[..., 3:]
    R = m[..., :3, :3]
    f_new = (f.unsqueeze(-2) @ R.transpose(-1, -2)).squeeze(-2)
    tau_new = (tau.unsqueeze(-2) @ R.transpose(-1, -2)).squeeze(-2)

    if not rotation_only:
        t = m[..., :3, 3]
        tau_offset = torch.cross(t, f_new, dim = -1)
        tau_new = tau_new + tau_offset

    res = torch.cat([f_new, tau_new], dim = -1)
    return _to_output(res, is_numpy)


def apply_mat_to_twist(
    twist: Union[np.ndarray, torch.Tensor],
    mat: Union[np.ndarray, torch.Tensor],
    rotation_only: bool = False,
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform Twist (Linear Vel, Angular Vel) using transformation matrix.
    
    Args:
        twist: (..., 6) [vx, vy, vz, wx, wy, wz]
        mat: (..., 4, 4) Transformation matrix
        rotation_only: If True, ignores translation effect on linear velocity.
    """
    v, is_numpy = _prepare_input(twist)
    m = torch.as_tensor(mat, device = v.device, dtype = v.dtype)

    if v.shape[-1] != 6:
        raise ValueError(f"Twist must have last dimension 6, got {v.shape[-1]}")

    m = _broadcast_mat_to_input(m, v, is_pose_input = False)
    lin = v[..., :3] 
    ang = v[..., 3:] 
    R = m[..., :3, :3]
    ang_new = (ang.unsqueeze(-2) @ R.transpose(-1, -2)).squeeze(-2)
    lin_new = (lin.unsqueeze(-2) @ R.transpose(-1, -2)).squeeze(-2)

    if not rotation_only:
        t = m[..., :3, 3]
        lin_offset = torch.cross(t, ang_new, dim=-1)
        lin_new = lin_new + lin_offset

    res = torch.cat([lin_new, ang_new], dim=-1)
    return _to_output(res, is_numpy)


def apply_mat_to_force(
    force: Union[np.ndarray, torch.Tensor],
    mat: Union[np.ndarray, torch.Tensor],
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform Force using transformation matrix.

    Args:
        force: (..., 3) [fx, fy, fz]
        mat: (..., 4, 4) Transformation matrix
    """
    f, is_numpy = _prepare_input(force)
    m = torch.as_tensor(mat, device = f.device, dtype = f.dtype)
    
    m = _broadcast_mat_to_input(m, f, is_pose_input = False)
    R = m[..., :3, :3]
    f_new = (f.unsqueeze(-2) @ m[..., :3, :3].transpose(-1, -2)).squeeze(-2)
    
    return _to_output(f_new, is_numpy)


def apply_mat_to_torque(
    torque: Union[np.ndarray, torch.Tensor],
    mat: Union[np.ndarray, torch.Tensor],
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform Torque using transformation matrix (Rotation only).

    Args:
        torque: (..., 3) [tx, ty, tz]
        mat: (..., 4, 4) Transformation matrix
    """
    tau, is_numpy = _prepare_input(torque)
    m = torch.as_tensor(mat, device = tau.device, dtype = tau.dtype)
    
    m = _broadcast_mat_to_input(m, tau, is_pose_input = False)
    R = m[..., :3, :3]
    tau_new = (tau.unsqueeze(-2) @ m[..., :3, :3].transpose(-1, -2)).squeeze(-2)
    
    return _to_output(tau_new, is_numpy)


def apply_mat_to_linear_velocity(
    vel: Union[np.ndarray, torch.Tensor],
    mat: Union[np.ndarray, torch.Tensor],
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform Linear Velocity using transformation matrix (Rotation only).

    Args:
        vel: (..., 3) [vx, vy, vz]
        mat: (..., 4, 4) Transformation matrix
    """
    v, is_numpy = _prepare_input(vel)
    m = torch.as_tensor(mat, device = v.device, dtype = v.dtype)
    
    m = _broadcast_mat_to_input(m, v, is_pose_input = False)
    R = m[..., :3, :3]
    v_new = (v.unsqueeze(-2) @ m[..., :3, :3].transpose(-1, -2)).squeeze(-2)
    
    return _to_output(v_new, is_numpy)


def apply_mat_to_angular_velocity(
    vel: Union[np.ndarray, torch.Tensor],
    mat: Union[np.ndarray, torch.Tensor],
    **kwargs
) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform Angular Velocity using transformation matrix.

    Args:
        vel: (..., 3) [wx, wy, wz]
        mat: (..., 4, 4) Transformation matrix
    """
    w, is_numpy = _prepare_input(vel)
    m = torch.as_tensor(mat, device = w.device, dtype = w.dtype)
    
    m = _broadcast_mat_to_input(m, w, is_pose_input = False)
    R = m[..., :3, :3]
    w_new = (w.unsqueeze(-2) @ m[..., :3, :3].transpose(-1, -2)).squeeze(-2)
    
    return _to_output(w_new, is_numpy)
