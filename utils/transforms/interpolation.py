"""
Interpolation utilities.
"""
from typing import Union, Optional, Literal

import torch
import numpy as np
import torch.nn.functional as F

from utils.transforms.rotation import RotationType
from utils.transforms.pose import xyz_rot_transform
from utils.transforms.generic import _prepare_input, _to_output, _smart_clip


def quat_slerp(
    q1: Union[np.ndarray, torch.Tensor], 
    q2: Union[np.ndarray, torch.Tensor], 
    alpha: Union[float, int, list, np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Spherical linear interpolation between quaternions with PyTorch backend.
    Supports Numpy/Tensor inputs and automatic broadcasting.
    
    Args:
        q1: (..., 4) First quaternion
        q2: (..., 4) Second quaternion
        alpha: Interpolation parameter. Scalar or Tensor/Array broadcastable to result.
               e.g., if q1 is (4,), alpha can be (N,) -> Result (N, 4)
    """
    q1_t, is_numpy_1 = _prepare_input(q1)
    q2_t, is_numpy_2 = _prepare_input(q2)
    
    if isinstance(alpha, (float, int, list)):
        alpha_t = torch.tensor(alpha, device = q1_t.device, dtype = q1_t.dtype)
    else:
        alpha_t = torch.as_tensor(alpha, device = q1_t.device, dtype = q1_t.dtype)

    q1_t = F.normalize(q1_t, p = 2, dim = -1)
    q2_t = F.normalize(q2_t, p = 2, dim = -1)

    if alpha_t.ndim > 0 and alpha_t.shape[-1] != 1:
        alpha_t = alpha_t.unsqueeze(-1)

    dot = (q1_t * q2_t).sum(dim = -1, keepdim = True)
    q2_t = torch.where(dot < 0, -q2_t, q2_t)
    theta = torch.acos(torch.clamp(torch.abs(dot), -1.0, 1.0))
    sin_theta = torch.sin(theta)

    safe_sin_theta = torch.where(sin_theta < 1e-6, torch.ones_like(sin_theta), sin_theta)

    ratio_a = torch.sin((1.0 - alpha_t) * theta) / safe_sin_theta
    ratio_b = torch.sin(alpha_t * theta) / safe_sin_theta
    res_slerp = ratio_a * q1_t + ratio_b * q2_t
    res_lerp = F.normalize((1.0 - alpha_t) * q1_t + alpha_t * q2_t, dim = -1)
    
    res = torch.where(sin_theta < 1e-6, res_lerp, res_slerp)

    return _to_output(res, is_numpy_1 or is_numpy_2)


def interpolate_pose(
    p1: Union[np.ndarray, torch.Tensor], 
    p2: Union[np.ndarray, torch.Tensor], 
    alpha: Union[float, int, list, np.ndarray, torch.Tensor], 
    rotation_rep: RotationType, 
    convention: Optional[str] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Interpolate between two poses of ANY representation.
    Strategy: Convert to (XYZ + Quaternion), interpolate, then convert back.
    
    Args:
        p0, p1: Start and End poses (Matrix, Euler, Axis-Angle, etc.)
        alpha: Interpolation ratio [0, 1]. Supports broadcasting.
        rep: The rotation representation of p0 and p1.
    """
    pose1_7d = xyz_rot_transform(
        p1, 
        from_rep = rotation_rep, 
        to_rep = RotationType.QUATERNION, 
        from_convention = convention
    )
    pose2_7d = xyz_rot_transform(
        p2, 
        from_rep = rotation_rep, 
        to_rep = RotationType.QUATERNION, 
        from_convention = convention
    )

    p1_t, is_numpy_1 = _prepare_input(pose1_7d)
    p2_t, is_numpy_2 = _prepare_input(pose2_7d)
    
    if isinstance(alpha, (float, int, list)):
        alpha_t = torch.tensor(alpha, device = p1_t.device, dtype = p1_t.dtype)
    else:
        alpha_t = torch.as_tensor(alpha, device = p1_t.device, dtype = p1_t.dtype)
    
    if alpha_t.ndim > 0 and alpha_t.shape[-1] != 1:
        alpha_t = alpha_t.unsqueeze(-1)

    xyz1, q1 = p1_t[..., :3], p1_t[..., 3:]
    xyz2, q2 = p2_t[..., :3], p2_t[..., 3:]
    xyz_interp = (1 - alpha_t) * xyz1 + alpha_t * xyz2
    q_interp = quat_slerp(q1, q2, alpha_t)
    pose_interp_7d = torch.cat([xyz_interp, q_interp], dim = -1)

    res = xyz_rot_transform(
        pose_interp_7d, 
        from_rep = RotationType.QUATERNION,
        to_rep = rotation_rep, 
        to_convention = convention
    )
    return _to_output(res, is_numpy_1 or is_numpy_2)


def interpolate_value(
    v1: Union[np.ndarray, torch.Tensor],
    t1: Union[float, np.ndarray, torch.Tensor],
    v2: Union[np.ndarray, torch.Tensor],
    t2: Union[float, np.ndarray, torch.Tensor],
    t: Union[float, np.ndarray, torch.Tensor],
    interp_type: Literal["linear", "nearest", "pose_linear", "before", "after"],
    rotation_rep: Optional[RotationType] = None,
    convention: Optional[str] = None
):
    """
    General interpolation between x0 and x1.
    Supports scalars, batches, Numpy arrays, and PyTorch tensors.
    
    Args:
        t0, t1: Time/Index of start and end points.
        x0, x1: Data at t0 and t1.
        t: Target time/index.
    """
    if interp_type == "before":
        return v1
    if interp_type == "after":
        return v2
    
    dt = t2 - t1
    if torch.is_tensor(dt) or isinstance(dt, np.ndarray):
        alpha = (t - t1) / (dt + 1e-9) 
    else:
        alpha = 0.0 if abs(dt) < 1e-9 else (t - t1) / dt
    
    alpha = _smart_clip(alpha, 0.0, 1.0) 
    
    if interp_type == "nearest":
        if torch.is_tensor(alpha):
            mask = (alpha >= 0.5).type(v1.dtype)
            return mask * v2 + (1 - mask) * v1
        else:
            return v2 if alpha >= 0.5 else v1

    elif interp_type == "linear":
        return (1 - alpha) * v1 + alpha * v2

    elif interp_type == "pose_linear":
        return interpolate_pose(
            v1, v2, alpha,
            rotation_rep = rotation_rep,
            convention = convention
        )
    
    else:
        raise ValueError(f"Unsupported interp_type: {interp_type}")


def resample_trajectory(
    data: np.ndarray, 
    source_freq: int, 
    source_length: int,
    target_freq: int, 
    target_length: int,
    sampling_method: Literal["linear", "nearest", "pose_linear", "before", "after"] = "linear", 
    rotation_rep: Optional[RotationType] = None,
    convention: Optional[str] = None
) -> np.ndarray:
    """
    Resample trajectory from source frequency to target frequency and pad to target length.
    
    Args:
        data: [source_length, D] input trajectory
        source_freq: source frequency Hz
        source_length: source length
        target_freq: target frequency Hz
        target_length: final padded length
        sampling_method: 'linear', 'nearest', or 'pose_linear'
        rotation_rep: RotationType for pose_linear (default None implies fallback or specific handling)
        convention: rotation convention for pose_linear
        
    Returns:
        [target_length, D] resampled and padded/truncated trajectory
    """
    assert data.shape[0] == source_length

    if source_freq == target_freq:
        partial_resampled = data
    
    else:
        T_resampled = int(np.ceil(source_length / source_freq * target_freq))
        if T_resampled <= 1:
            resampled = np.repeat(data[-1:], max(1, target_length), axis = 0)
            return resampled

        # Time indices
        t_source = np.arange(source_length)
        t_target = np.linspace(0, source_length - 1, T_resampled)
        
        # Prepare inputs for interpolation
        idx_floor = np.floor(t_target).astype(int)
        idx_ceil = np.clip(idx_floor + 1, 0, source_length - 1)
        
        d1 = data[idx_floor]
        d2 = data[idx_ceil]
        
        partial_resampled = interpolate_value(
            d1, idx_floor, d2, idx_ceil, t_target, 
            interp_type = sampling_method,
            rotation_rep = rotation_rep,
            convention = convention
        )
        
    if partial_resampled.shape[0] < target_length:
        pad_len = target_length - partial_resampled.shape[0]
        padding = np.repeat(partial_resampled[-1:], pad_len, axis = 0)
        resampled = np.concatenate([partial_resampled, padding], axis = 0)
    elif partial_resampled.shape[0] > target_length:
        resampled = partial_resampled[:target_length]
    else:
        resampled = partial_resampled
        
    return resampled
