"""
Rotation representation transformations based on pytorch.

References:
- rotation 9d: Levinson et al, An Analysis of SVD for Deep Rotation Estimation, NeurIPS 2020.
               https://github.com/amakadia/svd_for_pose
- rotation 10d: Peretroukhin et al, A Smooth Representation of SO(3) for Deep Rotation Learning with Uncertainty, RSS 2020.
                https://github.com/utiasSTARS/bingham-rotation-learning
"""
from typing import Union, Optional

import sys
import torch
import numpy as np
import torch.nn.functional as F

from enum import Enum

from utils.transforms.generic import _prepare_input, _to_output


from pytorch3d.transforms.rotation_conversions import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_euler_angles,
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix
)

class RotationType(Enum):
    AXIS_ANGLE = "axis_angle"
    EULER_ANGLES = "euler_angles"
    QUATERNION = "quaternion"
    MATRIX = "matrix"
    ROTATION_6D = "rotation_6d"
    ROTATION_9D = "rotation_9d"
    ROTATION_10D = "rotation_10d"
    
    @staticmethod
    def from_str(s: str):
        return RotationType(s)

    def dim(self):
        if self == RotationType.AXIS_ANGLE:
            return 3
        elif self == RotationType.EULER_ANGLES:
            return 3
        elif self == RotationType.QUATERNION:
            return 4
        elif self == RotationType.MATRIX:
            return 9
        elif self == RotationType.ROTATION_6D:
            return 6
        elif self == RotationType.ROTATION_9D:
            return 9
        elif self == RotationType.ROTATION_10D:
            return 10
        else:
            raise ValueError(f"Unknown rotation type: {self}")


def rotation_9d_to_matrix(rotation_9d: torch.Tensor) -> torch.Tensor:
    """
    Map 9D input vectors onto SO(3) rotation matrix.
    """
    batch_dim = rotation_9d.size()[:-1]
    m = rotation_9d.view(batch_dim + (3, 3))
    u, s, vt = torch.linalg.svd(m, full_matrices = False)
    det = torch.det(u @ vt)
    det = det.view(batch_dim + (1, 1))
    vt = torch.cat((vt[..., :2, :], vt[..., -1:, :] * det), dim = -2)
    r = u @ vt
    return r


def matrix_to_rotation_9d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Map rotation matrix to 9D rotation representation. The mapping is not unique.

    Note that the rotation matrix itself is a valid 9D rotation representation.
    """
    return matrix


def rotation_10d_to_matrix(rotation_10d: torch.Tensor) -> torch.Tensor:
    """
    Map 10D input vectors to SO(3) rotation matrix.
    """
    batch_dim = rotation_10d.size()[:-1]
    idx = torch.triu_indices(4, 4)
    A = rotation_10d.new_zeros(batch_dim + (4, 4))
    A[..., idx[0], idx[1]] = rotation_10d
    A[..., idx[1], idx[0]] = rotation_10d
    _, evs = torch.linalg.eigh(A, UPLO = 'U')
    quat = evs[..., 0]
    matrix = quaternion_to_matrix(quat)
    return matrix


def matrix_to_rotation_10d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Map rotation matrix to 10D rotation representation. The mapping is not unique.
    
    See: https://github.com/utiasSTARS/bingham-rotation-learning/issues/8
    """
    batch_dim = matrix.size()[:-2]
    quat = matrix_to_quaternion(matrix)
    A = torch.eye(4).repeat(batch_dim + (1, 1)).type(quat.dtype).to(quat.device) - quat.unsqueeze(-1) @ quat.unsqueeze(-2)
    idx = torch.triu_indices(4, 4)
    rotation_10d = A[..., idx[0], idx[1]]
    return rotation_10d


def rotation_transform(
    rot: Union[np.ndarray, torch.Tensor],
    from_rep: RotationType, 
    to_rep: RotationType, 
    from_convention: Optional[str] = None, 
    to_convention: Optional[str] = None
):
    """
    Transform a rotation representation into another equivalent rotation representation.
    Supports both numpy.ndarray and torch.Tensor inputs.
    """
    if from_rep == RotationType.EULER_ANGLES and from_convention is None:
        raise ValueError("from_convention is required for euler_angles")
    if to_rep == RotationType.EULER_ANGLES and to_convention is None:
        raise ValueError("to_convention is required for euler_angles")

    if from_rep == to_rep and from_convention == to_convention:
        return rot

    x, is_numpy = _prepare_input(rot)
    current_module = sys.modules[__name__]

    if from_rep != RotationType.MATRIX:
        converter_name = f"{from_rep.value}_to_matrix"
        to_mat_func = getattr(current_module, converter_name)
        kwargs = {"convention": from_convention} if from_convention else {}
        mat = to_mat_func(x, **kwargs)
    else:
        mat = x

    if to_rep != RotationType.MATRIX:
        converter_name = f"matrix_to_{to_rep.value}"
        to_target_func = getattr(current_module, converter_name)
        kwargs = {"convention": to_convention} if to_convention else {}
        ret = to_target_func(mat, **kwargs)
    else:
        ret = mat

    if to_rep == RotationType.QUATERNION:
        ret = F.normalize(ret, p = 2, dim = -1)

    return _to_output(ret, is_numpy)


def quat_angle(
    q1: Union[np.ndarray, torch.Tensor], 
    q2: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate the geodesic distance (angle in radians) between two quaternions.
    Supports broadcasting and zero-copy inputs.
    
    Formula: theta = 2 * arccos(| <q1, q2> |)
    """
    q1_t, is_numpy_1 = _prepare_input(q1)
    q2_t, is_numpy_2 = _prepare_input(q2)

    q1_t = F.normalize(q1_t, p = 2, dim = -1)
    q2_t = F.normalize(q2_t, p = 2, dim = -1)

    dot = (q1_t * q2_t).sum(dim=-1)
    dot = torch.clamp(torch.abs(dot), -1.0, 1.0)
    angle = 2.0 * torch.acos(dot)

    return _to_output(angle, is_numpy_1 or is_numpy_2)


def rotation_angle(
    rot1: Union[np.ndarray, torch.Tensor], 
    rot2: Union[np.ndarray, torch.Tensor], 
    rotation_rep1: RotationType, 
    rotation_rep2: RotationType, 
    convention1: Optional[str] = None, 
    convention2: Optional[str] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate angle between ANY two rotation representations.
    
    Strategy: Convert both to Quaternion first (most efficient for angle calc),
    then compute quat_angle.
    """
    q1 = rotation_transform(
        rot1, 
        from_rep = rotation_rep1, 
        to_rep = RotationType.QUATERNION, 
        from_convention = convention1
    )
    q2 = rotation_transform(
        rot2, 
        from_rep = rotation_rep2, 
        to_rep = RotationType.QUATERNION, 
        from_convention = convention2
    )
    return quat_angle(q1, q2)

