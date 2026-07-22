from typing import Optional

import numpy as np

from utils.transforms.pose import xyz_rot_to_mat
from utils.transforms.rotation import RotationType, rotation_transform


def calc_twist(
    pose: np.ndarray,
    rotation_rep: RotationType,
    convention: Optional[str] = None,
    freq: float = 1000.0
):
    """
    Args:
        poses: (N, ...) np.array, pose sequence;
        rotation_rep: RotationType;
        convention: rotation convention;
        freq: float, frequency.
        
    Returns:
        twist: (N, 6) twist sequence.
    """
    dt = 1.0 / freq
    pose = xyz_rot_to_mat(pose, rotation_rep = rotation_rep, convention = convention)

    t = pose[:, :3, 3]
    R = pose[:, :3, :3]

    v = np.gradient(t, dt, axis = 0)
    dR_mat = R[1:] @ np.linalg.inv(R[:-1])
    dR = rotation_transform(
        dR_mat,
        from_rep = RotationType.MATRIX,
        to_rep = RotationType.AXIS_ANGLE
    )
    w = dR / dt
    w = np.vstack([w, w[-1]])
    twist = np.concatenate([v, w], axis = -1)

    return twist
