from typing import Any, List, Optional, Literal

import numpy as np

from runner.utils.interp import interp_continuous_dynamics, interp_linear_dynamics


def build_trajectory(
    pred_tcp_poses: np.ndarray,
    cur_tcp_pose: np.ndarray,
    cur_tcp_vel: np.ndarray,
    source_freq: float,
    target_freq: float,
    interp_type: Literal["velocity_continuous", "acceleration_continuous", "linear"] = "linear"
):
    """
    Build a robot trajectory based on the prediction.
    """
    assert target_freq >= source_freq
    if interp_type in ["velocity_continuous", "acceleration_continuous"]:
        interp_poses = interp_continuous_dynamics(
            pred_tcp_poses = pred_tcp_poses,
            cur_tcp_pose = cur_tcp_pose,
            cur_tcp_vel = cur_tcp_vel,
            f_in = source_freq,
            f_out = target_freq,
            continuous_type = "vel" if interp_type == "velocity_continuous" else "acc"
        )
    elif interp_type == "linear":
        interp_poses = interp_linear_dynamics(
            pred_tcp_poses = pred_tcp_poses,
            cur_tcp_pose = cur_tcp_pose,
            f_in = source_freq,
            f_out = target_freq
        )
    else:
        raise AttributeError(f"Invalid interp_type: {interp_type}")

    return interp_poses
    

def build_indices(
    source_freq: float,
    target_freq: float,
    N_target: int,
    N_source: int,
) -> np.ndarray:
    """
    Build indices for other action after resampling.
    """
    int_indices = np.floor(np.arange(N_target) / target_freq * source_freq).astype(int)
    diff = np.diff(int_indices, prepend = int_indices[0])
    int_indices = np.clip(int_indices - 1, -1, N_source - 1)
    indices = np.where(diff != 0, int_indices, -1)
    return indices