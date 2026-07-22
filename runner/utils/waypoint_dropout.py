"""
Waypoint dropout utilities for the adaptive scheduler.
Uses DTW to align recent robot trajectory with predicted trajectory.
"""

import numpy as np
from easydict import EasyDict as edict

from utils.transforms.rotation import quat_angle
from runner.utils.scheduler_utils import compute_cartesian_velocity_from_poses
from logger import logger


def waypoint_dropout(
    raw_tcps: np.ndarray,
    ref_traj: np.ndarray,
    dropout_cfg: edict,
    dt: float,
) -> int:
    """
    DTW-based waypoint dropout: align recent trajectory with predicted trajectory
    to find the best starting point.
    
    The key insight:
    - ref_traj = robot's recent history, ref_traj[-1] is current position
    - raw_tcps = model's predicted future trajectory
    - We align ref_traj's TAIL with raw_tcps's HEAD to find where to start
    
    Args:
        raw_tcps: Raw TCP keypoints from model (N, ..., 7) with wxyz quaternion
        ref_traj: Recent TCP trajectory (M, ..., 7), ref_traj[-1] is current position
        dropout_cfg: Dropout configuration with max_dropout and weight
        dt: Time step between raw keypoints (for velocity estimation)
    
    Returns:
        start_idx: Index of first waypoint to keep in raw_tcps
    """
    if not dropout_cfg.enable:
        return 0
    
    raw_tcps = np.asarray(raw_tcps, dtype = np.float32)
    ref_traj = np.asarray(ref_traj, dtype = np.float32)
    
    if len(raw_tcps) <= 1:
        return 0
    
    if len(ref_traj) == 0:
        return 0
    
    max_dropout = dropout_cfg.max_dropout
    if max_dropout <= 0:
        max_dropout = len(raw_tcps)
    
    # Use DTW to align the tail of ref_traj with the head of raw_tcps
    start_idx = dtw_align_trajectories(
        predicted = raw_tcps,
        history = ref_traj,
        max_dropout = max_dropout,
        w_linear = dropout_cfg.weight_linear,
        w_angular = dropout_cfg.weight_angular,
        w_linear_vel = dropout_cfg.weight_linear_vel,
        w_angular_vel = dropout_cfg.weight_angular_vel,
        dt=dt,
    )
    
    logger.debug(
        "[Dropout] DTW: {} raw points -> start_idx={} (keep {})",
        len(raw_tcps), start_idx, len(raw_tcps) - start_idx
    )
    
    return start_idx


def _approx_vel(pose_seq: np.ndarray, dt: float) -> np.ndarray:
    """
    Approximate velocity from pose sequence.
    
    Args:
        pose_seq: (K, ..., 7) poses with wxyz quaternion
        dt: Time step
    
    Returns:
        (K, ..., 6) velocities (linear + angular)
    """
    shape = pose_seq.shape
    K = shape[0]
    
    # Reshape to (K, N_items, 7) to handle arbitrary middle dimensions
    flat_seq = pose_seq.reshape(K, -1, 7)
    N_items = flat_seq.shape[1]
    
    vel = np.zeros((K, N_items, 6), dtype=np.float32)
    if K < 2:
        # Reshape back to original structure but with last dim 6
        return vel.reshape(shape[:-1] + (6,))
    
    for i in range(1, K):
        for j in range(N_items):
            vel[i, j] = compute_cartesian_velocity_from_poses(flat_seq[i - 1, j], flat_seq[i, j], dt)
    
    vel[0] = vel[1]
    return vel.reshape(shape[:-1] + (6,))


def _frame_cost(
    a_pose: np.ndarray,
    b_pose: np.ndarray,
    a_vel: np.ndarray,
    b_vel: np.ndarray,
    w_linear: float,
    w_angular: float,
    w_linear_vel: float,
    w_angular_vel: float
) -> float:
    """
    Compute frame cost between two poses with velocities.
    
    Args:
        a_pose: (..., 7) pose with wxyz quaternion
        b_pose: (..., 7) pose with wxyz quaternion
        a_vel: (..., 6) velocity or None
        b_vel: (..., 6) velocity or None
        w_linear: Linear weight
        w_angular: Angular weight
        w_linear_vel: Linear velocity weight
        w_angular_vel: Angular velocity weight
    
    Returns:
        Weighted cost (summed over all items in ...)
    """
    # Flatten to (N_items, 7) and (N_items, 6)
    a_p = a_pose.reshape(-1, 7)
    b_p = b_pose.reshape(-1, 7)
    
    # Sum costs over all items (e.g. sum over arms)
    total_cost = 0.0
    
    for i in range(a_p.shape[0]):
        pos_cost = np.linalg.norm(a_p[i, :3] - b_p[i, :3])
        ang_cost = quat_angle(a_p[i, 3:], b_p[i, 3:])
        
        v_lin_cost = 0.0
        v_ang_cost = 0.0
        
        if a_vel is not None and b_vel is not None:
            a_v = a_vel.reshape(-1, 6)
            b_v = b_vel.reshape(-1, 6)
            v_lin_cost = np.linalg.norm(a_v[i, :3] - b_v[i, :3])
            v_ang_cost = np.linalg.norm(a_v[i, 3:] - b_v[i, 3:])
        
        total_cost += (
            w_linear * pos_cost +
            w_angular * ang_cost +
            w_linear_vel * v_lin_cost +
            w_angular_vel * v_ang_cost
        )
            
    return total_cost


def dtw_align_trajectories(
    predicted: np.ndarray,
    history: np.ndarray,
    max_dropout: int,
    w_linear: float,
    w_angular: float,
    w_linear_vel: float,
    w_angular_vel: float,
    dt: float,
) -> int:
    """
    Use DTW to align the tail of history with the head of predicted trajectory.
    
    Args:
        predicted: Predicted trajectory (N, ..., 7)
        history: Historical trajectory (M, ..., 7)
        max_dropout: Maximum search range
        w_linear: Linear weight
        w_angular: Angular weight
        w_linear_vel: Linear velocity weight
        w_angular_vel: Angular velocity weight
        dt: Time step for velocity estimation
    
    Returns:
        start_idx: Index in predicted that best matches current position
    """
    predicted = np.asarray(predicted, dtype = np.float32)
    history = np.asarray(history, dtype = np.float32)
    
    N = len(predicted)
    M = len(history)
    
    if N <= 1:
        return 0
    if M == 0:
        return 0
    
    # Limit the alignment to max_dropout frames
    k = min(max_dropout, N, M)
    
    A = predicted[:k]   # First k frames of predicted (k, ..., 7)
    B = history[-k:]    # Last k frames of history (k, ..., 7)
    
    m = len(A)
    n = len(B)
    
    if m <= 1 or n <= 1:
        return 0
    
    dt = max(dt, 1e-6)
    
    # Approximate velocities
    Av = _approx_vel(A, dt)
    Bv = _approx_vel(B, dt)
    
    # Standard DTW
    INF = 1e12
    D = np.full((m + 1, n + 1), INF, dtype = np.float32)
    D[0, 0] = 0.0
    
    # Track parent for backtracking: 0=diag, 1=up(i-1,j), 2=left(i,j-1)
    parent = np.zeros((m + 1, n + 1), dtype = np.int8)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = _frame_cost(A[i - 1], B[j - 1], Av[i - 1], Bv[j - 1], w_linear, w_angular, w_linear_vel, w_angular_vel)
            
            candidates = [
                (D[i - 1, j - 1], 0),  # diagonal
                (D[i - 1, j], 1),      # up (vertical)
                (D[i, j - 1], 2),      # left (horizontal)
            ]
            min_cost, min_parent = min(candidates, key = lambda x: x[0])
            D[i, j] = cost + min_cost
            parent[i, j] = min_parent
    
    # Find best endpoint for history (j=n)
    best_i = m
    best_cost = D[m, n]
    
    for i in range(1, m + 1):
        if D[i, n] < best_cost:
            best_cost = D[i, n]
            best_i = i
    
    # Backtrack
    i, j = best_i, n
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        p = parent[i, j]
        if p == 0:
            i -= 1
            j -= 1
        elif p == 1:
            i -= 1
        else:
            j -= 1
    
    path.reverse()
    
    if len(path) > 0:
        for a_idx, b_idx in reversed(path):
            if b_idx == n - 1:
                start_idx = a_idx
                break
        else:
            start_idx = path[-1][0]
    else:
        start_idx = 0
    
    start_idx = max(0, min(N - 1, start_idx))
    
    logger.trace(
        "[DTW] m={}, n={}, best_i={}, path_len={}, start_idx={}",
        m, n, best_i, len(path), start_idx
    )
    
    return int(start_idx)
