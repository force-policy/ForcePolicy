from typing import Any, List, Literal

import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline

from utils.transforms.rotation import quat_angle
from utils.transforms.interpolation import quat_slerp


def interp_continuous_dynamics(
    pred_tcp_poses: np.ndarray,
    cur_tcp_pose: np.ndarray,
    cur_tcp_vel: np.ndarray,
    f_in: float,
    f_out: float,
    continuous_type: Literal["vel", "acc"] = "vel",
    return_vel: bool = False,
    return_acc: bool = False
):
    """
    Interpolate robot TCP trajectory with specified order (3 for Cubic, 5 for Quintic).

    Args:
        pred_tcp_poses:  (N, 7) [x, y, z, w, x, y, z] (Future points)
        cur_tcp_pose:    (7, )  [x, y, z, w, x, y, z] (Current state)
        cur_tcp_vel:     (6,)   [vx, vy, vz, wx, wy, wz] (Current velocity)
        f_in:            Input frequency
        f_out:           Output frequency
        continuous_type: vel (Cubic, Vel continuous) or acc (Quintic, Acc continuous).
        return_vel:      Whether to return velocity
        return_acc:      Whether to return acceleration
        
    Returns:
        tcp_pose: (M, 7)
        tcp_vel:  (M, 6) optional if return_vel is False
        tcp_acc:  (M, 6) optional if return_acc is False
    """
    # Prepare time indices
    poses = np.concatenate([cur_tcp_pose[None, :], pred_tcp_poses], axis = 0)
    N = len(poses)
    times_in = np.arange(N) / f_in
    dt_out = 1.0 / f_out
    times_out = np.arange(0, times_in[-1] + dt_out, dt_out)
    times_out = times_out[times_out <= times_in[-1]]

    # Helper to construct Omega matrix for (w, x, y, z) quaternion
    def get_omega_mat(w_vec):
        wx, wy, wz = w_vec
        return np.array([
            [0,   -wx, -wy, -wz],
            [wx,   0,  -wz,  wy],
            [wy,   wz,  0,  -wx],
            [wz,  -wy,  wx,   0]
        ])

    # -------------------------------------------------------------------------
    # PART A: Interpolate XYZ
    # -------------------------------------------------------------------------
    xyz_in = poses[:, :3]
    v_start_xyz = cur_tcp_vel[:3]
    v_end_xyz = np.zeros(3)

    if continuous_type == "vel":
        # Cubic Spline (C1)
        bc_xyz = ((1, v_start_xyz), (1, v_end_xyz))
        spline_xyz = CubicSpline(times_in, xyz_in, axis = 0, bc_type = bc_xyz)
    else:
        # Quintic Spline (C2)
        a_start_xyz = np.zeros(3)
        a_end_xyz = np.zeros(3)
        bc_xyz = ([(1, v_start_xyz), (2, a_start_xyz)], [(1, v_end_xyz), (2, a_end_xyz)])
        spline_xyz = make_interp_spline(times_in, xyz_in, k = 5, axis = 0, bc_type = bc_xyz)

    out_xyz = spline_xyz(times_out)
    if return_vel: out_v_xyz = spline_xyz(times_out, nu = 1)
    if return_acc: out_a_xyz = spline_xyz(times_out, nu = 2)

    # -------------------------------------------------------------------------
    # PART B: Interpolate Quaternion
    # -------------------------------------------------------------------------
    qs_in = poses[:, 3:] # [w, x, y, z]
    w_start = cur_tcp_vel[3:]
    q_start = qs_in[0]
    
    # Calculate q_dot_start
    Omega_w = get_omega_mat(w_start)
    q_dot_start = 0.5 * Omega_w @ q_start
    q_dot_end = np.zeros(4)

    if continuous_type == "vel":
        # Cubic Logic
        bc_quat = ((1, q_dot_start), (1, q_dot_end))
        spline_quat = CubicSpline(times_in, qs_in, axis = 0, bc_type = bc_quat)
    else:
        # Quintic Logic
        alpha_start = np.zeros(3)
        Omega_alpha = get_omega_mat(alpha_start)
        q_ddot_start = 0.5 * (Omega_alpha @ q_start + Omega_w @ q_dot_start)
        q_ddot_end = np.zeros(4)
        
        bc_quat = ([(1, q_dot_start), (2, q_ddot_start)], [(1, q_dot_end), (2, q_ddot_end)])
        spline_quat = make_interp_spline(times_in, qs_in, k = 5, axis = 0, bc_type = bc_quat)

    # Evaluate Spline
    out_qs_raw = spline_quat(times_out)
    if return_vel or return_acc: out_q_dots_raw = spline_quat(times_out, nu = 1)
    if return_acc: out_q_ddots_raw = spline_quat(times_out, nu = 2)

    # -------------------------------------------------------------------------
    # PART C: Post-process Quaternion (Normalize & Recover Physics)
    # -------------------------------------------------------------------------
    out_qs = np.zeros_like(out_qs_raw)
    if return_vel: out_ws = np.zeros((len(times_out), 3))
    if return_acc: out_alphas = np.zeros((len(times_out), 3))

    for i in range(len(times_out)):
        # 1. Normalize
        q_raw = out_qs_raw[i]
        norm = np.linalg.norm(q_raw)
        q_norm = q_raw / norm if norm > 1e-6 else q_raw
        out_qs[i] = q_norm
        
        if return_vel or return_acc:
            q_dot = out_q_dots_raw[i]
            qw, qx, qy, qz = q_norm
            
            # Inverse matrix for (w, x, y, z)
            Q_inv = np.array([
                [ qw,  qx,  qy,  qz],
                [-qx,  qw,  qz, -qy],
                [-qy, -qz,  qw,  qx],
                [-qz,  qy, -qx,  qw]
            ])
            
            # 2. Recover Angular Velocity
            if return_vel:
                w_homo = 2.0 * Q_inv @ q_dot
                out_ws[i] = w_homo[1:]
                
            # 3. Recover Angular Acceleration
            if return_acc:
                q_ddot = spline_quat(times_out[i], nu = 2) if continuous_type == "vel" else out_q_ddots_raw[i]
                
                w_curr = out_ws[i] if return_vel else (2.0 * Q_inv @ q_dot)[1:]
                Omega_w_curr = get_omega_mat(w_curr)
                
                rhs_vec = 2.0 * q_ddot - Omega_w_curr @ q_dot
                alpha_homo = Q_inv @ rhs_vec
                out_alphas[i] = alpha_homo[1:]

    # -------------------------------------------------------------------------
    # PART D: Combine Results
    # -------------------------------------------------------------------------
    results = [np.hstack((out_xyz, out_qs))[1:, ...]]
    if return_vel: results.append(np.hstack((out_v_xyz, out_ws))[1:, ...])
    if return_acc: results.append(np.hstack((out_a_xyz, out_alphas))[1:, ...])
    
    if len(results) == 1: return results[0]
    return tuple(results)


def interp_pose_by_step(
    pose_start: np.ndarray,
    pose_end: np.ndarray,
    step_pos: float = 0.05,
    step_rot: float = 0.05,
    with_start: bool = True,
    with_end: bool = True,
) -> List[np.ndarray]:
    """Generate interpolated poses between two poses using quaternion slerp (wxyz)."""
    pose_start = np.asarray(pose_start, dtype=np.float32)
    pose_end = np.asarray(pose_end, dtype=np.float32)

    n_pos = np.linalg.norm(pose_end[:3] - pose_start[:3]) / step_pos
    n_rot = quat_angle(pose_end[3:], pose_start[3:]) / step_rot
    n_steps = int(max(n_pos, n_rot))

    if n_steps < 1:
        subpoints = [pose_start.copy(), pose_end.copy()]
    else:
        alpha = np.linspace(0, 1, n_steps + 2, dtype=np.float32)
        interp_pos = pose_start[:3] + alpha[:, np.newaxis] * (pose_end[:3] - pose_start[:3])
        interp_rot = quat_slerp(pose_start[3:], pose_end[3:], alpha = alpha)
        subpoints = list(np.concatenate([interp_pos, interp_rot], axis=-1))

    start_idx = 0 if with_start else 1
    end_idx = len(subpoints) if with_end else len(subpoints) - 1
    return subpoints[start_idx:end_idx]


def interp_linear_dynamics(
    pred_tcp_poses: np.ndarray,
    cur_tcp_pose: np.ndarray,
    f_in: float,
    f_out: float
) -> np.ndarray:
    """
    Interpolate robot TCP trajectory with using linear dynamics.

    Args:
        pred_tcp_poses:  (N, 7) [x, y, z, w, x, y, z] (Future points)
        cur_tcp_pose:    (7, )  [x, y, z, w, x, y, z] (Current state)
        f_in:            Input frequency
        f_out:           Output frequency
        
    Returns:
        tcp_pose: (M, 7)
    """
    # Combine data
    all_poses = np.concatenate([cur_tcp_pose[None, :], pred_tcp_poses], axis=0)
    
    dt_pred = 1.0 / f_in
    dt_target = 1.0 / f_out
    
    final_tcps = []

    for i in range(1, len(all_poses)):
        p_from = all_poses[i - 1]
        p_to = all_poses[i]
        
        d_pos = np.linalg.norm(p_to[:3] - p_from[:3])
        d_rot = quat_angle(p_from[3:], p_to[3:])
        
        v_lin = d_pos / max(dt_pred, 1e-6)
        v_ang = d_rot / max(dt_pred, 1e-6)
        
        step_pos = max(v_lin * dt_target, 1e-6)
        step_rot = max(v_ang * dt_target, 1e-6)
        
        interp_points = interp_pose_by_step(
            pose_start = p_from,
            pose_end = p_to,
            step_pos = step_pos,
            step_rot = step_rot,
            with_start = False,
            with_end = False
        )
        
        if len(interp_points) > 0:
            final_tcps.extend(interp_points)

        final_tcps.append(p_to)

    return np.array(final_tcps)