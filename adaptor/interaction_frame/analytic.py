"""
Analytic Interaction Frame Selection

Reference: Mohammadi, Ali Mousavi, et al. "A generic task model and control strategy to support learning, robust control, and generalization of contact-rich manipulation tasks." Robotics and Autonomous Systems (2025): 105270.
"""
from typing import Union, Tuple

import numpy as np
from utils.transforms.projection import apply_mat_to_twist, apply_mat_to_wrench


def skew(v) -> np.ndarray:
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def calc_ASIP(
    a_vecs: np.ndarray,
    b_vecs: np.ndarray,
    epsilon: float = 1e-6
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    ASIP (Average Screw-axes Intersection Point)

    Inputs:
        a_vecs: [N, 3] directional componenet (w, f)
        b_vecs: [N, 3] moment component (v, m)

    Outputs:
        p_asip: [3,] optimal point
        cov_det: uncertainty (det of covariance matrix)
        cov: covariance matrix
    """
    N = len(a_vecs)
    
    A_sum = np.zeros((3, 3))
    RHS_sum = np.zeros(3)
    
    resid_sum = 0.0
    
    for i in range(N):
        a = a_vecs[i]
        b = b_vecs[i]
        a_skew = skew(a)
        A_sum += a_skew @ a_skew.T 
        RHS_sum += np.cross(a, b)

    A = A_sum / N
    RHS = RHS_sum / N
    
    A_reg = A + epsilon * np.eye(3)
    try:
        p_asip = np.linalg.solve(A_reg, RHS)
    except np.linalg.LinAlgError:
        p_asip = np.zeros(3) 

    resid_norms = np.linalg.norm(np.cross(a_vecs, p_asip) + b_vecs, axis=1)**2
    sigma_sq = np.sum(resid_norms) / (N * (3 * N - 3) + 1e-9)
    
    cov = sigma_sq * np.linalg.inv(A_reg)
    cov_det = np.linalg.det(cov)
    
    return p_asip, cov_det, cov


def calc_AVOF(
    vecs: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    AVOF (Average Vector Orientation Frame) 

    Inputs:
        vecs: [N, 3] interested vector

    Outputs:
        R: [3, 3] rotation matrix
        cov_det: uncertainty (det of covariance matrix)
        cov: covariance matrix
    """
    N = len(vecs)
    
    C_c = (vecs.T @ vecs) / N
    U, S, Vt = np.linalg.svd(C_c)
    if np.linalg.det(U) < 0:
        U[:, 2] *= -1
    R_avof = U
    
    trace_val = np.trace(C_c)
    if trace_val < 1e-9:
        cov = np.eye(3) * 1e9
    else:
        cov = C_c / trace_val
    cov_det = np.linalg.det(cov)
    
    return R_avof, cov_det, cov


def align_vectors(
    R_target: np.ndarray, 
    R_source: np.ndarray,
) -> np.ndarray:
    best_R = R_source.copy()
    max_score = -np.inf
    if np.dot(R_target[:, 0], R_source[:, 0]) < 0:
        R_source_flip = R_source.copy()
        R_source_flip[:, 0] *= -1
        R_source_flip[:, 1] *= -1
        return R_source_flip
    return R_source


def analyze_interaction_frame(
    wrench: np.ndarray, 
    twist: np.ndarray,
) -> np.ndarray:
    """
    Get optimal interaction frame
    
    Inputs:
        wrench: [N, 6] (Fx, Fy, Fz, Mx, My, Mz) in TCP
        twist:        [N, 6] (Vx, Vy, Vz, Wx, Wy, Wz) in TCP
    
    Outputs:
        if_frame: [4, 4] interaction frame in TCP
    """
    N = len(wrench)
    w_vecs = twist[:, 3:6]
    v_vecs = twist[:, 0:3]
    f_vecs = wrench[:, 0:3]
    m_vecs = wrench[:, 3:6]
    
    # --- STEP 1: Origin Selection (ASIP) ---
    
    # 1.1 Analyze Twist (Motion)
    # Model 1: Min Velocity (Pure Rotation)
    p_t1, det_t1, cov_t1 = calc_ASIP(w_vecs, v_vecs)

    # Model 2: Constant Velocity (Translation) -> Use mean-subtracted data
    w_mean = np.mean(w_vecs, axis=0)
    v_mean = np.mean(v_vecs, axis=0)
    p_t2, det_t2, cov_t2 = calc_ASIP(w_vecs - w_mean, v_vecs - v_mean)
    
    # Select best Motion Origin
    if det_t1 < det_t2:
        p_twist, cov_twist = p_t1, cov_t1
        motion_type = "Rotation"
    else:
        p_twist, cov_twist = p_t2, cov_t2
        motion_type = "Translation"

    # 1.2 Analyze Wrench (Force)
    # Model 1: Min Moment (Pure Force)
    p_w1, det_w1, cov_w1 = calc_ASIP(f_vecs, m_vecs)

    # Model 2: Constant Moment -> Mean subtracted
    f_mean = np.mean(f_vecs, axis=0)
    m_mean = np.mean(m_vecs, axis=0)
    p_w2, det_w2, cov_w2 = calc_ASIP(f_vecs - f_mean, m_vecs - m_mean)
    
    # Select best Wrench Origin
    if det_w1 < det_w2:
        p_wrench, cov_wrench = p_w1, cov_w1
        force_type = "PureForce"
    else:
        p_wrench, cov_wrench = p_w2, cov_w2
        force_type = "ConstMoment"
        
    # 1.3 Fuse Origins (Weighted Average)
    # p_avg = (C_t^-1 + C_w^-1)^-1 * (C_t^-1 * p_t + C_w^-1 * p_w)
    try:
        prec_twist = np.linalg.inv(cov_twist)
        prec_wrench = np.linalg.inv(cov_wrench)
        prec_sum = prec_twist + prec_wrench
        cov_avg = np.linalg.inv(prec_sum)
        
        p_optimal = cov_avg @ (prec_twist @ p_twist + prec_wrench @ p_wrench)
    except:
        # Fallback if singular: simple average
        p_optimal = (p_twist + p_wrench) / 2.0

    # --- STEP 2: Orientation Selection (AVOF) [cite: 468-477] ---
    
    # Calculate vectors at optimal origin
    v_at_p = v_vecs + np.cross(p_optimal, w_vecs) # shift linear velocity
    m_at_p = m_vecs + np.cross(p_optimal, f_vecs) # shift moment
    
    # 2.2 Select Vectors of Interest
    # If Motion is Rotation (Model 1) -> Interest is w (angular velocity)
    # If Motion is Translation (Model 2) -> Interest is v (linear velocity)
    vecs_motion = w_vecs if motion_type == "Rotation" else v_at_p
    
    # If Wrench is Pure Force (Model 1) -> Interest is f (force)
    # If Wrench is Const Moment (Model 2) -> Interest is m (moment)
    vecs_wrench = f_vecs if force_type == "PureForce" else m_at_p
    
    # 2.3 Calculate AVOF for both
    R_motion, det_R1, cov_R1 = calc_AVOF(vecs_motion)
    R_wrench, det_R2, cov_R2 = calc_AVOF(vecs_wrench)
    
    # 2.4 Align and Fuse Orientations
    # Step 1a: Align R_wrench to R_motion
    R_wrench_aligned = align_vectors(R_motion, R_wrench)
    
    # Step 1c: Weighted Average
    w1 = 1.0 / (det_R1 + 1e-9)
    w2 = 1.0 / (det_R2 + 1e-9)
    
    # Standard SO(3) averaging: SVD(w1*R1 + w2*R2)
    M = w1 * R_motion + w2 * R_wrench_aligned
    U_avg, _, Vt_avg = np.linalg.svd(M)
    R_optimal = U_avg @ Vt_avg
    
    # Ensure determinant is +1
    if np.linalg.det(R_optimal) < 0:
        U_avg[:, 2] *= -1
        R_optimal = U_avg @ Vt_avg

    # --- Construct Final Result ---
    if_frame = np.eye(4)
    if_frame[:3, :3] = R_optimal
    if_frame[:3, 3] = p_optimal
    
    return if_frame


