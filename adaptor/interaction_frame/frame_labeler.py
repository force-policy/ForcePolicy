from enum import Enum
from typing import Tuple, Optional

import numpy as np
from logger import logger

from adaptor.configs.interaction_frame.frame_labeler import *
from utils.transforms.projection import apply_mat_to_twist, apply_mat_to_wrench


class TaskType(Enum):
    FREE_MOTION = 0
    CONTACT_MOTION = 1
    CONTACT_ANCHOR = 2
    CONTACT_SCREW = 3
    CONTACT_LEVERAGE = 4
    UNCERTAIN = 5


def compute_confidence(
    task_id: TaskType,
    f_norm: float,
    v_norm: float,
    w_norm: float,
    v_perp_norm: float,
    v_z: float,
    config: AdvancedFrameLabelerConfig
) -> float:
    """
    Compute confidence score [0, 1] for the classification.
    
    Confidence is based on margin from decision boundaries:
    - Higher margin from threshold = higher confidence
    - Near threshold = lower confidence (potential misclassification)
    
    Formula: conf = sigmoid(margin / scale) where margin is distance from threshold
    This gives smooth transition: conf=0.5 at boundary, ~0.73 at 1x scale, ~0.88 at 2x scale
    """
    def margin_to_conf(margin: float, scale: float) -> float:
        """Convert margin to confidence using sigmoid-like function."""
        # Use tanh for smooth 0-1 mapping: tanh(x) in [-1,1], so (1+tanh)/2 in [0,1]
        normalized = margin / scale
        return float(0.5 * (1.0 + np.tanh(normalized)))
    
    if task_id == TaskType.FREE_MOTION:
        # Confidence based on: how far below contact threshold
        # margin > 0 means clearly free motion, margin < 0 means close to contact
        margin = config.thres_force - f_norm  # positive when clearly free
        return margin_to_conf(margin, scale=config.thres_force)
        
    elif task_id == TaskType.CONTACT_MOTION:
        # Classified when: v_norm > v_th AND (v_perp > v_th OR w_norm > w_th)
        # Confidence based on: how far above velocity threshold
        margin_v = v_norm - config.thres_lin_vel
        margin_v_perp = v_perp_norm - config.thres_lin_vel
        # Use the primary margin (perpendicular velocity for motion)
        margin = max(margin_v_perp, margin_v * 0.5)
        return margin_to_conf(margin, scale=config.thres_lin_vel)
        
    elif task_id == TaskType.CONTACT_ANCHOR:
        # Classified when: v_norm > v_th AND v_perp < v_th AND w_norm < w_th
        # This is a tricky case: high v along Z but low v_perp
        # Confidence based on: how far v_perp is BELOW threshold (lower = more anchored)
        margin = config.thres_lin_vel - v_perp_norm  # positive when clearly anchor
        return margin_to_conf(margin, scale=config.thres_lin_vel)
        
    elif task_id == TaskType.CONTACT_SCREW:
        # Classified when: v_norm < v_th AND w_norm > w_th AND is_parallel
        # Confidence based on: how far above angular velocity threshold
        margin = w_norm - config.thres_ang_vel
        return margin_to_conf(margin, scale=config.thres_ang_vel)
        
    elif task_id == TaskType.CONTACT_LEVERAGE:
        # Classified when: v_norm < v_th AND w_norm > w_th AND NOT is_parallel
        # Same as screw but rotation axis is different
        margin = w_norm - config.thres_ang_vel
        return margin_to_conf(margin, scale=config.thres_ang_vel)
    
    # Default / UNCERTAIN
    return 0.1


class BaseFrameLabeler:
    def __init__(self, config: FrameLabelerBaseConfig):
        self.config = config
    
    def __call__(
        self,
        if_frame: np.ndarray,
        wrench: np.ndarray,
        twist: np.ndarray,
        last_task_id: TaskType
    ) -> Tuple[TaskType, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class VanillaFrameLabeler(BaseFrameLabeler):
    def __init__(self, config: VanillaFrameLabelerConfig):
        super(VanillaFrameLabeler, self).__init__(config)
    
    def __call__(
        self, 
        if_frame: np.ndarray, 
        wrench: np.ndarray,        
        twist: np.ndarray, 
        last_task_id: TaskType,
        specified_task_id: Optional[TaskType] = None
    ) -> Tuple[TaskType, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Simple threshold-based adaptation (ForceMimic style).
        """
        N = len(wrench)
        f = np.mean(wrench, axis = 0)[0: 3]
        f_norm = np.linalg.norm(f)
        is_contact = f_norm > self.config.thres_force
        
        if is_contact:
            task_id = TaskType.CONTACT_MOTION
        else:
            task_id = TaskType.FREE_MOTION
            if_frame = np.eye(4)
            
        confidence = 1.0  # Simple labeler has no confidence measure
        
        unified_poses = np.zeros((N, 4, 4), dtype = np.float32)
        unified_twists = np.zeros((N, 6), dtype = np.float32)
        unified_wrenches = np.zeros((N, 6), dtype = np.float32)
        unified_masks = np.zeros((N, 6), dtype = np.float32)
        unified_ref_force = np.zeros((N, 6), dtype = np.float32)
        
        inv_if_frame = np.linalg.inv(if_frame)
        
        for t in range(N):
            unified_poses[t] = if_frame
            unified_twists[t] = apply_mat_to_twist(twist[t], mat = inv_if_frame)
            unified_wrenches[t] = apply_mat_to_wrench(wrench[t], mat = inv_if_frame)
            
            if is_contact:
                unified_masks[t] = np.array([0, 0, 1, 0, 0, 0], dtype = np.float32)
                unified_ref_force[t] = np.array([0, 0, unified_wrenches[t, 2], 0, 0, 0], dtype = np.float32)
            else:
                unified_masks[t] = np.array([0, 0, 0, 0, 0, 0], dtype = np.float32)
                unified_ref_force[t] = np.zeros(6, dtype = np.float32)

        return task_id, confidence, unified_poses, unified_wrenches, unified_twists, unified_masks, unified_ref_force


class AdvancedFrameLabeler(BaseFrameLabeler):
    def __init__(self, config: AdvancedFrameLabelerConfig):
        super(AdvancedFrameLabeler, self).__init__(config)
    
    def __call__(
        self, 
        if_frame: np.ndarray, 
        wrench: np.ndarray,        
        twist: np.ndarray, 
        last_task_id: TaskType,
        specified_task_id: Optional[TaskType] = None
    ) -> Tuple[TaskType, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Unify frame definition and classify task (Default/Our Type).
        """
        # Use mean wrench for contact & Z-selection
        wrench_mean = np.mean(wrench, axis = 0) 
        force_mean = wrench_mean[0: 3]
        torque_mean = wrench_mean[3: 6]
        f_norm = np.linalg.norm(force_mean)
        t_norm = np.linalg.norm(torque_mean)
        
        is_contact = (f_norm > self.config.thres_force) or (t_norm > self.config.thres_torque)
        
        task_id = TaskType.FREE_MOTION
        best_axis_idx = 0
        z_axis_ref = np.array([0, 0, 1])

        if is_contact:
            # 1. Select Z-axis Candidate Index Analysis
            # R_ref = if_frame[0:3, 0:3] 
            # target_z_dir = force_mean / (f_norm + 1e-9)
            # force_projections = np.array([np.dot(target_z_dir, R_ref[:, k]) for k in range(3)])
            # best_axis_idx = np.argmax(np.abs(force_projections))
            # z_axis_ref = R_ref[:, best_axis_idx]
            # if force_projections[best_axis_idx] < 0:
            #     z_axis_ref = -z_axis_ref

            # 1. Directly use Identifier's Z-axis as the normal reference
            R_ref = if_frame[0:3, 0:3] 
            z_axis_ref = R_ref[:, 2]
            
            # Fix SVD sign ambiguity: ensure Z-axis points into the contact surface
            if np.dot(z_axis_ref, force_mean) < 0:
                z_axis_ref = -z_axis_ref


            # 2. Classification Analysis (using Mean Velocities)
            twist_mean = np.mean(twist, axis = 0)
            v_mean = twist_mean[0: 3]
            w_mean = twist_mean[3: 6]
            v_norm_mean = np.linalg.norm(v_mean)
            w_norm_mean = np.linalg.norm(w_mean)
            
            v_dot_z = np.dot(v_mean, z_axis_ref)
            v_perp_vec = v_mean - v_dot_z * z_axis_ref
            v_perp_norm_mean = np.linalg.norm(v_perp_vec)
            
            v_th = self.config.thres_lin_vel
            w_th = self.config.thres_ang_vel
            
            if w_norm_mean > 1e-6:
                w_dir = w_mean / w_norm_mean
                cos_theta = np.abs(np.dot(w_dir, z_axis_ref))
                is_parallel = cos_theta > self.config.thres_is_parallel
            else:
                is_parallel = False

            if v_norm_mean < v_th and w_norm_mean > w_th:
                if is_parallel:
                    task_id = TaskType.CONTACT_SCREW
                else:
                    task_id = TaskType.CONTACT_LEVERAGE
            
            elif v_norm_mean > v_th and w_norm_mean < w_th:
                if v_perp_norm_mean > v_th:
                    task_id = TaskType.CONTACT_MOTION
                else:
                    task_id = TaskType.CONTACT_ANCHOR
            
            elif v_norm_mean > v_th and w_norm_mean > w_th:
                task_id = TaskType.CONTACT_MOTION
            
            else:
                task_id = TaskType.UNCERTAIN
            
        if specified_task_id is not None:
            task_id = specified_task_id
        
        # Compute confidence based on task type
        # UNCERTAIN gets low confidence, others use margin-based calculation
        if task_id == TaskType.UNCERTAIN:
            confidence = 0.1
        else:
            confidence = compute_confidence(
                task_id = task_id,
                f_norm = f_norm,
                v_norm = v_norm_mean if is_contact else 0.0,
                w_norm = w_norm_mean if is_contact else 0.0,
                v_perp_norm = v_perp_norm_mean if is_contact else 0.0,
                v_z = v_dot_z if is_contact else 0.0,
                config = self.config
            )

        N = len(wrench)
        unified_poses = np.zeros((N, 4, 4), dtype = np.float32)
        unified_twists = np.zeros((N, 6), dtype = np.float32)
        unified_wrenches = np.zeros((N, 6), dtype = np.float32)
        unified_masks = np.zeros((N, 6), dtype = np.float32)
        unified_ref_force = np.zeros((N, 6), dtype = np.float32)
        
        R_tcp_opt_t = if_frame[0:3, 0:3]
        p_tcp_opt_t = if_frame[0:3, 3]

        for t in range(N):
            if task_id == TaskType.FREE_MOTION:
                R_tcp_unified_t = np.eye(3)
                p_tcp_unified_t = np.zeros(3)
            else:
                # 1. Z-Axis (Instantaneous from selected index)
                z_axis = z_axis_ref
                    
                # 2. X-Axis (Using Task Rule on Instantaneous Data)
                v_t = twist[t, 0:3]
                w_t = twist[t, 3:6]
                
                x_axis = np.array([1, 0, 0])
                
                if task_id == TaskType.CONTACT_SCREW:
                    # X: TCP X proj on Perp Z
                    x_base = np.array([0, 1, 0])
                    x_proj = x_base - np.dot(x_base, z_axis) * z_axis
                    if np.linalg.norm(x_proj) < 1e-3:
                        x_base = np.array([1, 0, 0])
                        x_proj = x_base - np.dot(x_base, z_axis) * z_axis
                    x_axis = x_proj / np.linalg.norm(x_proj)
                        
                elif task_id == TaskType.CONTACT_LEVERAGE:
                    # X: w x Z
                    x_cross = np.cross(w_t, z_axis)
                    if np.linalg.norm(x_cross) < 1e-6:
                        twist_mean = np.mean(twist, axis = 0)
                        w_mean = twist_mean[3: 6]
                        x_cross = np.cross(w_mean, z_axis)
                        if np.linalg.norm(x_cross) < 1e-6:
                            x_cross = np.cross(np.array([1, 0, 0]), z_axis)
                    x_axis = x_cross / np.linalg.norm(x_cross)
                        
                elif task_id == TaskType.CONTACT_MOTION:
                    # X: v_perp
                    v_dot_z = np.dot(v_t, z_axis)
                    v_perp_vec = v_t - v_dot_z * z_axis
                    if np.linalg.norm(v_perp_vec) < 1e-6:
                        twist_mean = np.mean(twist, axis = 0)
                        v_mean = twist_mean[0: 3]
                        v_perp_vec = v_mean - np.dot(v_mean, z_axis) * z_axis
                        if np.linalg.norm(v_perp_vec) < 1e-6:
                            v_perp_vec = np.array([1, 0, 0])
                    x_axis = v_perp_vec / np.linalg.norm(v_perp_vec)
                        
                elif task_id == TaskType.CONTACT_ANCHOR:
                    # raw
                    # y_axis_tcp = np.array([0, 1, 0])
                    # x_proj = y_axis_tcp - np.dot(y_axis_tcp, z_axis) * z_axis
                    # if np.linalg.norm(x_proj) < 1e-3:
                    #     x_proj = np.array([1, 0, 0])
                    # x_axis = x_proj / np.linalg.norm(x_proj)

                    # for practice
                    x_axis = z_axis
                    z_base = np.array([0, 0, 1])
                    z_proj = z_base - np.dot(z_base, x_axis) * x_axis
                    if np.linalg.norm(z_proj) < 1e-3:
                        z_base = np.array([0, 1, 0])
                        z_proj = z_base - np.dot(z_base, x_axis) * x_axis
                    z_axis = z_proj / np.linalg.norm(z_proj)
                
                # Orthogonalize
                # Y = Z x X
                y_axis = np.cross(z_axis, x_axis)

                if np.linalg.norm(y_axis) < 1e-6:
                    y_axis = np.array([0, 1, 0])
                    x_axis = np.cross(np.array([0, 1, 0]), z_axis)
                    y_axis = np.cross(z_axis, x_axis)
                else:
                    y_axis = y_axis / np.linalg.norm(y_axis)
                
                # Recompute X to ensure strict orthogonality
                x_axis = np.cross(y_axis, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                
                # Frame Construction
                R_tcp_unified_t = np.column_stack([x_axis, y_axis, z_axis])
                p_tcp_unified_t = p_tcp_opt_t

            # Store Results
            T_tcp_unified = np.eye(4)
            T_tcp_unified[0:3, 0:3] = R_tcp_unified_t
            T_tcp_unified[0:3, 3] = p_tcp_unified_t
            unified_poses[t] = T_tcp_unified
            unified_twists[t] = apply_mat_to_twist(twist[t], mat = np.linalg.inv(T_tcp_unified))
            unified_wrenches[t] = apply_mat_to_wrench(wrench[t], mat = np.linalg.inv(T_tcp_unified))
            
            # Labeling masks
            if task_id == TaskType.FREE_MOTION:
                unified_masks[t] = np.array([0, 0, 0, 0, 0, 0], dtype = np.float32)
                unified_ref_force[t] = np.array([0, 0, 0, 0, 0, 0], dtype = np.float32)
            elif task_id == TaskType.CONTACT_SCREW:
                unified_masks[t] = np.array([0, 0, 1, 0, 0, 1], dtype = np.float32)
                unified_ref_force[t] = np.array([0, 0, unified_wrenches[t, 2], 0, 0, unified_wrenches[t, 5]], dtype = np.float32)
            elif task_id == TaskType.CONTACT_LEVERAGE:
                unified_masks[t] = np.array([1, 1, 1, 0, 0, 0], dtype = np.float32)
                unified_ref_force[t] = np.array([0, 0, 0, 0, 0, 0], dtype = np.float32)
            elif task_id == TaskType.CONTACT_MOTION:
                unified_masks[t] = np.array([0, 0, 1, 0, 0, 0], dtype = np.float32)
                unified_ref_force[t] = np.array([0, 0, unified_wrenches[t, 2], 0, 0, 0], dtype = np.float32)
            elif task_id == TaskType.CONTACT_ANCHOR:
                # for raw
                # unified_masks[t] = np.array([1, 1, 1, 1, 1, 1], dtype = np.float32)
                # unified_ref_force[t] = np.array([0, 0, unified_wrenches[t, 2], 0, 0, 0], dtype = np.float32)
                # for practice
                unified_masks[t] = np.array([1, 1, 1, 1, 1, 1], dtype = np.float32)
                unified_ref_force[t] = np.array([unified_wrenches[t, 0], 0, 0, 0, 0, 0], dtype = np.float32)
            elif task_id == TaskType.UNCERTAIN:
                unified_masks[t] = np.array([-1, -1, -1, -1, -1, -1], dtype = np.float32)
                unified_ref_force[t] = np.zeros(6, dtype = np.float32)

        return task_id, confidence, unified_poses, unified_wrenches, unified_twists, unified_masks, unified_ref_force
