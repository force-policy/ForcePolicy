from typing import List, Optional

import numpy as np

from adaptor.configs.interaction_frame.frame_identifier import *
from adaptor.interaction_frame.analytic import analyze_interaction_frame
from adaptor.interaction_frame.optimization import optimize_interaction_frame


class BaseFrameIdentifier:
    def __init__(self, config: FrameIdentifierBaseConfig):
        self.config = config
    
    def __call__(
        self,
        wrench: np.ndarray,
        twist: np.ndarray,
        pose: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError


class ForceOnlyFrameIdentifier(BaseFrameIdentifier):
    """
    Force-only frame identification.
    """
    def __init__(self, config: ForceOnlyFrameIdentifierConfig):
        super(ForceOnlyFrameIdentifier, self).__init__(config)
    
    def __call__(
        self,
        wrench: np.ndarray,
        twist: np.ndarray,
        pose: np.ndarray
    ) -> np.ndarray:
        """ Identify interaction frame. """
        if_frame = np.eye(4)
        
        f = wrench[:, :3].mean(axis = 0)
        v = twist[:, :3].mean(axis = 0)
        
        # Identify z axis
        z_axis = f / (np.linalg.norm(f) + 1e-6)
        x_axis = v / (np.linalg.norm(v) + 1e-6)
        dot_prod = np.abs(np.sum(x_axis * z_axis))
        x_axis[dot_prod > self.config.thres_parallel] = np.array([0, 1, 0])
            
        # Generate orthogonal frame
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
            
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
            
        # Construct rotation matrix [x, y, z]
        R = np.stack([x_axis, y_axis, z_axis], axis = -1)
        if_frame[:3, :3] = R            
        return if_frame


class WrenchOnlyFrameIdentifier(BaseFrameIdentifier):
    """
    Wrench-only frame identification.
    """
    def __init__(self, config: WrenchOnlyFrameIdentifierConfig):
        super().__init__(config)
    
    def __call__(
        self,
        wrench: np.ndarray,
        twist: np.ndarray,
        pose: np.ndarray
    ) -> np.ndarray:
        if_frame = np.eye(4)

        f = wrench[:, :3]
        m = wrench[:, 3:]
        combined_wrench = np.vstack((f, m * self.config.weight_torque))

        try:
            U, S, Vh = np.linalg.svd(combined_wrench, full_matrices = False)
            z_axis = Vh[0]
        except np.linalg.LinAlgError:
            z_axis = np.array([0, 0, 1])

        v = twist[:, :3].mean(axis = 0)
        x_axis = v / (np.linalg.norm(v) + 1e-6)
        dot_prod = np.abs(np.sum(z_axis * x_axis))
        z_axis[dot_prod > self.config.thres_parallel] = np.array([0, 1, 0])
        
        # Generate orthogonal frame
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= (np.linalg.norm(y_axis) + 1e-6)

        x_axis = np.cross(y_axis, z_axis)
        x_axis /= (np.linalg.norm(x_axis) + 1e-6)

        # Construct rotation matrix [x, y, z]
        R = np.stack([x_axis, y_axis, z_axis], axis = -1)
        if_frame[:3, :3] = R            
        return if_frame


class LinearVelocityOnlyFrameIdentifier(BaseFrameIdentifier):
    """
    Linear velocity-only frame identification.
    """
    def __init__(self, config: LinearVelocityOnlyFrameIdentifierConfig):
        super(LinearVelocityOnlyFrameIdentifier, self).__init__(config)
    
    def __call__(
        self,
        wrench: np.ndarray,
        twist: np.ndarray,
        pose: np.ndarray
    ) -> np.ndarray:
        """ Identify interaction frame. """
        if_frame = np.eye(4)
        
        f = wrench[:, :3].mean(axis = 0)
        v = twist[:, :3].mean(axis = 0)
        
        # Identify z axis
        z_axis = f / (np.linalg.norm(f) + 1e-6)
        x_axis = v / (np.linalg.norm(v) + 1e-6)
        dot_prod = np.abs(np.sum(x_axis * z_axis))
        z_axis[dot_prod > self.config.thres_parallel] = np.array([0, 1, 0])
            
        # Generate orthogonal frame
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
            
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)
            
        # Construct rotation matrix [x, y, z]
        R = np.stack([x_axis, y_axis, z_axis], axis = -1)
        if_frame[:3, :3] = R            
        return if_frame


class TwistOnlyFrameIdentifier(BaseFrameIdentifier):
    """
    Twist-only frame identification.
    """
    def __init__(self, config: TwistOnlyFrameIdentifierConfig):
        super(TwistOnlyFrameIdentifier, self).__init__(config)
    
    def __call__(
        self,
        wrench: np.ndarray,
        twist: np.ndarray,
        pose: np.ndarray
    ) -> np.ndarray:
        """ Identify interaction frame. """
        if_frame = np.eye(4)

        v = twist[:, :3]
        w = twist[:, 3:]
        combined_motion = np.vstack((v, w * self.config.weight_angular))

        try:
            U, S, Vh = np.linalg.svd(combined_motion, full_matrices = False)
            x_axis = Vh[0]
        except np.linalg.LinAlgError:
            x_axis = np.array([1, 0, 0])
        
        f = wrench[:, :3].mean(axis = 0)
        z_axis = f / (np.linalg.norm(f) + 1e-6)
        dot_prod = np.abs(np.sum(x_axis * z_axis))
        z_axis[dot_prod > self.config.thres_parallel] = np.array([0, 1, 0])
            
        # Generate orthogonal frame
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
            
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)
            
        # Construct rotation matrix [x, y, z]
        R = np.stack([x_axis, y_axis, z_axis], axis = -1)
        if_frame[:3, :3] = R            
        return if_frame


class TwistWrenchFrameIdentifier(BaseFrameIdentifier):
    """
    Twist/wrench frame identification.
    
    Config specify modes:
    - 'auto': Use velocity threshold to decide (original logic)
    - 'twist': Force twist-dominant mode (X-axis from motion SVD)
    - 'wrench': Force wrench-dominant mode (Z-axis from wrench SVD)

    The `specify` argument of __call__ is an OPTIONAL per-patch override (e.g. produced by
    scripts/classify_power_source.py). When it is None (default) the behavior is unchanged and
    the single `self.config.specify` value is used, so existing pipelines are unaffected.
    """
    def __init__(self, config: TwistWrenchFrameIdentifierConfig):
        super(TwistWrenchFrameIdentifier, self).__init__(config)
    
    def __call__(
        self,
        wrench: np.ndarray,
        twist: np.ndarray,
        pose: np.ndarray,
        specify: Optional[str] = None
    ) -> np.ndarray:
        """ Identify interaction frame. """
        if_frame = np.eye(4)
        
        v = twist[:, :3]
        w = twist[:, 3:]
        f = wrench[:, :3]
        m = wrench[:, 3:]

        v_norm = np.linalg.norm(v, axis = 1)
        w_norm = np.linalg.norm(w, axis = 1)

        # Per-patch override takes precedence over the single global config value.
        mode = specify if specify is not None else self.config.specify

        # Determine which mode to use
        if mode == "twist":
            use_twist_dominant = True
        elif mode == "wrench":
            use_twist_dominant = False
        else:  # 'auto': use threshold-based decision (original logic)
            use_twist_dominant = (v_norm.mean() > self.config.thres_lin_vel or 
                                  w_norm.mean() > self.config.thres_ang_vel)

        if use_twist_dominant:
            # Twist-dominant: X-axis from motion SVD, Z-axis from force
            combined_motion = np.vstack((v, w * self.config.weight_angular))
            U, S, Vh = np.linalg.svd(combined_motion, full_matrices = False)
            x_axis = Vh[0]
        
            f_mean = f.mean(axis = 0)
            z_axis = f_mean / (np.linalg.norm(f_mean) + 1e-6)
            dot_prod = np.abs(np.sum(x_axis * z_axis))
            z_axis[dot_prod > self.config.thres_parallel] = np.array([0, 1, 0])
                
            # Generate orthogonal frame
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
                
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)
                
            # Construct rotation matrix [x, y, z]
            R = np.stack([x_axis, y_axis, z_axis], axis = -1)
            if_frame[:3, :3] = R            
        else:
            # Wrench-dominant: Z-axis from wrench SVD, X-axis from velocity
            combined_wrench = np.vstack((f, m * self.config.weight_torque))
            U, S, Vh = np.linalg.svd(combined_wrench, full_matrices = False)
            z_axis = Vh[0]
            
            v_mean = v.mean(axis = 0)
            x_axis = v_mean / (np.linalg.norm(v_mean) + 1e-6)
            dot_prod = np.abs(np.sum(z_axis * x_axis))
            z_axis[dot_prod > self.config.thres_parallel] = np.array([0, 1, 0])
            
            # Generate orthogonal frame
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)

            x_axis = np.cross(y_axis, z_axis)
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)

            # Construct rotation matrix [x, y, z]
            R = np.stack([x_axis, y_axis, z_axis], axis = -1)
            if_frame[:3, :3] = R            
        
        return if_frame
    

class AnalyticFrameIdentifier(BaseFrameIdentifier):
    """
    Analytic-style frame identification.
    """
    def __init__(self, config: AnalyticFrameIdentifierConfig):
        super(AnalyticFrameIdentifier, self).__init__(config)
    
    def __call__(
        self,
        wrench: np.ndarray,
        twist: np.ndarray,
        pose: np.ndarray
    ) -> np.ndarray:
        """ Identify interaction frame. """
        if_frame = analyze_interaction_frame(wrench, twist)
        if self.config.frame_origin_tcp:
            if_frame[:3, 3] = 0
        return if_frame


class OptimizationFrameIdentifier(BaseFrameIdentifier):
    """
    Optimization-style frame identification.
    """
    def __init__(self, config: OptimizationFrameIdentifierConfig):
        super(OptimizationFrameIdentifier, self).__init__(config)
    
    def __call__(
        self,
        wrench: np.ndarray,
        twist: np.ndarray,
        pose: np.ndarray
    ) -> np.ndarray:
        """ Identify interaction frame. """
        return optimize_interaction_frame(
            wrench,
            twist,
            pose,
            with_origin = (not self.config.frame_origin_tcp),
            num_steps = self.config.num_steps,
            lr = self.config.lr,
            loss_types = self.config.loss_types,
            loss_weights = self.config.loss_weights
        )


class TwoStageFrameIdentifier(BaseFrameIdentifier):
    """
    Two-stage frame identification (analytic + optimization).
    """
    def __init__(self, config: TwoStageFrameIdentifierConfig):
        super(TwoStageFrameIdentifier, self).__init__(config)
    
    def __call__(
        self,
        wrench: np.ndarray,
        twist: np.ndarray,
        pose: np.ndarray
    ) -> np.ndarray:
        if_frame_0 = analyze_interaction_frame(wrench, twist)      
        if_frame_opt = optimize_interaction_frame(
            wrench,
            twist,
            pose,
            with_origin = (not self.config.frame_origin_tcp),
            num_steps = self.config.num_steps,
            lr = self.config.lr,
            init_guess = if_frame_0,
            loss_types = self.config.loss_types,
            loss_weights = self.config.loss_weights
        )
        return if_frame_opt
