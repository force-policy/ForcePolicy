""" 1Euro filter implementation for pose filtering.

Reference: https://cristal.univ-lille.fr/~casiez/1euro/
"""

import time
import numpy as np

from utils.transforms.rotation import RotationType
from utils.transforms.interpolation import quat_slerp
from utils.transforms.rotation import rotation_transform
from runner.configs.filter import OneEuroFilterConfig, RotationOneEuroFilterConfig, PoseOneEuroFilterConfig


def smoothing_factor(t_e, cutoff):
    r = 2 * np.pi * cutoff
    return 1.0 / (1.0 + r * t_e)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    """
    1Euro filter for one-dimensional or multi-dimensional signals.
    Reduces jitter while maintaining responsiveness to intentional movements.
    
    Reference: https://cristal.univ-lille.fr/~casiez/1euro/
    """
    def __init__(self, config: OneEuroFilterConfig):
        """
        Args:
            config: Configuration object.
        """
        self.mincutoff = np.array(config.mincutoff)
        self.beta = np.array(config.beta)
        self.dcutoff = np.array(config.dcutoff)
            
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
        
    def reset(self):
        """ Reset the filter state. """
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
        
    def __call__(self, x, t=None):
        """
        Filter a value.
        
        Args:
            x: Input value (scalar or array)
            t: Timestamp (seconds). If None, uses time.time()
            
        Returns:
            Filtered value
        """
        if t is None:
            t = time.time()
            
        x = np.array(x)
            
        if self.x_prev is None:
            # First value, no filtering
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x
            
        # Compute time delta
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
            
        # Estimate derivative (velocity)
        dx = (x - self.x_prev) / dt
        
        # Filter the derivative
        a_d = smoothing_factor(dt, self.dcutoff)
        edx = exponential_smoothing(a_d, dx, self.dx_prev)
        self.dx_prev = edx
        
        # Compute adaptive cutoff frequency
        cutoff = self.mincutoff + self.beta * np.abs(edx)
        
        # Filter the signal
        a = smoothing_factor(dt, cutoff)
        filtered_x = exponential_smoothing(a, x, self.x_prev)
        
        # Update state
        self.x_prev = filtered_x
        self.t_prev = t
        
        return filtered_x


class RotationOneEuroFilter:
    """
    1Euro filter for quaternion rotations using SLERP.
    Filters rotations by computing angular velocity and using adaptive SLERP interpolation.
    """
    def __init__(self, config: RotationOneEuroFilterConfig):
        """
        Args:
            config: Configuration object.
        """
        self.mincutoff = np.array(config.mincutoff)
        self.beta = np.array(config.beta)
        self.dcutoff = np.array(config.dcutoff)
        self.rotation_rep = config.rotation_rep
        self.convention = config.convention

        assert self.rotation_rep != RotationType.MATRIX

        self.rot_prev = None
        self.dx_prev = None # Smoothed angular velocity
        self.t_prev = None
        
    def reset(self):
        """ Reset the filter state. """
        self.rot_prev = None
        self.dx_prev = None
        self.t_prev = None
        
    def __call__(self, rotation, t = None):
        """
        Filter a rotation using SLERP.
        
        Args:
            rotation: Rotation in the specified representation
            t: Timestamp (seconds). If None, uses time.time()
            
        Returns:
            Filtered rotation in the specified representation
        """
        if t is None:
            t = time.time()
        
        rotation = rotation_transform(
            rotation, 
            from_rep = self.rotation_rep, 
            to_rep = RotationType.QUATERNION,
            from_convention = self.convention,
            to_convention = None
        )
        
        if self.rot_prev is None:
            # First rotation, no filtering
            self.rot_prev = rotation.copy()
            self.dx_prev = 0.0
            self.t_prev = t
            return rotation
        
        # Compute time delta
        dt = t - self.t_prev
        if dt <= 0:
            return self.rot_prev
        
        # Compute angular velocity from quaternion difference
        # Use dot product to measure quaternion distance
        dot_product = np.clip(np.dot(self.rot_prev, rotation), -1.0, 1.0)
        # Handle quaternion double cover (q and -q represent same rotation)
        if dot_product < 0:
            rotation = -rotation
            dot_product = -dot_product
        
        # Angular distance (in radians)
        # Using 2 * arccos(|q1 . q2|) gives the full angle theta.
        # But for derivative magnitude in 1Euro, we can just use theta.
        theta = 2 * np.arccos(dot_product)
        
        # Angular velocity (rad/s)
        angular_vel = theta / dt
        
        # Filter angular velocity using standard exponential smoothing with dcutoff
        a_d = smoothing_factor(dt, self.dcutoff)
        filtered_angular_vel = exponential_smoothing(a_d, angular_vel, self.dx_prev)
        self.dx_prev = filtered_angular_vel
        
        # Compute adaptive cutoff frequency
        cutoff = self.mincutoff + self.beta * abs(filtered_angular_vel)
        
        # Convert cutoff to interpolation alpha for SLERP
        alpha = smoothing_factor(dt, cutoff)
        
        # Use SLERP to interpolate between previous filtered rotation and current rotation
        filtered_rot = quat_slerp(self.rot_prev, rotation, alpha)
        
        # Update state
        self.rot_prev = filtered_rot.copy()
        self.t_prev = t

        filtered_rot = rotation_transform(
            filtered_rot,
            from_rep = RotationType.QUATERNION,
            to_rep = self.rotation_rep,
            from_convention = None,
            to_convention = self.convention
        )
        
        return filtered_rot


class PoseOneEuroFilter:
    """
    1Euro filter for 6D poses (3D translation + quaternion rotation using SLERP).
    """
    def __init__(self, config: PoseOneEuroFilterConfig):
        """
        Args:
            config: Configuration object.
        """
        self.trans_filter = OneEuroFilter(config = config.trans)
        self.rot_filter = RotationOneEuroFilter(config = config.rot)
        
    def reset(self):
        """ Reset all filters. """
        self.trans_filter.reset()
        self.rot_filter.reset()
        
    def __call__(self, pose, t = None):
        """
        Filter a 7D pose [x, y, z, qw, qx, qy, qz].
        Translation is filtered component-wise, rotation is filtered using SLERP.
        
        Args:
            pose_7d: 7D pose array [translation(3), quaternion(4)]
            t: Timestamp (seconds). If None, uses time.time()
            
        Returns:
            Filtered 7D pose
        """
        if t is None:
            t = time.time()
        
        pose = np.asarray(pose)
        filtered_trans = self.trans_filter(pose[:3], t)
        filtered_rot = self.rot_filter(pose[3:], t)
        
        return np.concatenate([filtered_trans, filtered_rot])
