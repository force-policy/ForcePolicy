"""
Utility helpers shared by scheduling components.
"""

import math
from typing import List
import numpy as np


def to_bool6(flag: bool) -> List[bool]:
    return [bool(flag)] * 6


def same_axis(a: List[bool], b: List[bool]) -> bool:
    return all(bool(x) == bool(y) for x, y in zip(a, b))


def unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Quaternion (w,x,y,z) rotating vector a to vector b (both 3D, unit).
    Minimal rotation. Handle parallel/anti-parallel cases.
    """
    a = unit(a)
    b = unit(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if dot < -1.0 + 1e-8:
        axis = unit(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if abs(np.dot(axis, a)) > 0.9:
            axis = unit(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        axis = unit(np.cross(a, axis))
        return np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float32)
    axis = unit(np.cross(a, b))
    ang = math.acos(dot)
    s = math.sin(ang / 2.0)
    return np.array([math.cos(ang / 2.0), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)


def force_to_frame_quat_tcp(force_tcp_xyz: np.ndarray) -> np.ndarray:
    """
    Build quaternion (w,x,y,z) of control frame expressed in TCP:
    make its z-axis align to force direction.
    """
    f = np.asarray(force_tcp_xyz, dtype=np.float32).reshape(3)
    f_dir = unit(f)
    if np.allclose(f_dir, 0.0):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    z_tcp = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    q = quat_from_two_vectors(z_tcp, f_dir)
    q = q / np.linalg.norm(q)
    return q.astype(np.float32)


def compute_cartesian_velocity_from_poses(pose_prev: np.ndarray, pose_next: np.ndarray, dt: float) -> np.ndarray:
    """
    Approximate 6D cartesian velocity (vx,vy,vz, wx,wy,wz) from two poses.
    pose_prev, pose_next: (7,) [x,y,z,w,x,y,z]
    dt: time interval in seconds
    """
    dt = max(float(dt), 1e-6)
    p0 = np.asarray(pose_prev, dtype=np.float32)
    p1 = np.asarray(pose_next, dtype=np.float32)

    v_lin = (p1[:3] - p0[:3]) / dt

    q0 = p0[3:7]
    q1 = p1[3:7]
    q0 = q0 / max(np.linalg.norm(q0), 1e-9)
    q1 = q1 / max(np.linalg.norm(q1), 1e-9)

    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    cw0, cx0, cy0, cz0 = w0, -x0, -y0, -z0

    rw = w1 * cw0 - (x1 * cx0 + y1 * cy0 + z1 * cz0)
    rx = w1 * cx0 + cw0 * x1 + (y1 * cz0 - z1 * cy0)
    ry = w1 * cy0 + cw0 * y1 + (z1 * cx0 - x1 * cz0)
    rz = w1 * cz0 + cw0 * z1 + (x1 * cy0 - y1 * cx0)

    q_rel = np.array([rw, rx, ry, rz], dtype=np.float32)
    q_rel = q_rel / max(np.linalg.norm(q_rel), 1e-9)

    rw = float(np.clip(q_rel[0], -1.0, 1.0))
    angle = 2.0 * math.acos(rw)
    s = math.sqrt(max(1.0 - rw * rw, 0.0))

    if angle < 1e-6 or s < 1e-6:
        axis = np.zeros(3, dtype=np.float32)
    else:
        axis = q_rel[1:] / s

    v_ang = axis * (angle / dt)

    vel6 = np.zeros(6, dtype=np.float32)
    vel6[:3] = v_lin
    vel6[3:] = v_ang
    return vel6
