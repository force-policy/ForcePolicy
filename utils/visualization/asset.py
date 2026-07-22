import cv2
import numpy as np
import open3d as o3d


def create_sphere(center, radius = 0.02, color = [1, 0, 0]):
    """Create a circle mesh at the given center point."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere


def create_arrow(start_point, direction, length = 0.05, color = [0, 1, 0]):
    """Create an arrow mesh from start_point in the given direction."""
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)
    
    end_point = start_point + direction * length

    shaft = o3d.geometry.TriangleMesh.create_cylinder(radius = 0.0005, height = length)
    
    shaft_center = (start_point + end_point) / 2
    shaft.translate(shaft_center)
    
    z_axis = np.array([0, 0, 1])
    if np.allclose(direction, z_axis):
        rotation_matrix = np.eye(3)
    else:
        rotation_axis = np.cross(z_axis, direction)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        cos_angle = np.dot(z_axis, direction)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                        [rotation_axis[2], 0, -rotation_axis[0]],
                        [-rotation_axis[1], rotation_axis[0], 0]])
        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    shaft.rotate(rotation_matrix, center = shaft_center)
    
    head = o3d.geometry.TriangleMesh.create_cone(radius=0.005, height=length*0.2)
    head.translate(end_point)
    head.rotate(rotation_matrix, center = end_point)
    
    arrow = shaft + head
    arrow.paint_uniform_color(color)
    
    return arrow

