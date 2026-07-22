from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.visualization.asset import create_sphere, create_arrow


DEFAULT_PALETTE = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]]


def visualize_2d(
    images: List[Tuple[str, np.ndarray]], 
    poses: List[Tuple[str, np.ndarray]] = [],
    ft: List[Tuple[str, np.ndarray]] = []
) -> None:
    """
    Visualize data in 2D using Matplotlib.
    
    Args:
        images: List of (name, image_data) tuples.
        poses: Unified list of (label, pose_data) tuples.
        ft: Unified list of (label, ft_data) tuples.
    """
    num_cams = len(images)
    has_poses = len(poses) > 0
    has_ft = len(ft) > 0
    
    total_plots = num_cams + int(has_poses) # + int(has_ft) # TODO
    if total_plots == 0: return

    fig = plt.figure(figsize = (5 * total_plots, 5))
    plot_idx = 1
    
    # Poses
    if has_poses:
        ax_3d = fig.add_subplot(1, total_plots, plot_idx, projection = '3d')
        plot_idx += 1
  
        color_counter = 0
        for label, val in poses:
            data = val
            if data.ndim == 1:
                data = data[None, :]
            ax_3d.plot(
                data[:, 0], data[:, 1], data[:, 2], 
                label = label, 
                c = np.array(DEFAULT_PALETTE[color_counter % len(DEFAULT_PALETTE)]), 
                marker = 'o' if label[:4] == "obs/" else '^'
            )
            color_counter += 1

        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.legend()
        ax_3d.set_title("Poses")
    
    # FT
    # TODO: visualize f/t

    # Cameras
    for (name, img) in images:
        ax_img = fig.add_subplot(1, total_plots, plot_idx)
        plot_idx += 1
        img = np.transpose(img, (1, 2, 0))
        ax_img.imshow(img)
        ax_img.set_title(f"Image: {name}")

    plt.show()


def visualize_3d(
    pcd: o3d.geometry.PointCloud,
    poses: List[Tuple[str, np.ndarray]]
):     
    """
    Visualize the actions in a given scene.
    
    Args:
        pcd: Point cloud to visualize.
        poses: List of (label, pose_data) tuples.
    """
    geometries = [pcd]

    for i, (label, val) in enumerate(poses):
        data = val
        if data.ndim == 1:
            data = data[None, :]
        for k in range(len(data)):
            pos = data[k, :3]
            sphere = create_sphere(pos, radius = 0.01, color = DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)])
            geometries.append(sphere)

    o3d.visualization.draw_geometries(geometries)
