"""
3D Visualization for Interaction Frames using Open3D.
Supports interactive frame-by-frame viewing.
"""

import open3d as o3d
import numpy as np
import json
import os
from utils.transforms.pose import xyz_rot_to_mat
from utils.transforms.rotation import RotationType


class InteractionFrameVisualizer:
    """
    Interactive visualizer for interaction frames on point cloud.
    
    Controls:
        N / Right Arrow: Next frame
        P / Left Arrow: Previous frame
        Q / Escape: Quit
    """
    
    def __init__(self, intrinsic_matrix, extrinsic_matrix=None):
        """
        Args:
            intrinsic_matrix: (3, 3) numpy array
            extrinsic_matrix: (4, 4) numpy array, T_base_cam. 
        """
        self.intrinsic = intrinsic_matrix
        self.extrinsic = extrinsic_matrix if extrinsic_matrix is not None else np.eye(4)
        self.T_cam_base = np.linalg.inv(self.extrinsic)
        
        # Data storage
        self.color_paths = []
        self.depth_paths = []
        self.frame_data = []  # List of lists of (4,4) transforms per image
        self.current_idx = 0
        
        # Open3D objects
        self.vis = None
        self.pcd = None
        self.frame_meshes = []
        
        # Prevent recursive updates
        self._updating = False
        
        # TCP visualization
        self.tcp_poses = None
        self.tcp_mesh = None
        self.show_tcp = True  # Toggle with 'T' key
        
        # Fix translation: if True, frame translation matches TCP pose
        self.fix_translation_p = False  # Toggle with 'F' key
        
        # Velocity visualization
        self.tcp_vels = None  # List of (6,) arrays: [vx, vy, vz, ωx, ωy, ωz]
        self.force_torques = None  # List of (6,) arrays: [fx, fy, fz, τx, τy, τz]
        self.tcp_pose_vecs = None  # List of (7,) arrays: [x, y, z, qw, qx, qy, qz]
        self.display_velocity = False  # Toggle with 'V' key
        self.velocity_arrows = []  # Store velocity arrow meshes
        
    def set_data(self, color_paths, depth_paths, frame_data, tcp_poses=None, 
                 tcp_vels=None, force_torques=None, tcp_pose_vecs=None):
        """
        Set the data for visualization.
        
        Args:
            color_paths: List of color image paths
            depth_paths: List of depth image paths  
            frame_data: List of lists, each inner list contains (4,4) transforms for that frame
            tcp_poses: Optional list of (4,4) TCP transforms per image (in camera frame)
            tcp_vels: Optional list of (6,) arrays: [vx, vy, vz, ωx, ωy, ωz] per image
            force_torques: Optional list of (6,) arrays: [fx, fy, fz, τx, τy, τz] per image
            tcp_pose_vecs: Optional list of (7,) arrays: [x, y, z, qw, qx, qy, qz] per image
        """
        self.color_paths = color_paths
        self.depth_paths = depth_paths
        self.frame_data = frame_data
        self.tcp_poses = tcp_poses
        self.tcp_vels = tcp_vels
        self.force_torques = force_torques
        self.tcp_pose_vecs = tcp_pose_vecs
        self.current_idx = 0
    
    def create_point_cloud(self, color_img_path, depth_img_path):
        """
        Create Open3D PointCloud from RGB and Depth images.
        """
        color = o3d.io.read_image(color_img_path)
        depth = o3d.io.read_image(depth_img_path)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, 
            depth_scale=1000.0, 
            depth_trunc=3.0, 
            convert_rgb_to_intensity=False
        )
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=np.asarray(color).shape[1],
            height=np.asarray(color).shape[0],
            fx=self.intrinsic[0, 0],
            fy=self.intrinsic[1, 1],
            cx=self.intrinsic[0, 2],
            cy=self.intrinsic[1, 2]
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        return pcd

    def create_coordinate_frame(self, T_frame, size=0.05):
        """
        Create a coordinate frame mesh.
        """
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        frame.transform(T_frame)
        return frame
    
    def create_tcp_frame(self, T_frame, size=0.08):
        """
        Create a TCP coordinate frame mesh with distinct color (yellow/magenta).
        Uses larger size and paints it yellow for visibility.
        """
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        frame.transform(T_frame)
        # Paint all vertices yellow for visibility
        frame.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
        return frame
    
    def create_large_arrow(self, start_point, direction, length, color):
        """
        Create a large, visible arrow mesh.
        
        Args:
            start_point: (3,) array, start position
            direction: (3,) array, direction vector (will be normalized)
            length: float, arrow length
            color: (3,) array, RGB color [0-1]
        
        Returns:
            o3d.geometry.TriangleMesh: Arrow mesh
        """
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)
        
        end_point = start_point + direction * length
        
        # Larger shaft radius for visibility
        shaft_radius = 0.003  # Increased from 0.0005
        shaft = o3d.geometry.TriangleMesh.create_cylinder(radius=shaft_radius, height=length)
        
        shaft_center = (start_point + end_point) / 2
        shaft.translate(shaft_center)
        
        # Rotate shaft to align with direction
        z_axis = np.array([0, 0, 1])
        if np.allclose(direction, z_axis):
            rotation_matrix = np.eye(3)
        elif np.allclose(direction, -z_axis):
            rotation_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            rotation_axis = np.cross(z_axis, direction)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            cos_angle = np.dot(z_axis, direction)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                          [rotation_axis[2], 0, -rotation_axis[0]],
                          [-rotation_axis[1], rotation_axis[0], 0]])
            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        shaft.rotate(rotation_matrix, center=shaft_center)
        
        # Larger arrow head for visibility
        head_radius = 0.008  # Increased from 0.005
        head_height = length * 0.25  # Increased from 0.2
        head = o3d.geometry.TriangleMesh.create_cone(radius=head_radius, height=head_height)
        head.translate(end_point)
        head.rotate(rotation_matrix, center=end_point)
        
        arrow = shaft + head
        arrow.paint_uniform_color(color)
        
        return arrow

    def transform_frames_to_camera(self, frames_base):
        """
        Transform frame(s) from Base Frame to Camera Frame.
        """
        return np.matmul(self.T_cam_base, frames_base)

    def update_visualization(self):
        """Update the visualization with current frame data."""
        if self._updating:
            return
        if self.current_idx >= len(self.color_paths):
            return
        
        self._updating = True
        try:
            # Remove old geometries
            if self.pcd is not None:
                self.vis.remove_geometry(self.pcd, reset_bounding_box=False)
            for mesh in self.frame_meshes:
                self.vis.remove_geometry(mesh, reset_bounding_box=False)
            self.frame_meshes.clear()
            if self.tcp_mesh is not None:
                self.vis.remove_geometry(self.tcp_mesh, reset_bounding_box=False)
                self.tcp_mesh = None
            # Remove velocity arrows
            for arrow in self.velocity_arrows:
                self.vis.remove_geometry(arrow, reset_bounding_box=False)
            self.velocity_arrows.clear()
            
            # Create new point cloud
            c_path = self.color_paths[self.current_idx]
            d_path = self.depth_paths[self.current_idx]
            self.pcd = self.create_point_cloud(c_path, d_path)
            
            # Add point cloud
            self.vis.add_geometry(self.pcd, reset_bounding_box=(self.current_idx == 0))
            
            # Add interaction frames
            frames = self.frame_data[self.current_idx]
            # Get TCP pose for fixing translation if enabled
            T_tcp = None
            if self.fix_translation_p and self.tcp_poses is not None and self.tcp_poses[self.current_idx] is not None:
                T_tcp = self.tcp_poses[self.current_idx]
            
            for T in frames:
                # If fix_translation_p is enabled, set frame translation to TCP translation
                if T_tcp is not None:
                    T_fixed = T.copy()
                    T_fixed[:3, 3] = T_tcp[:3, 3]
                    mesh = self.create_coordinate_frame(T_fixed)
                else:
                    mesh = self.create_coordinate_frame(T)
                self.frame_meshes.append(mesh)
                self.vis.add_geometry(mesh, reset_bounding_box=False)
            
            # Add TCP frame if enabled and available
            tcp_info = ""
            if self.show_tcp and self.tcp_poses is not None and self.tcp_poses[self.current_idx] is not None:
                T_tcp_display = self.tcp_poses[self.current_idx]
                self.tcp_mesh = self.create_tcp_frame(T_tcp_display)
                self.vis.add_geometry(self.tcp_mesh, reset_bounding_box=False)
                tcp_info = " [TCP: ON]"
            elif self.tcp_poses is not None:
                tcp_info = " [TCP: OFF]"
            
            fix_info = " [Fix Translation: ON]" if self.fix_translation_p else ""
            
            # Add velocity arrows if enabled
            vel_info = ""
            if self.display_velocity and self.tcp_poses is not None and self.tcp_poses[self.current_idx] is not None:
                T_tcp_display = self.tcp_poses[self.current_idx]
                tcp_pos = T_tcp_display[:3, 3]
                
                # Display linear velocity arrow
                if self.tcp_vels is not None and self.tcp_vels[self.current_idx] is not None:
                    tcp_vel = self.tcp_vels[self.current_idx]
                    linear_vel = tcp_vel[:3]  # [vx, vy, vz]
                    angular_vel = tcp_vel[3:]  # [ωx, ωy, ωz]
                    
                    # Transform linear velocity from world frame to camera frame
                    # Linear velocity is a vector, transform using rotation only
                    R_world_cam = self.T_cam_base[:3, :3]
                    linear_vel_cam = R_world_cam @ linear_vel
                    
                    # Draw linear velocity arrow
                    vel_magnitude = np.linalg.norm(linear_vel)
                    if vel_magnitude > 1e-6:
                        # Scale: 1 m/s -> 0.15 m arrow length (increased from 0.05)
                        arrow_length = max(vel_magnitude * 0.15, 0.02)  # Minimum 2cm for visibility
                        linear_vel_dir = linear_vel_cam / vel_magnitude
                        # Bright cyan color for better visibility
                        vel_arrow = self.create_large_arrow(tcp_pos, linear_vel_dir, length=arrow_length, color=[0, 1, 1])  # Cyan
                        self.velocity_arrows.append(vel_arrow)
                        self.vis.add_geometry(vel_arrow, reset_bounding_box=False)
                    
                    # Draw angular velocity arrow
                    angular_vel_magnitude = np.linalg.norm(angular_vel)
                    if angular_vel_magnitude > 1e-6:
                        # Transform angular velocity (also rotation only)
                        angular_vel_cam = R_world_cam @ angular_vel
                        # Scale: 1 rad/s -> 0.10 m arrow length (increased from 0.03)
                        arrow_length = max(angular_vel_magnitude * 0.10, 0.02)  # Minimum 2cm for visibility
                        angular_vel_dir = angular_vel_cam / angular_vel_magnitude
                        # Bright magenta color for better visibility
                        ang_vel_arrow = self.create_large_arrow(tcp_pos, angular_vel_dir, length=arrow_length, color=[1, 0, 1])  # Magenta
                        self.velocity_arrows.append(ang_vel_arrow)
                        self.vis.add_geometry(ang_vel_arrow, reset_bounding_box=False)
                
                vel_info = " [Velocity: ON]"
            
            # Print information text
            info_lines = []
            if self.tcp_pose_vecs is not None and self.tcp_pose_vecs[self.current_idx] is not None:
                tcp_pose_vec = self.tcp_pose_vecs[self.current_idx]
                info_lines.append(f"TCP Pose: {tcp_pose_vec.tolist()}")
            
            if self.tcp_vels is not None and self.tcp_vels[self.current_idx] is not None:
                tcp_vel = self.tcp_vels[self.current_idx]
                info_lines.append(f"TCP Vel: {tcp_vel.tolist()}")
            
            if self.force_torques is not None and self.force_torques[self.current_idx] is not None:
                ft = self.force_torques[self.current_idx]
                info_lines.append(f"Force/Torque: {ft.tolist()}")
            
            if info_lines:
                print("\n--- Current Frame Info ---")
                for line in info_lines:
                    print(line)
                print("--------------------------\n")
            
            # Update title
            self.vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
            
            print(f"Frame {self.current_idx + 1}/{len(self.color_paths)}, "
                  f"Interaction frames: {len(frames)}{tcp_info}{fix_info}{vel_info}")
            
            self.vis.update_renderer()
        finally:
            self._updating = False
    
    def next_frame(self, vis):
        """Go to next frame."""
        if self._updating:
            return False
        if self.current_idx < len(self.color_paths) - 1:
            self.current_idx += 1
            self.update_visualization()
        return False
    
    def prev_frame(self, vis):
        """Go to previous frame."""
        if self._updating:
            return False
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_visualization()
        return False
    
    def toggle_tcp(self, vis):
        """Toggle TCP visualization."""
        if self._updating:
            return False
        self.show_tcp = not self.show_tcp
        print(f"TCP visualization: {'ON' if self.show_tcp else 'OFF'}")
        self.update_visualization()
        return False
    
    def toggle_fix_translation(self, vis):
        """Toggle fix translation mode (F key)."""
        if self._updating:
            return False
        self.fix_translation_p = not self.fix_translation_p
        print(f"Fix translation: {'ON' if self.fix_translation_p else 'OFF'}")
        self.update_visualization()
        return False
    
    def toggle_velocity(self, vis):
        """Toggle velocity visualization (V key)."""
        if self._updating:
            return False
        self.display_velocity = not self.display_velocity
        print(f"Velocity display: {'ON' if self.display_velocity else 'OFF'}")
        self.update_visualization()
        return False
    
    def run(self, start_idx=0):
        """
        Run the interactive visualization.
        
        Controls:
            N / Right Arrow: Next frame
            P / Left Arrow: Previous frame
            T: Toggle TCP visualization
            F: Toggle fix translation (frame xyz matches TCP pose)
            V: Toggle velocity display (linear and angular velocity arrows)
            Q / Escape: Quit
        """
        if len(self.color_paths) == 0:
            print("No data to visualize.")
            return
            
        self.current_idx = min(start_idx, len(self.color_paths) - 1)
        
        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Interaction Frame Viewer", width=1280, height=720)
        
        # Register key callbacks
        # N and Right Arrow for next
        self.vis.register_key_callback(ord('N'), self.next_frame)
        self.vis.register_key_callback(262, self.next_frame)  # Right arrow
        
        # P and Left Arrow for previous  
        self.vis.register_key_callback(ord('P'), self.prev_frame)
        self.vis.register_key_callback(263, self.prev_frame)  # Left arrow
        
        # T for toggle TCP
        self.vis.register_key_callback(ord('T'), self.toggle_tcp)
        
        # F for toggle fix translation
        self.vis.register_key_callback(ord('F'), self.toggle_fix_translation)
        
        # V for toggle velocity display
        self.vis.register_key_callback(ord('V'), self.toggle_velocity)
        
        # Initial display
        self.update_visualization()
        
        print("\n=== Controls ===")
        print("N / Right Arrow: Next frame")
        print("P / Left Arrow: Previous frame")
        print("T: Toggle TCP visualization")
        print("F: Toggle fix translation (frame xyz matches TCP pose)")
        print("V: Toggle velocity display (linear and angular velocity arrows)")
        print("Q / Escape: Quit")
        print("================\n")
        
        # Run the visualizer
        self.vis.run()
        self.vis.destroy_window()
