"""
Visualize Interaction Frames in 3D Point Cloud.

Interactive frame-by-frame visualization using Open3D.
Controls:
    N / Right Arrow: Next frame
    P / Left Arrow: Previous frame
    T: Toggle TCP visualization
    F: Toggle fix translation (frame xyz matches TCP pose)
    V: Toggle velocity display (linear and angular velocity arrows)
    Q / Escape: Quit
Example:
python -m scripts.visualize.vis_label_3d --cam_dir test_data/bottle/scene_0001/cam_104122060902  --tcp_file test_data/bottle/scene_0001/lowdim/lowdim.h5 --labeled_file test_data/bottle/scene_0001/lowdim/lowdim_labeled.h5
"""

import argparse
import os
import h5py
import json
import numpy as np
from pathlib import Path

from adaptor.visualization.vis_3d import InteractionFrameVisualizer
from utils.transforms.pose import xyz_rot_to_mat
from utils.transforms.rotation import RotationType


def get_timestamp_from_filename(filename):
    return int(os.path.splitext(filename)[0])


def load_intrinsic(intrinsic_file):
    """Load intrinsic matrix from npy file, handling various formats."""
    intrinsic_data = np.load(intrinsic_file, allow_pickle=True)
    
    if intrinsic_data.ndim == 0:
        intrinsic_data = intrinsic_data.item()
        if isinstance(intrinsic_data, dict):
            if 'intrinsic' in intrinsic_data:
                intrinsic_matrix = np.array(intrinsic_data['intrinsic'])
            elif 'K' in intrinsic_data:
                intrinsic_matrix = np.array(intrinsic_data['K'])
            else:
                # Format: {serial_number: K_matrix}
                first_key = list(intrinsic_data.keys())[0]
                intrinsic_matrix = np.array(intrinsic_data[first_key])
                print(f"Loaded intrinsic for camera serial: {first_key}")
        else:
            intrinsic_matrix = np.array(intrinsic_data)
    else:
        intrinsic_matrix = intrinsic_data
    
    return intrinsic_matrix.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Visualize Interaction Frames in 3D Point Cloud")
    parser.add_argument("--cam_dir", type=str, required=True, help="Camera directory with color/ and depth/")
    parser.add_argument("--tcp_file", type=str, required=True, help="HDF5 file with TCP poses")
    parser.add_argument("--labeled_file", type=str, required=True, help="HDF5 file with labeled frames")
    parser.add_argument("--start_idx", type=int, default=0, help="Start image index")
    parser.add_argument("--intrinsic", type=str, default=None, help="Path to intrinsic params (.npy)")
    parser.add_argument("--extrinsic", type=str, default=None, help="Path to extrinsic params (.json)")
    args = parser.parse_args()

    cam_dir = Path(args.cam_dir)
    color_dir = cam_dir / "color"
    depth_dir = cam_dir / "depth"
    
    # 1. Load Calibration
    # Try multiple possible calibration file names
    calib_candidates = ["result.json", "calib.json", "extrinsic.json"]
    calib_file = None
    if args.extrinsic:
        calib_file = Path(args.extrinsic)
    else:
        for name in calib_candidates:
            candidate = cam_dir / name
            if candidate.exists():
                calib_file = candidate
                break
    
    if args.intrinsic:
        intrinsic_file = Path(args.intrinsic)
    else:
        intrinsic_file = cam_dir / "intrinsic.npy"
    
    if not intrinsic_file.exists():
        print(f"Error: {intrinsic_file} not found.")
        return
    
    intrinsic_matrix = load_intrinsic(intrinsic_file)
    print(f"Intrinsic matrix shape: {intrinsic_matrix.shape}")
    
    extrinsic_matrix = None
    if calib_file is not None:
        print(f"Loading extrinsic from: {calib_file}")
        with open(calib_file, 'r') as f:
            calib_data = json.load(f)
            pose_in_link = np.array(calib_data["pose_in_link"])
            # pose_in_link is T_world_cam (camera pose in world frame)
            # Format: [x, y, z, qw, qx, qy, qz] (quaternion w-first based on pytorch3d)
            extrinsic_matrix = xyz_rot_to_mat(pose_in_link, rotation_rep=RotationType.QUATERNION)
            if not isinstance(extrinsic_matrix, np.ndarray):
                extrinsic_matrix = extrinsic_matrix.numpy()
            print(f"T_world_cam position: {extrinsic_matrix[:3, 3]}")
    else:
        print("WARNING: No calibration file found! Extrinsic defaults to identity.")
        print(f"  Searched for: {calib_candidates}")

    # 2. Get Images
    color_imgs = sorted([f.name for f in color_dir.glob("*.png")], key=get_timestamp_from_filename)
    
    if len(color_imgs) == 0:
        print(f"No images found in {color_dir}")
        return
    
    print(f"Found {len(color_imgs)} images")

    # 3. Load Data
    with h5py.File(args.labeled_file, 'r') as f:
        keys_labeled = list(f.keys())
        frame_pose_key = next((k for k in keys_labeled if 'frame_pose' in k), None)
        if frame_pose_key:
            frame_poses_in_tcp = f[frame_pose_key][:]
        else:
            print(f"Error: frame_pose not found in {args.labeled_file}")
            return
            
        frame_timestamps = f['timestamp'][:] if 'timestamp' in f else None

    with h5py.File(args.tcp_file, 'r') as f:
        keys = list(f.keys())
        tcp_key = next((k for k in keys if 'tcp_pose' in k), None)
        if tcp_key:
            tcp_poses = f[tcp_key][:]
        else:
            print(f"Error: tcp_pose not found in {args.tcp_file}")
            return
        
        # Extract suffix from tcp_key (e.g., "tcp_pose_062703" -> "062703")
        # Handle both "tcp_pose_xxx" and "tcp_pose" formats
        if tcp_key.startswith('tcp_pose_'):
            suffix = tcp_key[len('tcp_pose_'):]
            tcp_vel_key = f'tcp_vel_{suffix}'
            force_torque_key = f'force_torque_{suffix}'
        else:
            # Try exact match first
            tcp_vel_key = 'tcp_vel'
            force_torque_key = 'force_torque'
        
        tcp_vels = None
        force_torques = None
        
        # Try to find tcp_vel with the computed key, or search for any tcp_vel_xxx
        if tcp_vel_key in f:
            tcp_vels = f[tcp_vel_key][:]
            print(f"Loaded tcp_vel data: {tcp_vel_key}, shape: {tcp_vels.shape}")
        else:
            # Try to find any tcp_vel_xxx key
            tcp_vel_candidates = [k for k in keys if 'tcp_vel' in k]
            if tcp_vel_candidates:
                tcp_vel_key = tcp_vel_candidates[0]
                tcp_vels = f[tcp_vel_key][:]
                print(f"Loaded tcp_vel data: {tcp_vel_key}, shape: {tcp_vels.shape}")
            else:
                print(f"Warning: tcp_vel not found in {args.tcp_file}")
        
        # Try to find force_torque with the computed key, or search for any force_torque_xxx
        if force_torque_key in f:
            force_torques = f[force_torque_key][:]
            print(f"Loaded force_torque data: {force_torque_key}, shape: {force_torques.shape}")
        else:
            # Try to find any force_torque_xxx key
            ft_candidates = [k for k in keys if 'force_torque' in k]
            if ft_candidates:
                force_torque_key = ft_candidates[0]
                force_torques = f[force_torque_key][:]
                print(f"Loaded force_torque data: {force_torque_key}, shape: {force_torques.shape}")
            else:
                print(f"Warning: force_torque not found in {args.tcp_file}")
            
        if frame_timestamps is None:
            ts_key = next((k for k in keys if 'timestamp' in k), None)
            if ts_key:
                frame_timestamps = f[ts_key][:]
            else:
                print("Error: timestamp not found.")
                return
        
        tcp_timestamps = frame_timestamps  # Use same timestamps for tcp data

    # 4. Prepare data per image
    vis = InteractionFrameVisualizer(intrinsic_matrix, extrinsic_matrix)
    
    color_paths = []
    depth_paths = []
    frame_data = []
    tcp_data = []  # TCP poses per image (in camera frame)
    tcp_vel_data = []  # TCP velocities per image
    force_torque_data = []  # Force/torque per image
    tcp_pose_vec_data = []  # TCP pose vectors (7D) per image for display
    
    for i, img_name in enumerate(color_imgs):
        ts_curr = get_timestamp_from_filename(img_name)
        
        if i + 1 < len(color_imgs):
            ts_next = get_timestamp_from_filename(color_imgs[i+1])
        else:
            ts_next = ts_curr + 1000000000
        
        # Filter frames in time window
        mask = (frame_timestamps >= ts_curr) & (frame_timestamps < ts_next)
        relevant_indices = np.where(mask)[0]
        
        # Compute camera-frame transforms with their timestamps
        frames_with_ts = []
        tcp_cam = None  # Store the TCP pose closest to current timestamp
        tcp_vel_curr = None  # Store the TCP velocity closest to current timestamp
        force_torque_curr = None  # Store the force_torque closest to current timestamp
        tcp_pose_vec_curr = None  # Store the TCP pose vector for display
        min_tcp_dist = float('inf')
        
        for idx in relevant_indices:
            T_tcp_frame = frame_poses_in_tcp[idx]
            tcp_pose_vec = tcp_poses[idx]
            T_base_tcp = xyz_rot_to_mat(tcp_pose_vec, rotation_rep=RotationType.QUATERNION)
            if not isinstance(T_base_tcp, np.ndarray):
                T_base_tcp = T_base_tcp.numpy()
            
            T_base_frame = T_base_tcp @ T_tcp_frame
            T_cam_frame = vis.transform_frames_to_camera(T_base_frame)
            frames_with_ts.append((frame_timestamps[idx], T_cam_frame))
            
            # Track TCP pose closest to current timestamp
            dist = abs(frame_timestamps[idx] - ts_curr)
            if dist < min_tcp_dist:
                min_tcp_dist = dist
                tcp_cam = vis.transform_frames_to_camera(T_base_tcp)
                tcp_pose_vec_curr = tcp_pose_vec.copy()
                
                # Get corresponding tcp_vel and force_torque
                if tcp_vels is not None and idx < len(tcp_vels):
                    tcp_vel_curr = tcp_vels[idx].copy()
                if force_torques is not None and idx < len(force_torques):
                    force_torque_curr = force_torques[idx].copy()
        
        # Limit maximum frames: select those closest to current image timestamp
        MAX_FRAMES_PER_IMAGE = 100
        if len(frames_with_ts) > MAX_FRAMES_PER_IMAGE:
            # Sort by distance to current timestamp, take closest
            frames_with_ts.sort(key=lambda x: abs(x[0] - ts_curr))
            frames_with_ts = frames_with_ts[:MAX_FRAMES_PER_IMAGE]
        
        frames_cam = [f[1] for f in frames_with_ts]
        
        color_paths.append(str(color_dir / img_name))
        depth_paths.append(str(depth_dir / img_name))
        frame_data.append(frames_cam)
        tcp_data.append(tcp_cam)
        tcp_vel_data.append(tcp_vel_curr)
        force_torque_data.append(force_torque_curr)
        tcp_pose_vec_data.append(tcp_pose_vec_curr)
    
    print(f"Prepared {len(color_paths)} frames for visualization")
    
    # 5. Run visualization
    vis.set_data(
        color_paths, depth_paths, frame_data, 
        tcp_poses=tcp_data,
        tcp_vels=tcp_vel_data,
        force_torques=force_torque_data,
        tcp_pose_vecs=tcp_pose_vec_data
    )
    vis.run(start_idx=args.start_idx)


if __name__ == "__main__":
    main()
