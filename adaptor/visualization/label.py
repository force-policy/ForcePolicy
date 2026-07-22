
import os
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils.transforms.rotation import RotationType
from utils.transforms.pose import xyz_rot_to_mat
from logger import logger

# Configure matplotlib to use TrueType fonts instead of Type 3 for PDF compatibility
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType fonts
matplotlib.rcParams['ps.fonttype'] = 42   # TrueType fonts for PS as well

# New Task Definitions from FrameLabeler behavior
# TASK_SCREW = 0
# TASK_LEVERAGE = 1
# TASK_MOTION = 2
# TASK_ANCHOR = 3
# TASK_FREE = 4

TASK_COLORS = ['#FFD700', '#FF8C00', '#1E90FF', '#DC143C', '#32CD32', 'gray']
TASK_LABELS = ['Rotation', 'Leverage', 'Surface', 'Insertion', 'Free', 'Uncertain']

def recover_task_id(mask):
    # mask: (6,) boolean array corresponding to [Mx, My, Mz, Fx, Fy, Fz]
    # Inferred from labeler_new.py logic:
    # Free: [0,0,0,0,0,0]
    # Screw: [0,0,1,0,0,1]
    # Leverage: [0,0,0,1,1,1]
    # Motion: [0,0,0,0,0,1]
    # Anchor: [1,1,1,1,1,1]
    
    m = mask.astype(int)
    sig = tuple(m)
    
    if sig == (0,0,1,0,0,1): return 0 # Screw
    if sig == (1,1,1,0,0,0): return 1 # Leverage
    if sig == (0,0,1,0,0,0): return 2 # Motion
    if sig == (1,1,1,1,1,1): return 3 # Anchor
    if sig == (0,0,0,0,0,0): return 4 # Free
    return 5 # Unknown

def visualize_labeling(tcp_file, frame_file, start_ts = None, end_ts = None, save_path=None, axis_limits=None, only_contact_phase=True, auto_scale=True, vis_2d_config=None):
    if vis_2d_config is None:
        from adaptor.configs.vis_2d import Vis2DConfig
        vis_2d_config = Vis2DConfig()
    """
    Visualize the labeling results.
    
    Args:
        tcp_file (str): Path to the HDF5 file containing TCP poses.
        frame_file (str): Path to the HDF5 file containing labeled frame data.
        save_path (str, optional): Path to save the visualization image. If None, the plot is shown.
        axis_limits (dict, optional): Custom axis limits for the 3D plot.
            Format: {'x': (min, max), 'y': (min, max), 'z': (min, max)}
            If None, use auto_scale or matplotlib default.
        only_contact_phase (bool): If True, only show interaction frames during contact phases
            (skip Free phase frames in 3D plot). Default is True.
        auto_scale (bool): If True, auto-scale 3D plot based on contact phase interaction frames
            (centered on geometric center, range = 2x longest edge). If False, use matplotlib default.
    """
    # 1. Read data
    with h5py.File(tcp_file, 'r') as f1:
        # Try to find the tcp pose key. Fallback to searching known keys.
        keys = list(f1.keys())
        tcp_key = next((k for k in keys if 'tcp_pose' in k), None)
        if tcp_key is None:
            logger.error(f"Could not find tcp_pose key in {keys}")
            return
        tcp_data = f1[tcp_key][:]
        ts_data = f1['timestamp'][:]

        
    with h5py.File(frame_file, 'r') as f2:
        if 'frame_pose' not in f2:
            logger.error("Invalid frame file: missing 'frame_pose'")
            return
            
        frame_pose = f2['frame_pose'][:] # (N, 4, 4)
        twist_frame = f2['twist_frame'][:] # (N, 6)
        wrench_frame = f2['wrench_frame'][:] # (N, 6)
        mask_frame = f2['mask_frame'][:] # (N, 6)
        ref_force_frame = f2['ref_force_frame'][:] # (N, 6)
    
    if start_ts is not None:
        start_idx = np.argmin(np.abs(ts_data - start_ts))
    else:
        start_idx = 0
    
    if end_ts is not None:
        end_idx = np.argmin(np.abs(ts_data - end_ts))
    else:
        end_idx = len(ts_data)

    tcp_data = tcp_data[start_idx:end_idx]
    ts_data = ts_data[start_idx:end_idx]
    frame_pose = frame_pose[start_idx:end_idx]
    twist_frame = twist_frame[start_idx:end_idx]
    wrench_frame = wrench_frame[start_idx:end_idx]
    mask_frame = mask_frame[start_idx:end_idx]
    ref_force_frame = ref_force_frame[start_idx:end_idx]

    twist_restored = twist_frame
    wrench_restored = wrench_frame
    mask_restored = mask_frame
    ref_restored = ref_force_frame
    
    N = len(twist_restored)
    task_ids = np.array([recover_task_id(mask_restored[i]) for i in range(N)])

    # 2. Layout with configured font sizes
    plt.rcParams.update(vis_2d_config.rc_params)
    
    fig = plt.figure(figsize=vis_2d_config.figsize)
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1])
    
    # 3D Plot (Left Column)
    ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
    
    # Plotting Columns: Rotational (Middle), Translational (Right)
    # Rows: X, Y, Z
    axes_rot = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    axes_lin = [fig.add_subplot(gs[i, 2]) for i in range(3)]
    
    axis_labels = ['X', 'Y', 'Z']
    time = np.arange(N)

    def add_id_background(ax, ids):
        changes = np.where(ids[:-1] != ids[1:])[0] + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(ids)]))
        values = ids[starts]
        for s, e, v in zip(starts, ends, values):
            c = TASK_COLORS[v]
            ax.axvspan(s, e, color=c, alpha=vis_2d_config.bg_alpha, linewidth=vis_2d_config.bg_linewidth)

    def plot_data(ax, vel_data, force_data, ref_data, mask_data, ids, title, lab_vel, lab_force):
        add_id_background(ax, ids)
        
        # Velocity (Left Axis)
        color1 = vis_2d_config.vel_color
        lns1 = ax.plot(time, vel_data, color=color1, label=lab_vel, linewidth=vis_2d_config.vel_linewidth, linestyle=vis_2d_config.vel_linestyle)
        ax.set_ylabel(lab_vel, color=color1)
        ax.tick_params(axis='y', labelcolor=color1)
        ax.grid(True, alpha=0.3)
        
        # Force (Right Axis)
        ax2 = ax.twinx()
        color2 = vis_2d_config.force_color
        lns2 = ax2.plot(time, force_data, color=color2, label=lab_force, linewidth=vis_2d_config.force_linewidth, linestyle=vis_2d_config.force_linestyle)
        
        # Ref Force (Target)
        active_mask = mask_data > 0
        lns3 = []
        if np.any(active_mask):
             lns3 = ax2.plot(time, ref_data, color=vis_2d_config.ref_color, label='Ref', linewidth=vis_2d_config.ref_linewidth, linestyle=vis_2d_config.ref_linestyle, alpha=vis_2d_config.ref_alpha)

        ax2.set_ylabel(lab_force, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax.set_title(title, pad=vis_2d_config.title_pad, fontsize=vis_2d_config.title_fontsize)
        
        # Legend (Optional per subplot, or unified at top)
        # lns = lns1 + lns2 + lns3
        # labs = [l.get_label() for l in lns]
        # ax.legend(lns, labs, loc=0, fontsize='x-small')

    # Plot Rotational (w, m)
    for i in range(3):
        plot_data(
            axes_rot[i],
            twist_restored[:, i],  
            wrench_restored[:, i], 
            ref_restored[:, i],  
            mask_restored[:, i],  
            task_ids,
            f"Translational {axis_labels[i]}",
            "LinVel (m/s)", "Force (N)"
        )

    for i in range(3):
        plot_data(
            axes_lin[i],
            twist_restored[:, 3+i], 
            wrench_restored[:, 3+i], 
            ref_restored[:, 3+i],  
            mask_restored[:, 3+i],   
            task_ids,
            f"Rotational {axis_labels[i]}",
            "AngVel (r/s)", "Moment (Nm)"
        )

    # Legend - exclude Leverage (index=1) and Uncertain (index=5)
    # Place legend centered above the 3D plot title
    visible_indices = [0, 2, 3, 4]  # Rotation, Surface, Insertion, Free
    legend_elements = [Patch(facecolor=TASK_COLORS[i], label=TASK_LABELS[i], alpha=0.3) for i in visible_indices]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.17, 0.98), ncol=4, title="Task Types", framealpha=0.9)

    # 3D Plot
    pos = tcp_data[..., :3]
    T_base_tcp = xyz_rot_to_mat(tcp_data, rotation_rep = RotationType.QUATERNION)
    T_base_frame = T_base_tcp @ frame_pose
    
    # Transform to Base: T_base_frame = T_base_tcp @ T_tcp_frame
    # p_base_frame = p_base_tcp + R_base_tcp @ p_tcp_frame
    R_base_tcp = T_base_tcp[..., :3, :3]
    R_base_frame = T_base_frame[..., :3, :3]
    p_base_frame = T_base_frame[..., :3, 3]
    # p_base_frame = pos 

    ax = ax_3d
    ds = 1
    # Only plot TCP trajectory (black line), no interaction frame trajectory
    ax.plot(pos[::ds, 0], pos[::ds, 1], pos[::ds, 2], 'k-', alpha=0.7, linewidth=1.5)

    def plot_frame(ax, R_mat, p_vec, scale=vis_2d_config.plot_3d_arrow_scale, colors=None):
        if colors is None:
            colors = ['r', 'g', 'b']
        for k in range(3):
            d = R_mat[:, k] * scale
            ax.quiver(p_vec[0], p_vec[1], p_vec[2], d[0], d[1], d[2], color=colors[k], arrow_length_ratio=0.1)

    step = 200
    # Plot Interaction frames (red, green, blue arrows)
    for i in range(0, N, step):
        # Skip Free phase frames if only_contact_phase is True
        if only_contact_phase and task_ids[i] == 4:  # 4 = Free
            continue
        plot_frame(ax, R_base_frame[i], p_base_frame[i], scale=vis_2d_config.plot_3d_arrow_scale)
        
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title("Trajectory & Interaction Frames")
    ax.set_box_aspect((1, 1, 1))
    
    # Set view angle: elev=elevation from XY plane, azim=azimuth in XY plane
    # azim=45: view from an isometric angle instead of directly along -Y axis
    ax.view_init(elev=15, azim=45)
    
    # Set axis limits: use custom if provided, otherwise auto-scale or matplotlib default
    if axis_limits is not None:
        if 'x' in axis_limits:
            ax.set_xlim(axis_limits['x'][0], axis_limits['x'][1])
        if 'y' in axis_limits:
            ax.set_ylim(axis_limits['y'][0], axis_limits['y'][1])
        if 'z' in axis_limits:
            ax.set_zlim(axis_limits['z'][0], axis_limits['z'][1])
    elif auto_scale:
        # Auto-scale based on contact phase interaction frames only
        # Use only non-Free phase frames for range calculation
        contact_mask = task_ids != 4  # 4 = Free
        if np.any(contact_mask):
            contact_positions = p_base_frame[contact_mask]
            mid = np.mean(contact_positions, axis=0)  # Geometric center of contact frames
            extent = np.max(contact_positions, axis=0) - np.min(contact_positions, axis=0)
            max_extent = extent.max()  # Longest edge
            half_range = vis_2d_config.plot_3d_range_scale * max_extent
        else:
            # Fallback if no contact phase: use all data
            all_p = np.concatenate([pos, p_base_frame], axis=0)
            mid = np.mean(all_p, axis=0)
            max_extent = (np.max(all_p, axis=0) - np.min(all_p, axis=0)).max()
            half_range = vis_2d_config.plot_3d_range_scale * max_extent
        
        ax.set_xlim(mid[0] - half_range, mid[0] + half_range)
        ax.set_ylim(mid[1] - half_range, mid[1] + half_range)
        ax.set_zlim(mid[2] - half_range, mid[2] + half_range)
    # else: use matplotlib default auto-scale (do nothing)
    
    # Custom legend: TCP Trajectory (black line) + Interaction Frame (RGB arrows symbol)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='k', linewidth=1.5, label='TCP Trajectory'),
        Line2D([0], [0], marker='>', color='w', markerfacecolor='r', markersize=8,
               markeredgecolor='r', label='Interaction Frame', linestyle='None')
    ]
    # Add small colored markers to represent the 3 axes
    ax.plot([], [], 'r>', markersize=6, label='')  # dummy for legend
    ax.plot([], [], 'g>', markersize=6, label='')
    ax.plot([], [], 'b>', markersize=6, label='')
    
    # Create legend with two entries
    from matplotlib.legend_handler import HandlerTuple
    legend_handles = [
        Line2D([0], [0], color='k', linewidth=2, label='TCP Trajectory'),
        (Line2D([0], [0], marker='>', color='r', linestyle='None', markersize=8),
         Line2D([0], [0], marker='>', color='g', linestyle='None', markersize=8),
         Line2D([0], [0], marker='>', color='b', linestyle='None', markersize=8))
    ]
    ax.legend(legend_handles, ['TCP Trajectory', 'Interaction Frame'], 
              handler_map={tuple: HandlerTuple(ndivide=None)}, loc='upper left')

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        # Save as PDF for vector graphics quality
        pdf_path = save_path if save_path.endswith('.pdf') else save_path + '.png'
        plt.savefig(pdf_path, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Visualization saved to {pdf_path}")
    else:
        plt.show()
