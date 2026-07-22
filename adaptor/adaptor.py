"""
Data Adaptor for Labeling
"""
from typing import List, Optional

import os
import json
import h5py
import numpy as np
from tqdm import tqdm

from logger import logger
from adaptor.configs.adaptor import DataAdaptorConfig
from adaptor.visualization import visualize_labeling
from adaptor.interaction_frame import get_frame_labeler, get_frame_identifier, TaskType, TwistWrenchFrameIdentifier

from utils.transforms.twist import calc_twist
from utils.transforms.pose import xyz_rot_to_mat
from utils.transforms.rotation import RotationType
from utils.transforms.projection import apply_mat_to_twist, apply_mat_to_wrench


class DataAdaptor:
    def __init__(self, config: DataAdaptorConfig):
        self.config = config
        self.frame_identifier = get_frame_identifier(self.config.frame_identifier_config)
        self.frame_labeler = get_frame_labeler(self.config.frame_labeler_config)

    def _identify_frame(self, wrench, twist, pose, specify: Optional[str] = None):
        """
        Call the frame identifier, optionally overriding `specify` for this patch.
        The override is only forwarded to the twist-wrench identifier (the only one that
        supports it); for every other identifier and when specify is None, the call is
        identical to before, so existing behavior is unchanged.
        """
        if specify is not None and isinstance(self.frame_identifier, TwistWrenchFrameIdentifier):
            return self.frame_identifier(wrench = wrench, twist = twist, pose = pose, specify = specify)
        return self.frame_identifier(wrench = wrench, twist = twist, pose = pose)

    def label(
        self, 
        lowdim_file: str, 
        save_lowdim_file: str,
        data_freq: int = 1000,
        tcp_pose_key: str = "tcp_pose",
        tcp_vel_key: str = "tcp_vel",
        wrench_key: str = "force_torque",
        timestamp_key: str = "timestamp",
        tcp_pose_rotation_rep: RotationType = RotationType.QUATERNION,
        tcp_pose_convention: Optional[str] = None,
        vis_file: Optional[str] = None,
        vis_mid_step_file: Optional[str] = None,
        specify_seq: Optional[List[str]] = None
    ):
        with h5py.File(lowdim_file, 'r') as f:
            tcp_pose = f[tcp_pose_key][:]
            tcp_vel = f[tcp_vel_key][:]
            wrench = f[wrench_key][:]
            lowdim_ts = f[timestamp_key][:]
        
        pose = xyz_rot_to_mat(
            tcp_pose, 
            rotation_rep = tcp_pose_rotation_rep,
            convention = tcp_pose_convention
        ) # base -> tcp
        
        if self.config.calc_twist_from_pose:
            twist = calc_twist(pose, RotationType.MATRIX, freq = data_freq)
        else:
            twist = apply_mat_to_twist(tcp_vel, mat = np.linalg.inv(pose), rotation_only = True)
        
        N = len(wrench)
        patch_size = N if self.config.patch_size == 0 else self.config.patch_size

        num_patches = len(range(0, N, patch_size))
        if specify_seq is not None and len(specify_seq) != num_patches:
            raise ValueError(
                f"specify_seq length ({len(specify_seq)}) does not match the number of patches "
                f"({num_patches}) for patch_size={patch_size}, N={N}. Make sure "
                f"classify_power_source.py used the same --patch_size as this adaptor config."
            )

        # ========== PASS 1: Initial classification + confidence ==========
        patch_task_ids = []      # Task type per patch
        patch_confidences = []   # Confidence per patch
        patch_if_frames = []     # Representative frame per patch
        patch_results = []       # Full results per patch
        patch_ranges = []        # (start, end) index for each patch

        for patch_idx, i in enumerate(tqdm(range(0, N, patch_size), desc = "Pass 1: Classification")):
            end_idx = min(i + patch_size, N)
            patch_ranges.append((i, end_idx))
            
            if_frame = self._identify_frame(
                wrench = wrench[i: end_idx],
                twist = twist[i: end_idx],
                pose = pose[i: end_idx],
                specify = specify_seq[patch_idx] if specify_seq is not None else None
            )

            task_id, confidence, unified_poses, wrench_frame, twist_frame, mask_frame, ref_force_frame = self.frame_labeler(
                if_frame = if_frame,
                wrench = wrench[i: end_idx],
                twist = twist[i: end_idx],
                last_task_id = TaskType.FREE_MOTION  # Not used anymore
            )
            
            patch_task_ids.append(task_id)
            patch_confidences.append(confidence)
            patch_if_frames.append(unified_poses[0])  # Store first frame as representative
            patch_results.append((unified_poses, wrench_frame, twist_frame, mask_frame, ref_force_frame))

        # Save intermediate results if requested
        if vis_mid_step_file is not None:
            self._save_intermediate_results(
                patch_results, lowdim_ts, patch_ranges, 
                lowdim_file, vis_mid_step_file
            )

        # ========== PASS 2: Temporal smoothing (absorption) ==========
        if self.config.enable_smoothing:
            smoothed_task_ids = self._smooth_runs(
                patch_task_ids, patch_confidences, patch_if_frames
            )
        else:
            smoothed_task_ids = patch_task_ids

        # ========== PASS 3: Re-generate data for absorbed patches ==========
        # Re-run labeler with force_task_id to regenerate all keys with corrected type
        for patch_idx, (orig_tid, smoothed_tid) in enumerate(zip(patch_task_ids, smoothed_task_ids)):
            if orig_tid != smoothed_tid:
                start_idx, end_idx = patch_ranges[patch_idx]
                
                # Re-run frame identifier
                if_frame = self._identify_frame(
                    wrench = wrench[start_idx: end_idx],
                    twist = twist[start_idx: end_idx],
                    pose = pose[start_idx: end_idx],
                    specify = specify_seq[patch_idx] if specify_seq is not None else None
                )
                
                # Re-run labeler with specified task_id
                _, _, unified_poses, wrench_frame, twist_frame, mask_frame, ref_force_frame = self.frame_labeler(
                    if_frame = if_frame,
                    wrench = wrench[start_idx: end_idx],
                    twist = twist[start_idx: end_idx],
                    last_task_id = smoothed_tid,
                    specified_task_id = smoothed_tid  # Specify the corrected type
                )
                
                patch_results[patch_idx] = (unified_poses, wrench_frame, twist_frame, mask_frame, ref_force_frame)
                # logger.info(f"Patch {patch_idx}: {orig_tid.name} -> {smoothed_tid.name}")
        
        # Concatenate all results
        if_frames = np.concatenate([r[0] for r in patch_results], axis = 0)
        wrench_frames = np.concatenate([r[1] for r in patch_results], axis = 0)
        twist_frames = np.concatenate([r[2] for r in patch_results], axis = 0)
        mask_frames = np.concatenate([r[3] for r in patch_results], axis = 0)
        ref_force_frames = np.concatenate([r[4] for r in patch_results], axis = 0)

        with h5py.File(save_lowdim_file, 'w') as f:
            f.create_dataset('frame_pose', data = if_frames)
            f.create_dataset('twist_frame', data = twist_frames)
            f.create_dataset('wrench_frame', data = wrench_frames)
            f.create_dataset('mask_frame', data = mask_frames)
            f.create_dataset('ref_force_frame', data = ref_force_frames)
            f.create_dataset('timestamp', data = lowdim_ts)
        
        if vis_file is not None:
            visualize_labeling(
                lowdim_file,
                save_lowdim_file,
                save_path = vis_file
            )
    
    def _save_intermediate_results(
        self, 
        patch_results, 
        lowdim_ts, 
        patch_ranges,
        lowdim_file: str,
        vis_mid_step_file: str
    ):
        """Save intermediate Pass 1 results for visualization comparison."""
        import tempfile
        
        # Concatenate results
        if_frames = np.concatenate([r[0] for r in patch_results], axis = 0)
        wrench_frames = np.concatenate([r[1] for r in patch_results], axis = 0)
        twist_frames = np.concatenate([r[2] for r in patch_results], axis = 0)
        mask_frames = np.concatenate([r[3] for r in patch_results], axis = 0)
        ref_force_frames = np.concatenate([r[4] for r in patch_results], axis = 0)
        
        # Save to temp file and visualize
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as tmp:
            tmp_path = tmp.name
        
        with h5py.File(tmp_path, 'w') as f:
            f.create_dataset('frame_pose', data = if_frames)
            f.create_dataset('twist_frame', data = twist_frames)
            f.create_dataset('wrench_frame', data = wrench_frames)
            f.create_dataset('mask_frame', data = mask_frames)
            f.create_dataset('ref_force_frame', data = ref_force_frames)
            f.create_dataset('timestamp', data = lowdim_ts)
        
        visualize_labeling(lowdim_file, tmp_path, save_path = vis_mid_step_file)
        
        # Clean up temp file
        import os
        os.unlink(tmp_path)
    
    def _average_frame_z_axis(self, if_frames: list, start: int, end: int) -> np.ndarray:
        """
        Compute the average Z-axis direction from frames in [start, end).
        Returns a normalized 3D vector representing the average Z-axis.
        """
        z_axes = []
        for i in range(start, end):
            frame = if_frames[i]  # 4x4 transform matrix
            z_axis = frame[:3, 2]  # Extract Z-axis (third column of rotation)
            z_axes.append(z_axis / (np.linalg.norm(z_axis) + 1e-8))
        
        # Average and normalize
        avg_z = np.mean(z_axes, axis=0)
        return avg_z / (np.linalg.norm(avg_z) + 1e-8)
    
    def _z_axis_dot(self, z1: np.ndarray, z2: np.ndarray) -> float:
        """Compute dot product (similarity) between two Z-axis vectors."""
        return float(np.abs(np.dot(z1, z2)))
    
    def _find_runs(self, task_ids, confidences):
        """
        Segment sequence into runs (consecutive patches with same type).
        Returns list of (task_id, start_idx, length, avg_confidence).
        """
        runs = []
        i = 0
        while i < len(task_ids):
            tid = task_ids[i]
            start = i
            conf_sum = confidences[i]
            i += 1
            while i < len(task_ids) and task_ids[i] == tid:
                conf_sum += confidences[i]
                i += 1
            length = i - start
            avg_conf = conf_sum / length
            runs.append((tid, start, length, avg_conf))
        return runs
    
    def _smooth_runs(self, task_ids, confidences, if_frames):
        """
        Apply temporal smoothing via run-based absorption.
        Short, low-confidence runs are absorbed by confident neighbors.
        Iterates until no UNCERTAIN runs remain.
        """
        smoothed = list(task_ids)
        max_iterations = 10  # Safety limit
        
        for iteration in range(max_iterations):
            runs = self._find_runs(smoothed, confidences)
            
            # Check if any UNCERTAIN runs remain
            has_uncertain = any(tid == TaskType.UNCERTAIN for tid, _, _, _ in runs)
            
            if iteration == 0:
                logger.info(f"Smoothing: found {len(runs)} runs")
                for run_idx, (tid, start, length, avg_conf) in enumerate(runs):
                    logger.info(f"  Run {run_idx}: {tid.name}, start={start}, len={length}, conf={avg_conf:.2f}")
            elif has_uncertain:
                logger.info(f"Smoothing iteration {iteration + 1}: {len(runs)} runs, resolving remaining UNCERTAIN")
            
            made_changes = False
            
            for run_idx, (tid, start, length, avg_conf) in enumerate(runs):
                # ========== Compute adjusted confidence with run-level factors ==========
                adjusted_conf = avg_conf
                
                # 1. Length penalty: shorter runs are less reliable
                # Penalty = 1.0 when len >= min_run_length, decreases linearly to 0.5 at len=1
                if length < self.config.min_run_length:
                    length_factor = 0.5 + 0.5 * (length / self.config.min_run_length)
                    adjusted_conf *= length_factor
                
                # 2. Neighbor type consistency: if neighbors have same type, boost confidence
                prev_run = runs[run_idx - 1] if run_idx > 0 else None
                next_run = runs[run_idx + 1] if run_idx < len(runs) - 1 else None
                
                prev_same = prev_run and prev_run[0] == tid
                next_same = next_run and next_run[0] == tid
                
                if prev_same and next_same:
                    # Both neighbors same type: this run is likely correct
                    adjusted_conf = min(1.0, adjusted_conf * 1.2)
                elif prev_same or next_same:
                    # One neighbor same: neutral
                    pass
                else:
                    # Neither neighbor same: this could be noise
                    adjusted_conf *= 0.8
                
                # ========== Check if this run is a candidate for absorption ==========
                is_uncertain = tid == TaskType.UNCERTAIN
                is_short = length < self.config.min_run_length
                is_low_conf = adjusted_conf < self.config.conf_threshold
                
                if not (is_uncertain or (is_short and is_low_conf)):
                    if is_low_conf and not is_short:
                        logger.debug(f"  Run {run_idx} ({tid.name}): low conf ({adjusted_conf:.2f}) but len={length} >= {self.config.min_run_length}")
                    continue  # Keep this run
                
                # Get neighbor info
                prev_run = runs[run_idx - 1] if run_idx > 0 else None
                next_run = runs[run_idx + 1] if run_idx < len(runs) - 1 else None
                
                if prev_run is None and next_run is None:
                    continue  # Only run, can't absorb
                
                # Compute representative frames using windowed averaging
                window = self.config.sim_window_size
                
                # Current run: all patches
                curr_end = start + length
                frame_curr = self._average_frame_z_axis(if_frames, start, curr_end)
                
                # Prev run: last N patches
                if prev_run:
                    prev_start, prev_len = prev_run[1], prev_run[2]
                    prev_end = prev_start + prev_len
                    prev_window_start = max(prev_start, prev_end - window)
                    frame_prev = self._average_frame_z_axis(if_frames, prev_window_start, prev_end)
                else:
                    frame_prev = None
                
                # Next run: first N patches
                if next_run:
                    next_start, next_len = next_run[1], next_run[2]
                    next_window_end = min(next_start + window, next_start + next_len)
                    frame_next = self._average_frame_z_axis(if_frames, next_start, next_window_end)
                else:
                    frame_next = None
                
                # Compute similarities using averaged Z-axes
                sim_prev = self._z_axis_dot(frame_curr, frame_prev) if frame_prev is not None else 0.0
                sim_next = self._z_axis_dot(frame_curr, frame_next) if frame_next is not None else 0.0
                
                # Check if neighbors are similar to each other (noise case)
                if frame_prev is not None and frame_next is not None:
                    sim_neighbors = self._z_axis_dot(frame_prev, frame_next)
                else:
                    sim_neighbors = 0.0
                
                # Decision logic
                absorb_to = None
                
                # Helper: get CURRENT type of neighbor (may have been absorbed already)
                def get_current_type(run_info):
                    if run_info is None:
                        return None
                    # Use smoothed array to get current type (may differ from original)
                    return smoothed[run_info[1]]  # run_info[1] is start index
                
                prev_type = get_current_type(prev_run)
                next_type = get_current_type(next_run)
                
                # Debug: log similarity values for UNCERTAIN runs
                if is_uncertain:
                    logger.info(f"  UNCERTAIN Run {run_idx}: sim_prev={sim_prev:.2f}, sim_next={sim_next:.2f}, sim_neighbors={sim_neighbors:.2f}")
                
                # UNCERTAIN runs: ALWAYS absorb to the more similar neighbor (binary choice)
                # No threshold requirement - they must be resolved to a definite type
                if is_uncertain:
                    if prev_type is None:
                        absorb_to = next_type
                    elif next_type is None:
                        absorb_to = prev_type
                    else:
                        # Pick the more similar neighbor
                        absorb_to = prev_type if sim_prev >= sim_next else next_type
                    logger.info(f"    -> absorbed to {absorb_to.name}")
                
                # Regular runs: use threshold-based logic to protect real boundaries
                elif sim_neighbors > self.config.sim_threshold:
                    # Neighbors are similar: this is likely noise in a continuous segment
                    # Absorb to the more confident neighbor (use original conf from runs)
                    if prev_run and next_run:
                        absorb_to = prev_type if prev_run[3] >= next_run[3] else next_type
                    elif prev_run:
                        absorb_to = prev_type
                    else:
                        absorb_to = next_type
                    logger.info(f"  Short Run {run_idx} ({tid.name}, len={length}, conf={avg_conf:.2f}): "
                               f"sim_neighbors={sim_neighbors:.2f} > {self.config.sim_threshold} -> absorbed to {absorb_to.name}")
                        
                elif sim_prev > self.config.sim_threshold and sim_next < self.config.sim_threshold:
                    # Similar to prev, different from next: belongs to prev segment
                    absorb_to = prev_type
                    logger.info(f"  Short Run {run_idx} ({tid.name}, len={length}, conf={avg_conf:.2f}): "
                               f"sim_prev={sim_prev:.2f} > threshold -> absorbed to {absorb_to.name}")
                    
                elif sim_next > self.config.sim_threshold and sim_prev < self.config.sim_threshold:
                    # Similar to next, different from prev: belongs to next segment  
                    absorb_to = next_type
                    logger.info(f"  Short Run {run_idx} ({tid.name}, len={length}, conf={avg_conf:.2f}): "
                               f"sim_next={sim_next:.2f} > threshold -> absorbed to {absorb_to.name}")
                    
                else:
                    # Different from both neighbors: this might be a real transition
                    # Don't absorb to preserve potential boundary
                    logger.debug(f"Run {run_idx} ({tid}): preserving as potential boundary "
                               f"(sim_prev={sim_prev:.2f}, sim_next={sim_next:.2f})")
                    continue
                
                # Apply absorption
                if absorb_to is not None:
                    for j in range(start, start + length):
                        smoothed[j] = absorb_to
                    made_changes = True
            
            # Check termination: no UNCERTAIN remaining or no changes made
            remaining_uncertain = any(t == TaskType.UNCERTAIN for t in smoothed)
            if not remaining_uncertain or not made_changes:
                if remaining_uncertain:
                    logger.warning(f"Could not resolve all UNCERTAIN runs after {iteration + 1} iterations")
                break
        
        return smoothed
    

    def label_task(
        self,
        task_path: str,
        lowdim_name: str,
        save_lowdim_name: str,
        data_freq: int = 1000,
        tcp_pose_key: str = "tcp_pose",
        tcp_vel_key: str = "tcp_vel",
        wrench_key: str = "force_torque",
        timestamp_key: str = "timestamp",
        tcp_pose_rotation_rep: RotationType = RotationType.QUATERNION,
        tcp_pose_convention: Optional[str] = None,
        vis_name: Optional[str] = None,
        vis_mid_step_name: Optional[str] = None,
        specify_seq_name: Optional[str] = None
    ) -> None: 

        for scene_name in sorted(os.listdir(task_path)):
            logger.info(f"Processing {scene_name} ...")
            lowdim_dir = os.path.join(task_path, scene_name, "lowdim")

            lowdim_path = os.path.join(lowdim_dir, lowdim_name)
            if not os.path.exists(lowdim_path):
                continue
            
            save_lowdim_path = os.path.join(lowdim_dir, save_lowdim_name)
            
            if vis_name is not None:
                vis_file = os.path.join(lowdim_dir, vis_name)
            else:
                vis_file = None
            
            if vis_mid_step_name is not None:
                vis_mid_step_file = os.path.join(lowdim_dir, vis_mid_step_name)
            else:
                vis_mid_step_file = None

            # Optional per-patch `specify` sequence (e.g. from scripts/classify_power_source.py).
            # If not provided or missing, fall back to the single global config `specify`.
            specify_seq = None
            if specify_seq_name is not None:
                specify_seq_path = os.path.join(lowdim_dir, specify_seq_name)
                if os.path.exists(specify_seq_path):
                    with open(specify_seq_path, "r") as sf:
                        specify_seq = json.load(sf)["specify"]
                else:
                    logger.warning(
                        f"{scene_name}: specify sequence '{specify_seq_name}' not found, "
                        f"falling back to config specify='{getattr(self.config.frame_identifier_config, 'specify', 'auto')}'."
                    )

            self.label(
                lowdim_file = lowdim_path,
                save_lowdim_file = save_lowdim_path,
                data_freq = data_freq,
                tcp_pose_key = tcp_pose_key,
                tcp_vel_key = tcp_vel_key,
                wrench_key = wrench_key,
                timestamp_key = timestamp_key,
                tcp_pose_rotation_rep = tcp_pose_rotation_rep,
                tcp_pose_convention = tcp_pose_convention,
                vis_file = vis_file,
                vis_mid_step_file = vis_mid_step_file,
                specify_seq = specify_seq
            )
