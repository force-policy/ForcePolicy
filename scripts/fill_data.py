"""
Script to fill lowdim data to obtain 1000hz data.

Authors: Hongjie Fang.

Usage:
    python -m scripts.fill_data
           --data_path data/test
           --fields_info tcp_pose:lowdim:tcp_pose:float32:pose_linear
                         force_torque:lowdim:force_torque:float32:linear
           --save_name lowdim_filled
           --pedal_name lowdim
           --pedal_field pedal_0

Example:
    python -m scripts.fill_data --data_path data/cable4/ --fields_info tcp_pose_062703:lowdim:tcp_pose_062703:float32:pose_linear force_torque_062703:lowdim:force_torque_062703:float32:linear ee_command_062703:lowdim:ee_command_062703:float32:linear ee_state_062703:lowdim:ee_state_062703:float32:linear tcp_vel_062703:lowdim:tcp_vel_062703:float32:linear --save_name lowdim_filled --pedal_name lowdim --pedal_field pedal_0
    
    python -m scripts.fill_data --data_path data/cable5/ --fields_info tcp_pose_062046:lowdim:tcp_pose_062046:float32:pose_linear force_torque_062046:lowdim:force_torque_062046:float32:linear ee_command_062046:lowdim:ee_command_062046:float32:linear ee_state_062046:lowdim:ee_state_062046:float32:linear tcp_vel_062046:lowdim:tcp_vel_062046:float32:linear --save_name lowdim_filled --pedal_name lowdim --pedal_field pedal_0
"""

import os
import h5py
import copy
import argparse
import numpy as np

from tqdm import tqdm
from easydict import EasyDict as edict
from utils.transforms.interpolation import quat_slerp



def interpolate_value(t0, x0, t1, x1, t, interp_type):
    """
    Given data point (t0, x0) and (t1, x1), interpolate x at the given t.
    """
    if t == t0:
        return x0
    if t == t1:
        return x1
    alpha = 0.0 if t0 == t1 else (t - t0) / (t1 - t0)
    if interp_type == "nearest":
        return x1 if alpha >= 0.5 else x0
    elif interp_type == "linear":
        return (1 - alpha) * x0 + alpha * x1
    elif interp_type == "pose_linear":
        pos0, quat0 = np.asarray(x0[:3], dtype = np.float32), np.asarray(x0[3:], dtype = np.float32)
        pos1, quat1 = np.asarray(x1[:3], dtype = np.float32), np.asarray(x1[3:], dtype = np.float32)
        pos = (1 - alpha) * pos0 + alpha * pos1
        quat = quat_slerp(quat0, quat1, alpha)
        return np.concatenate([pos, quat])
    else:
        raise ValueError(f"Unsupported interp_type: {interp_type}")


def seq_fill(
    src_values,
    src_timestamps,
    timestamps,
    dtype,
    interp_type
):
    """
    Fill the sequence given source (value, timestamp) pairs. 
    Require that src_timestamps and timestamps are ordered by time.
    """
    dtype = getattr(np, dtype)

    src_idx = 0
    values = []
    for t in timestamps:
        while src_idx < len(src_timestamps) and src_timestamps[src_idx] < t:
            src_idx += 1
        if src_idx == 0:
            values.append(src_values[src_idx])
        elif src_idx == len(src_timestamps):
            values.append(src_values[src_idx - 1])
        else:
            interp_value = interpolate_value(
                t0 = src_timestamps[src_idx - 1],
                x0 = src_values[src_idx - 1],
                t1 = src_timestamps[src_idx],
                x1 = src_values[src_idx],
                t = t,
                interp_type = interp_type
            )
            values.append(interp_value)
            
    return np.stack(values).astype(dtype)


def interpolate_h5_ms(
    fields_info,
    save_path,
    start_ts = None,
    end_ts = None,
    pedal_path = None,
    pedal_field = None
):
    # calculate valid timestamps (excluding pedal)
    new_timestamps = np.arange(start_ts, end_ts + 1, 1, dtype = np.int64)
    if pedal_path is not None and pedal_field is not None:
        with h5py.File(pedal_path, "r") as pedal_file:
            pedal_values = seq_fill(
                src_timestamps = pedal_file["timestamp"][:],
                src_values = pedal_file[pedal_field][:],
                timestamps = new_timestamps,
                dtype = "float32",
                interp_type = "nearest"
            )
        pedal_values = pedal_values.reshape(-1)
        new_timestamps = new_timestamps[pedal_values == 0]

    # generated new interpolated files
    fields_info = edict(fields_info)
    file_cache = {}
    def cache_load(file_name):
        if file_name not in file_cache.keys():
            file_cache[file_name] = h5py.File(file_name, "r")
        return file_cache[file_name]

    interp_file = h5py.File(save_path, "w")
    interp_file.create_dataset("timestamp", data = new_timestamps, dtype = "int64")

    for field, info in fields_info.items():
        lowdim_file = cache_load(info.file)
        field_values = seq_fill(
            src_timestamps = lowdim_file["timestamp"][:],
            src_values = lowdim_file[info.field][:],
            timestamps = new_timestamps,
            dtype = info.dtype,
            interp_type = info.interp_type
        )
        interp_file.create_dataset(field, data = field_values, dtype = field_values.dtype)
        
    # close files
    interp_file.close()
    for key, file in file_cache.items():
        file.close()


def parse_fields_info(fields_info_list):
    """
    Parses a list of field info arguments into a dictionary.
    Each element should be a string: field=file,field, dtype, interp_type
    """
    result = {}
    for item in fields_info_list:
        # Format: field:name:field:dtype:interp_type
        parts = item.split(':')
        if len(parts) != 5:
            raise ValueError(f"Invalid field info format: {item}")
        field_name, name, field, dtype, interp_type = parts
        result[field_name] = {
            'file': name,
            'field': field,
            'dtype': dtype,
            'interp_type': interp_type
        }
    return result


def find_ts(scene_path):
    min_ts, max_ts = np.inf, -np.inf
    for cam_dir in sorted(os.listdir(scene_path)):
        if cam_dir[:4] != "cam_":
            continue
        color_list = sorted(os.listdir(os.path.join(scene_path, cam_dir, "color")))
        min_ts = min(min_ts, int(color_list[0][:-4]))
        max_ts = max(max_ts, int(color_list[-1][:-4]))
    return min_ts, max_ts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Interpolate HDF5 file fields with timestamp selection.')
    parser.add_argument('--data_path', required = True, help = 'data path')
    parser.add_argument('--fields_info', nargs = '+', required = True, help = 'List of field info in the format field:name:field:dtype:interp_type')
    parser.add_argument('--save_name', required = True, help = 'Output HDF5 file name')
    parser.add_argument('--pedal_name', help = 'Pedal HDF5 file name (optional)')
    parser.add_argument('--pedal_field', help = 'Pedal field (optional)')

    args = parser.parse_args()
    fields_info = parse_fields_info(args.fields_info)
    
    scene_list = sorted(os.listdir(args.data_path))
    for scene_name in tqdm(scene_list):
        scene_path = os.path.join(args.data_path, scene_name)
        lowdim_path = os.path.join(scene_path, "lowdim")
        start_ts, end_ts = find_ts(scene_path)
        
        finfo = copy.deepcopy(fields_info)
        for field_name in fields_info.keys():
            finfo[field_name]['file'] = os.path.join(lowdim_path, "{}.h5".format(finfo[field_name]['file']))

        interpolate_h5_ms(
            fields_info = finfo,
            save_path = os.path.join(lowdim_path, "{}.h5".format(args.save_name)),
            start_ts = start_ts,
            end_ts = end_ts,
            pedal_path = os.path.join(lowdim_path, "{}.h5".format(args.pedal_name)),
            pedal_field = args.pedal_field
        )
    