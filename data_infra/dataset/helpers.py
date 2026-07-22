import os
import torch
import numpy as np


def get_scene_list(data_path):
    return [os.path.join(data_path, x) for x in os.listdir(data_path) if x[:6] == "scene_"]


def get_timestamps(cam_path, default_subpath = 'color'):
    return sorted([
        int(x[:-4]) for x in os.listdir(os.path.join(cam_path, default_subpath))
        if x[-4:] == '.png'
    ])


def find_corresponding_timestamp(ts, target_ts, return_indices = False):
    """
    Find the corresponding timestamps in target_ts for all timestamp in ts.
    """
    ts = np.asanyarray(ts)
    target_ts = np.asanyarray(target_ts)
    temp_idx = np.searchsorted(target_ts, ts)
    idx_left = np.clip(temp_idx - 1, 0, len(target_ts) - 1)
    idx_right = np.clip(temp_idx, 0, len(target_ts) - 1)
    dist_left = np.abs(ts - target_ts[idx_left])
    dist_right = np.abs(ts - target_ts[idx_right])
    final_idxs = np.where(dist_left < dist_right, idx_left, idx_right)
    return final_idxs if return_indices else target_ts[final_idxs]


def generate_offsets(length, freq = 100, data_freq = 1000, direction = 1, remove_first = False):
    assert data_freq % freq == 0
    step_size = data_freq // freq * direction
    offset = int(remove_first) * step_size
    return np.arange(offset, offset + step_size * length, step_size)
    


def convert_to_tensor(sample):
    """
    Convert sample dict to torch tensors recursively.
    """
    if isinstance(sample, dict):
        for key, value in sample.items():
            sample[key] = convert_to_tensor(value)
        return sample
    elif isinstance(sample, list):
        return [convert_to_tensor(item) for item in sample]
    elif isinstance(sample, np.ndarray):
        if not sample.flags.writeable:
            sample = sample.copy()
        return torch.from_numpy(sample)
    elif isinstance(sample, (int, float)):
        return torch.tensor(sample)
    else:
        return sample