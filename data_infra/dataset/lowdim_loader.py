import os
import h5py
import numpy as np

from data_infra.dataset.helpers import generate_offsets
from data_infra.dataset.helpers import find_corresponding_timestamp


class LowdimLoader:
    """
    Cache load lowdim files into memory.
    """
    def __init__(self):
        self.lowdim_files = {}
    
    def load(
        self,
        lowdim_path, 
        lowdim_name, 
        field = None
    ):
        file_path = os.path.join(lowdim_path, "{}.h5".format(lowdim_name))
        if file_path not in self.lowdim_files.keys():
            self.lowdim_files[file_path] = {}
            with h5py.File(file_path, "r") as file:
                for key in file.keys():
                    self.lowdim_files[file_path][key] = np.asarray(file[key][:])
        return self.lowdim_files[file_path][field] if field else self.lowdim_files[file_path]
    
    def find_timestamp_idx(
        self,
        lowdim_path, 
        lowdim_name, 
        ts
    ):
        all_ts = self.load(lowdim_path, lowdim_name, "timestamp")
        return find_corresponding_timestamp(ts, all_ts, return_indices = True)


    def generate_indices(
        self,
        lowdim_path,
        lowdim_name,
        ts,
        length,
        freq,
        data_freq,
        direction = 1,
        remove_first = False
    ):
        max_len = len(self.load(lowdim_path, lowdim_name, "timestamp"))
        offsets = generate_offsets(length, freq, data_freq, direction, remove_first)
        if direction == -1:
            offsets = offsets[::-1]
        indices = self.find_timestamp_idx(lowdim_path, lowdim_name, ts)
        return np.clip(indices[:, None] + offsets, 0, max_len - 1)
