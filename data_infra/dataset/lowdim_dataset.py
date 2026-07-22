from typing import Any, Dict, List, Optional

import os
import time
import json
import h5py
import torch
import numpy as np

from tqdm import tqdm
from easydict import EasyDict as edict

from data_infra.configs import DatasetConfig
from data_infra.dataset.helpers import get_scene_list
from data_infra.dataset.helpers import convert_to_tensor
from data_infra.dataset.lowdim_loader import LowdimLoader


class LowdimPolicyDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        dataset_config: DatasetConfig
    ) -> None:
        super(LowdimPolicyDataset, self).__init__()

        assert dataset_config.type == "lowdim"

        self.data_path = dataset_config.data_path
        self.obs_cfg = dataset_config.obs
        self.action_cfg = dataset_config.action
        self.pivot_key = dataset_config.pivot_key
        self.robot_poses = dataset_config.robot_poses
        self.prefix_config = dataset_config.prefix_config

        self.lowdim_loader = LowdimLoader()
        self._fetch_all_samples()
        
        if dataset_config.repeat_dataset:
            self.all_samples = self.all_samples * dataset_config.repeat_dataset
    

    def _fetch_all_samples(self) -> None:
        self.all_scenes = get_scene_list(self.data_path)
        self.all_samples = []
        
        # Load all samples.
        for scene_path in tqdm(self.all_scenes):
            lowdim_dir = os.path.join(scene_path, "lowdim")
            
            pivot_ts = self.lowdim_loader.load(
                lowdim_path = lowdim_dir,
                lowdim_name = self.obs_cfg.lowdim[self.pivot_key].lowdim_name,
                field = "timestamp"
            )
            
            # Process all lowdim data indices
            obs_lowdim_corresponding_ts = edict({})
            for key, info in self.obs_cfg.lowdim.items():
                if key == self.pivot_key:
                    continue
                obs_lowdim_corresponding_ts[key] = self.lowdim_loader.generate_indices(
                    lowdim_path = lowdim_dir, 
                    lowdim_name = info.lowdim_name, 
                    ts = pivot_ts,
                    length = info.length,
                    freq = info.freq,
                    data_freq = info.data_freq,
                    direction = -1,
                    remove_first = info.remove_first
                )
                
            # Process all action indices
            action_lowdim_corresponding_ts = edict({})
            for key, info in self.action_cfg.lowdim.items():
                action_lowdim_corresponding_ts[key] = self.lowdim_loader.generate_indices(
                    lowdim_path = lowdim_dir, 
                    lowdim_name = info.lowdim_name, 
                    ts = pivot_ts,
                    length = info.length,
                    freq = info.freq,
                    data_freq = info.data_freq,
                    direction = 1,
                    remove_first = info.remove_first
                )
                
            # Every sample
            for i, ts in enumerate(pivot_ts):
                sample_idx = edict({})
                sample_idx.obs = edict({})
                sample_idx.action = edict({})
                sample_idx.obs.lowdim = edict({})
                sample_idx.action.lowdim = edict({})
                sample_idx.scene_path = scene_path

                sample_idx.obs.lowdim[self.pivot_key] = i
                
                for key, info in self.obs_cfg.lowdim.items():
                    if key != self.pivot_key:
                        sample_idx.obs.lowdim[key] = obs_lowdim_corresponding_ts[key][i]

                for key, info in self.action_cfg.lowdim.items():
                    sample_idx.action.lowdim[key] = action_lowdim_corresponding_ts[key][i]

                self.all_samples.append(sample_idx)
    

    def __len__(self) -> int:
        return len(self.all_samples)

    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        assert 0 <= idx < len(self.all_samples)

        sample_idx = self.all_samples[idx]

        sample = edict({})
        sample.obs = edict({})
        sample.action = edict({})

        # Load robot pose
        for robot_name, robot_pose in self.robot_poses.items():
            sample.obs[f"{self.prefix_config.robot}/{robot_name}"] = robot_pose
        
        # Load observation
        for key, info in self.obs_cfg.lowdim.items():
            indices = sample_idx.obs.lowdim[key]
            sample.obs[key] = self.lowdim_loader.load(
                lowdim_path = os.path.join(sample_idx.scene_path, "lowdim"), 
                lowdim_name = info.lowdim_name, 
                field = info.field
            )[indices]
        
        # Load action
        for key, info in self.action_cfg.lowdim.items():
            indices = sample_idx.action.lowdim[key]
            sample.action[key] = self.lowdim_loader.load(
                lowdim_path = os.path.join(sample_idx.scene_path, "lowdim"), 
                lowdim_name = info.lowdim_name, 
                field = info.field
            )[indices]
        
        return convert_to_tensor(sample)
