from typing import Any, Dict, List, Tuple, Optional

import os
import time
import json
import h5py
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from easydict import EasyDict as edict

from data_infra.configs import DatasetConfig
from data_infra.dataset.helpers import get_scene_list
from data_infra.dataset.helpers import get_timestamps
from data_infra.dataset.helpers import convert_to_tensor
from data_infra.dataset.helpers import find_corresponding_timestamp
from data_infra.dataset.lowdim_loader import LowdimLoader
from utils.transforms.rotation import RotationType
from utils.transforms.pose import xyz_rot_to_mat


# TODO: padding mask

class VisionPolicyDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        dataset_config: DatasetConfig
    ) -> None:
        super(VisionPolicyDataset, self).__init__()

        self.data_path = dataset_config.data_path
        self.obs_cfg = dataset_config.obs
        self.action_cfg = dataset_config.action
        self.robot_poses = dataset_config.robot_poses
        self.prefix_config = dataset_config.prefix_config

        self.lowdim_loader = LowdimLoader()
        self._fetch_all_samples()

        if dataset_config.repeat_dataset:
            self.all_samples = self.all_samples * dataset_config.repeat_dataset
    

    def _get_pivot_cameras(self, scene_path: str) -> List[Tuple[str, str]]:
        """
        Determine the pivot cameras for the given scene.
        If main cameras exist, return ALL of them as pivots.
        If no main cameras exist, return the first available aux camera.
        Returns list of (camera_id, camera_type).
        """
        pivots = []
        # Check all main cameras
        for cam_id in self.obs_cfg.vision.main_cameras:
            if os.path.exists(os.path.join(scene_path, "cam_{}".format(cam_id))):
                pivots.append((cam_id, 'main'))
        
        if pivots != []:
            return pivots
            
        # Fallback to aux cameras if no main cameras found
        for key, cam_id in self.obs_cfg.vision.aux_cameras.items():
            if os.path.exists(os.path.join(scene_path, "cam_{}".format(cam_id))):
                return [(cam_id, key)]
        
        return []

    def _fetch_all_samples(self) -> None:
        self.all_scenes = get_scene_list(self.data_path)
        self.all_samples = []
        
        # Load all samples.
        for scene_path in tqdm(self.all_scenes):
            pivot_cameras = self._get_pivot_cameras(scene_path)
            if pivot_cameras == []:
                continue

            lowdim_dir = os.path.join(scene_path, "lowdim")

            # Load all auxiliary cameras timestamps.
            aux_cam_ts = {}
            for key, cam_id in self.obs_cfg.vision.aux_cameras.items():
                cam_path = os.path.join(scene_path, "cam_{}".format(cam_id))
                if os.path.exists(cam_path):
                    aux_cam_ts[key] = get_timestamps(cam_path)
            
            for (cam_id, cam_name) in pivot_cameras:
                # Process all vision data indices.
                cam_path = os.path.join(scene_path, "cam_{}".format(cam_id))

                cam_ts = get_timestamps(cam_path)
                aux_cam_corresponding_ts = edict({
                    key: find_corresponding_timestamp(cam_ts, aux_cam_ts_list) 
                    for key, aux_cam_ts_list in aux_cam_ts.items()
                })

                # Process all lowdim data indices
                obs_lowdim_corresponding_ts = edict({})
                obs_lowdim_proprio_ref = edict({})
                for key, info in self.obs_cfg.lowdim.items():
                    obs_lowdim_corresponding_ts[key] = self.lowdim_loader.generate_indices(
                        lowdim_path = lowdim_dir, 
                        lowdim_name = info.lowdim_name, 
                        ts = cam_ts,
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
                        ts = cam_ts,
                        length = info.length,
                        freq = info.freq,
                        data_freq = info.data_freq,
                        direction = 1,
                        remove_first = info.remove_first
                    )
                
                # Every sample
                for i, ts in enumerate(cam_ts):
                    sample_idx = edict({})
                    sample_idx.obs = edict({})
                    sample_idx.action = edict({})
                    sample_idx.obs.lowdim = edict({})
                    sample_idx.action.lowdim = edict({})
                    sample_idx.scene_path = scene_path
                    sample_idx.obs.colors = edict({})
                    sample_idx.obs.depths = edict({})
                    sample_idx.obs.extrinsics = edict({})
                    sample_idx.obs.intrinsics = edict({})
                    
                    for key in self.obs_cfg.vision.colors:
                        if key == 'main':
                            sample_idx.obs.colors[key] = (cam_id, ts)
                        else:
                            sample_idx.obs.colors[key] = (self.obs_cfg.vision.aux_cameras[key], aux_cam_ts[key][i])

                    for key in self.obs_cfg.vision.depths:
                        if key == 'main':
                            sample_idx.obs.depths[key] = (cam_id, ts)
                        else:
                            sample_idx.obs.depths[key] = (self.obs_cfg.vision.aux_cameras[key], aux_cam_ts[key][i]) 

                    for key in self.obs_cfg.vision.extrinsics:
                        sample_idx.obs.extrinsics[key] = cam_id if key == 'main' else self.obs_cfg.vision.aux_cameras[key]

                    for key in self.obs_cfg.vision.intrinsics:
                        sample_idx.obs.intrinsics[key] = cam_id if key == 'main' else self.obs_cfg.vision.aux_cameras[key]

                    # [obs/lowdim]
                    for key, info in self.obs_cfg.lowdim.items():
                        sample_idx.obs.lowdim[key] = obs_lowdim_corresponding_ts[key][i]
                    
                    # [action/lowdim]
                    for key, info in self.action_cfg.lowdim.items():
                        sample_idx.action.lowdim[key] = action_lowdim_corresponding_ts[key][i]

                    self.all_samples.append(sample_idx)
    

    def __len__(self) -> int:
        return len(self.all_samples)


    def load_color_image(
        self,
        cam_path: str,
        ts: int
    ) -> np.ndarray:
        img = np.array(Image.open(os.path.join(cam_path, 'color', "{}.png".format(ts))), dtype = np.float32) / 255.0
        img = img.transpose([2, 0, 1]) # (C, H, W)
        return img
    

    def load_depth_image(
        self,
        cam_path: str,
        ts: int,
        depth_scale: float = 1000.0
    ) -> np.ndarray:
        depth = np.array(Image.open(os.path.join(cam_path, 'depth', "{}.png".format(ts))), dtype = np.float32) / depth_scale
        return depth
    
    
    def load_intrinsic(
        self,
        cam_path
    ) -> np.ndarray:
        return np.load(os.path.join(cam_path, "intrinsic.npy")).astype(np.float32)
    

    def load_extrinsic(
        self,
        cam_path
    ) -> np.ndarray:
        with open(os.path.join(cam_path, "extrinsic.json")) as f:
            extrinsic = json.load(f)
        ex = np.array(extrinsic["pose_in_link"], dtype = np.float32)
        return xyz_rot_to_mat(ex, rotation_rep = RotationType.QUATERNION)
    
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        assert 0 <= idx < len(self.all_samples)

        sample_idx = self.all_samples[idx]

        sample = edict({})
        sample.obs = edict({})
        sample.action = edict({})

        # Load robot pose
        for robot_name, robot_pose in self.robot_poses.items():
            sample.obs[f"{self.prefix_config.robot}/{robot_name}"] = robot_pose

        # Load lowdim vision.
        for key in self.obs_cfg.vision.colors:
            cam_id, ts = sample_idx.obs.colors[key]
            sample.obs[f"{self.prefix_config.color}/{key}"] = self.load_color_image(
                cam_path = os.path.join(sample_idx.scene_path, f"cam_{cam_id}"),
                ts = ts
            )

        for key in self.obs_cfg.vision.depths:
            cam_id, ts = sample_idx.obs.depths[key]
            sample.obs[f"{self.prefix_config.depth}/{key}"] = self.load_depth_image(
                cam_path = os.path.join(sample_idx.scene_path, f"cam_{cam_id}"),
                ts = ts,
                depth_scale = 4000.0 if cam_id[0] == 'f' else 1000.0
            )

        for key in self.obs_cfg.vision.extrinsics:
            cam_id = sample_idx.obs.extrinsics[key]
            sample.obs[f"{self.prefix_config.extrinsic}/{key}"] = self.load_extrinsic(
                cam_path = os.path.join(sample_idx.scene_path, f"cam_{cam_id}")
            )

        for key in self.obs_cfg.vision.intrinsics:
            cam_id = sample_idx.obs.intrinsics[key]
            sample.obs[f"{self.prefix_config.intrinsic}/{key}"] = self.load_intrinsic(
                cam_path = os.path.join(sample_idx.scene_path, f"cam_{cam_id}")
            )
        
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