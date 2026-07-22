"""
Trainer.
"""
from typing import Any, Dict, List, Optional

import os
import time
import h5py
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

from logger import logger
from policy import get_policy, PolicyWrapper
from data_infra.processor import DataProcessor
from data_infra.dataset.vision_dataset import VisionPolicyDataset
from vision_feat_generator.configs import VisionFeatGeneratorConfig
from utils.common import set_seed, to_device, num_params, num_trainable_params


def flush_buffer(
    scene_path: str, 
    buffer: Dict[str, List[torch.Tensor]], 
    file_prefix: str = "vision_feat", 
    vision_feat_key: str = "vision_feat"
) -> None:
    """ Flush the buffer to the file system. """
    lowdim_dir = os.path.join(scene_path, "lowdim")
    os.makedirs(lowdim_dir, exist_ok = True)
    
    for cam_id, data in buffer.items():
        data.sort(key = lambda x: x[0])
        
        timestamps = np.array([x[0] for x in data], dtype = np.int64)
        feats = np.array([x[1] for x in data], dtype = np.float32)
        
        save_path = os.path.join(lowdim_dir, f"{file_prefix}_{cam_id}.h5")
        
        with h5py.File(save_path, 'w') as f:
            f.create_dataset(vision_feat_key, data = feats)
            f.create_dataset('timestamp', data = timestamps)


class VisionFeatGenerator:
    def __init__(
        self,
        config: VisionFeatGeneratorConfig
    ) -> None:
        """ Initialization. """
        self.config = config

        self._set_env_vars()
        set_seed(config.seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self._build_dataset()
        self._build_model()


    def _set_env_vars(self) -> None:
        """ Set environment variables. """
        for key, value in self.config.env_vars.items():
            os.environ[key] = value


    def _build_dataset(self) -> None:
        """ Build dataset. """
        logger.info("Loading dataset ...")
            
        self.dataset = VisionPolicyDataset(self.config.dataset_config)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size = self.config.batch_size,
            num_workers = self.config.num_workers,
            shuffle = False,
            drop_last = False
        )
        self.processor = DataProcessor(self.config.processor_config)


    def _build_model(self) -> None:
        """ Build policy model. """
        logger.info("Loading policy ...")

        self.policy = get_policy(self.config.policy_config).to(self.device)
        n_trainable_parameters = num_trainable_params(self.policy)
        n_parameters = num_params(self.policy)
        logger.info(f"Number of parameters: {n_parameters / 1e6:.2f}M")
        logger.info(f"Number of trainable parameters: {n_trainable_parameters / 1e6:.2f}M")

        ckpt = torch.load(self.config.ckpt_path, map_location = self.device)
        self.policy.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt.keys() else ckpt)
        self.policy.eval()

        self.policy_wrapper = PolicyWrapper(
            self.policy, 
            self.config.policy_wrapper_config
        )
    

    def generate(self) -> None:
        """ Generate vision features. """
        logger.info("Generating vision features ...")
        time.sleep(0.5)

        global_idx = 0
        current_scene = None
        buffer = defaultdict(list)

        for data in tqdm(self.dataloader):
            vision_feats = self._generate_step(data)
            vision_feats = vision_feats.detach().cpu().numpy()

            for b in range(self.config.batch_size):
                sample_idx = self.dataset.all_samples[global_idx + b]
                scene_path = sample_idx.scene_path
                
                if current_scene is not None and scene_path != current_scene:
                    flush_buffer(
                        scene_path = current_scene, 
                        buffer = buffer, 
                        file_prefix = self.config.file_prefix,
                        vision_feat_key = self.config.vision_feat_key
                    )
                    buffer.clear()

                current_scene = scene_path
                cam_id, ts = sample_idx.obs.colors['main']
                buffer[cam_id].append((ts, vision_feats[b]))

            global_idx += self.config.batch_size
        
        if current_scene is not None and len(buffer) > 0:
            flush_buffer(
                scene_path = current_scene, 
                buffer = buffer, 
                file_prefix = self.config.file_prefix,
                vision_feat_key = self.config.vision_feat_key
            )
            buffer.clear()


    def _generate_step(self, data: Dict[str, Any]) -> torch.Tensor:
        """ Generate vision features (a forward step). """
        with torch.inference_mode():
            data = to_device(data, device = self.device)
            obs_dict = self.processor(data["obs"])
            vision_feat = self.policy_wrapper.get_vision_feat(
                obs_dict = obs_dict, 
                batch_size = self.config.batch_size
            )
        return vision_feat
