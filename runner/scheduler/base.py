"""
Scheduler base.
"""
from typing import Dict

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data_infra.configs.point_mask import *
from data_infra.ops.point_cloud import generate_points
from runner.configs.scheduler import SchedulerBaseConfig
from runner.agent.base import BaseAgent
from runner.configs.agent import AgentObsKeysConfig
from policy import get_policy, PolicyWrapper
from data_infra.processor import DataProcessor

from utils.common import set_seed
from utils.transforms.rotation import RotationType
from utils.transforms.projection import apply_mat_to_pose
from utils.transforms.interpolation import resample_trajectory
from utils.visualization.scene import visualize_2d, visualize_3d


class SchedulerBase:
    def __init__(
        self,
        config: SchedulerBaseConfig,
        agent: BaseAgent,
        agent_obs_keys: AgentObsKeysConfig,
        device: torch.device
    ) -> None:
        """ Initialization. """
        self.config = config
        self.agent = agent
        self.agent_obs_keys = agent_obs_keys
        self.device = device

        self.platform_config = agent.config.platform_config
        self.prefix_config = agent.config.prefix_config

        set_seed(self.config.seed)

        # Policy and its wrapper
        self.policy = get_policy(self.config.policy_config).to(self.device)
        if self.config.policy_config.name != "VirtualPolicy":
                assert os.path.exists(self.config.ckpt_path), "Checkpoint {} does not exist.".format(self.config.ckpt_path)
                checkpoint = torch.load(self.config.ckpt_path, map_location = self.device)
                self.policy.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
                self.policy.eval()
        self.policy_wrapper = PolicyWrapper(self.policy, self.config.policy_wrapper_config)

        # Agent
        self.agent.ready()
        
        # Data processor
        self.processor = DataProcessor(self.config.processor_config)

        self.last_inference = None
    
    
    def model_inference(self) -> Dict[str, np.ndarray]:
        """ Model inference. """
        with torch.inference_mode():
            # create observation
            obs_raw, time_obs = self.agent.get_obs(self.agent_obs_keys) 
            obs = self.agent.convert_obs(obs_raw, self.device)

            # model inference
            obs_dict = self.processor(obs, enable_aug = False, process_type = "forward")
            action_dict = self.policy_wrapper(obs_dict, action_dict = None, batch_size = 1)
            action = self.processor(obs, action_dict, process_type = "backward")

            # convert back to agent action
            action_raw = self.agent.convert_action(action)
            action_raw = self.agent.to_agent(action_raw)
        
        # resample action to match sync frequency
        synced_action = {}
        aux_action = {}
        for key in action_raw.keys():
            if key in self.config.sync_action_config.keys():
                config = self.config.sync_action_config[key]
                synced_action[key] = resample_trajectory(
                    data = action_raw[key], 
                    source_freq = config.source_frequency, 
                    source_length = config.source_length,
                    target_freq = self.config.sync_frequency,
                    target_length = self.config.sync_horizon,
                    sampling_method = config.interpolation,
                    rotation_rep = config.rotation_rep,
                    convention = config.convention
                )
            else:
                aux_action[key] = action_raw[key]
        
        # save last inference [raw observation, synced action, aux action, latency, obs time]
        self.last_inference = (obs_raw, synced_action, aux_action, time.time() - time_obs, time_obs)
        return synced_action, aux_action


    def visualize(self) -> None:
        """ Visualize. """
        if self.config.visualization is None:
            return
        getattr(self, f"_visualize_{self.config.visualization.mode}")()


    def _visualize_2d(self) -> None:
        """ Visualize in 2D. """
        obs, action = self.last_inference[:2]
        vis_cfg = self.config.visualization

        # Images
        images = []
        for cam_name in vis_cfg.camera_names:
            images.append((cam_name, obs[f"{self.prefix_config.color}/{cam_name}"]))

        # Poses
        poses = []
        for key in vis_cfg.obs_pose_keys:
            poses.append((f"obs/{key}", obs[key]))
        for key in vis_cfg.action_pose_keys:
            poses.append((f"action/{key}", action[key]))

        # Force/Torque
        ft = []
        for key in vis_cfg.obs_ft_keys:
            ft.append((f"obs/{key}", obs[key]))
        for key in vis_cfg.action_ft_keys:
            ft.append((f"action/{key}", action[key]))

        # Visualization
        visualize_2d(images = images, poses = poses, ft = ft)


    def _visualize_3d(self) -> None:
        """ Visualize in 3D. """
        import open3d as o3d
        obs, action = self.last_inference[:2]
        vis_cfg = self.config.visualization
        
        # Point Clouds
        points_list = []
        colors_list = []
        for cam_id in vis_cfg.camera_names:
            depth_key = f"{self.prefix_config.depth}/{cam_id}"
            color_key = f"{self.prefix_config.color}/{cam_id}"
            intrinsic_key = f"{self.prefix_config.intrinsic}/{cam_id}"
            extrinsic_key = f"{self.prefix_config.extrinsic}/{cam_id}"      
            assert depth_key in obs, f"Depth key {depth_key} not found in observation."
            
            depth = obs[depth_key]
            rgb = obs[color_key] if color_key in obs else np.ones((*depth.shape, 3), dtype = np.float32) * 0.5
            intrinsic = obs[intrinsic_key]
            extrinsic = obs[extrinsic_key]
            
            pcd_data = generate_points(
                depth = torch.from_numpy(depth).float(), 
                intrinsic = torch.from_numpy(intrinsic).float(), 
                extrinsic = torch.from_numpy(extrinsic).float(),
                color = torch.from_numpy(rgb).float(),
                frame = "world",
                fill_hole = False,
                flatten = True,
                world_mask_config = vis_cfg.world_mask_config
            )
            points_list.append(pcd_data[:, :3].numpy())
            colors_list.append(pcd_data[:, 3:].numpy())

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(points_list, axis = 0))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors_list, axis = 0))
        
        # Poses
        poses = []
        for key in vis_cfg.obs_pose_keys:
            device_type, device_name, device_key = key.split('/')
            if device_type == "robot":
                T_world_base = obs[f"{self.prefix_config.robot}/{device_name}"]
                value = apply_mat_to_pose(obs[key], mat = T_world_base, rotation_rep = RotationType.QUATERNION)
            else:
                value = obs[key]
            poses.append((f"obs/{key}", value))
        
        for key in vis_cfg.action_pose_keys:
            device_type, device_name, device_key = key.split('/')
            if device_type == "robot":
                T_world_base = obs[f"{self.prefix_config.robot}/{device_name}"]
                value = apply_mat_to_pose(action[key], mat = T_world_base, rotation_rep = RotationType.QUATERNION)
            else:
                value = action[key]
            poses.append((f"action/{key}", value))
            
        # Visualization
        visualize_3d(pcd = pcd, poses = poses)
    

    def start(self) -> None:
        """ Start. """
        raise NotImplementedError


    def stop(self) -> None:
        """ Stop. """
        raise NotImplementedError
