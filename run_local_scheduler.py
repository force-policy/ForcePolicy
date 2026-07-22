# LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
import sys
import os
import torch
import numpy as np
import cv2
import h5py
from PIL import Image

from runner.configs.agent import LocalAgentConfig
from runner.configs.scheduler import LocalSchedulerConfig, VisualizationConfig

from runner.agent import LocalAgent
from runner.scheduler.local import LocalScheduler

from configs import get_config


if __name__ == "__main__":
    cam_id = "main"
    robot_name = "main"

    policy_config = get_config("force_policy.vision_feat_wrist_vision_cond_comb_lowdim_cond_gated_with_global_comb_action_force_diffusion_frame_mask_mlp", "policy")
    wrapper_config = get_config("force_policy.vision_feat_wrist_vision_cond_comb_lowdim_cond_gated_with_global_comb_action_force_diffusion_frame_mask_mlp", "wrapper")
    processor_config = get_config("charger.force_policy.wrist", "processor")
    
    colors = {}
    depths = {}
    intrinsics = {}
    extrinsics = {}
    robot_poses = {}
    lowdim = {}
    
    colors[cam_id] = np.array(Image.open("/data/hongjie/data/FORTE_new/charger/scene_0001/cam_104122062823/color/1765865054704.png"), dtype = np.uint8)
    lowdim["tcp_pose"] = np.array([[0,0,0,1,0,0,0]] * 100, dtype=np.float32) #np.load("/data/hongjie/data/FORTE_new/charger/scene_0001/lowdim/test_tcp.npy")
    lowdim["force_torque"] = np.array([[0,0,0,0,0,0]] * 100, dtype=np.float32) # np.load("/data/hongjie/data/FORTE_new/charger/scene_0001/lowdim/test_ft.npy")
    #lowdim["tcp_pose"] = np.load("test_proprio.npy").squeeze(0)
    #print(lowdim["tcp_pose"])
    #lowdim["force_torque"] = np.load("test_ft.npy").squeeze(0)
    lowdim["aux_tcp_pose"] = lowdim["tcp_pose"][-1:, ...]
    with h5py.File("data/charger/scene_0001/lowdim/vision_feat_840412062188.h5", "r") as f:
        lowdim["vision_feat"] = f["vision_feat"][41]
    
    robot_poses[robot_name] = np.eye(4, dtype = np.float32)

    # Checkpoint Path
    ckpt_path = "logs/log_force_policy_vision_feat_wrist_vision_comb_lowdim_gated_comb_action_force_diffusion_frame_mask_mlp_20251221/policy_step_60000_seed_233.ckpt" # TODO
        
    # Create Local Agent Config
    agent_config = LocalAgentConfig(
        common_config = processor_config.common_config,
        colors = colors,
        depths = {},
        intrinsics = {},
        extrinsics = {},
        robot_poses = robot_poses,
        lowdim = lowdim,
        agent_action_providers = {
            "robot/main/tcp_pose": "action_tcp",
            "robot/main/force_frame": "action_frame",
            "robot/main/tcp_wrench": "action_force_torque",
            "robot/main/force_frame_mask": "action_frame_mask"
        }
    )
    
    # Create Agent
    agent = LocalAgent(agent_config)
    
    # Create Scheduler Config
    scheduler_config = LocalSchedulerConfig(
        processor_config = processor_config,
        policy_config = policy_config,
        policy_wrapper_config = wrapper_config,
        ckpt_path = ckpt_path,
        seed = 233,
        camera_name = "main",
        visualization = VisualizationConfig(
            mode = "2d",
            camera_names = ["main"],
            action_pose_keys = ["robot/main/tcp_pose"],
        )
    )
    
    scheduler = LocalScheduler(
        scheduler_config, 
        agent = agent,
        agent_obs_keys = None,
        device = torch.device("cuda")
    )
    action = scheduler.model_inference()
    print(action.keys())
    for key in action.keys():
        print(key, action[key])
    scheduler.visualize()

