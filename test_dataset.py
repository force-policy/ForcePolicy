import os
import sys
import torch
import argparse
import numpy as np

from data_infra.dataset.vision_dataset import VisionPolicyDataset
from data_infra.processor import DataProcessor
from configs import get_config
from utils.visualization.asset import create_sphere, create_arrow
from data_infra.ops.normalization import unnormalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize(batch, processor, processor_config):
    obs = batch['obs']
    action = batch['action']

    print("Processing batch...")
    # Enable augmentation to see effects, or disable to see raw alignment
    obs_results, action_results = processor(obs, action, enable_aug = False)

    print("=" * 60)
    for key in action_results.keys():
        print(key, action_results[key])
        print("=" * 60)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # 3D Plot for TCP
    ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
    
    # Proprio TCP
    if "proprio_tcp" in obs_results:
        proprio = obs_results["proprio_tcp"]
        # proprio is likely (Batch, ...). We take batch 0.
        # If it's pose (Batch, 3+3/4), we take first 3 for position.
        # unnormalize if needed
        if "proprio_tcp" in processor_config.obs_data_configs:
             cfg = processor_config.obs_data_configs["proprio_tcp"]
             if cfg.norm_config:
                 proprio = unnormalize(proprio, cfg.norm_config)
        
        proprio = proprio[0].detach().cpu().numpy()
        # Assume first 3 are XYZ

        if proprio.ndim == 2:
            # Trajectory
            ax_3d.plot(proprio[:, 0], proprio[:, 1], proprio[:, 2], c='b', marker='o', label='Proprio TCP')
        else:
            ax_3d.scatter(proprio[0], proprio[1], proprio[2], c='b', marker='o', label='Proprio TCP')

        ax_3d.text(proprio[-1, 0], proprio[-1, 1], proprio[-1, 2], 'Start', color='black')

    # Action TCP
    if "action_tcp" in action_results:
        action_tcp = action_results["action_tcp"]
        # unnormalize if needed
        if "action_tcp" in processor_config.action_data_configs:
             cfg = processor_config.action_data_configs["action_tcp"]
             if cfg.norm_config:
                 action_tcp = unnormalize(action_tcp, cfg.norm_config)
        
        action_tcp = action_tcp[0].detach().cpu().numpy() # (Horizon, Dim) or (Dim,)
        if action_tcp.ndim == 2:
             # Trajectory
             ax_3d.plot(action_tcp[:, 0], action_tcp[:, 1], action_tcp[:, 2], c='r', marker='^', label='Action TCP')
        else:
             ax_3d.scatter(action_tcp[0], action_tcp[1], action_tcp[2], c='r', marker='^', label='Action TCP')

    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.legend()
    ax_3d.set_title("TCP Poses")

    # Image Wrist
    ax_img = fig.add_subplot(1, 2, 2)
    if "image_wrist" in obs_results:
        img_wrist = obs_results["image_wrist"]
        if "image_wrist" in processor_config.obs_data_configs:
             cfg = processor_config.obs_data_configs["image_wrist"]
             if cfg.norm_config:
                 img_wrist = unnormalize(img_wrist, cfg.norm_config)
        
        # (Batch, C, H, W) -> (H, W, C)
        img_wrist = img_wrist[0].detach().cpu().permute(1, 2, 0).numpy()
        ax_img.imshow(img_wrist)
        ax_img.set_title("Wrist Image")
    else:
        ax_img.text(0.5, 0.5, "image_wrist not found", ha='center')
        
    plt.show()

    print("\n--- Low-dim Data ---")
    for key, value in obs_results.items():
        print(f"{key}: shape={value.shape if hasattr(value, 'shape') else 'N/A'}")
             
    print("--------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize processed data.")
    parser.add_argument("--dataset", type=str, required=True, help="dataset config name")
    parser.add_argument("--processor", type=str, required=True, help="processor config name")
    args = parser.parse_args()

    print(f"Loading dataset config: {args.dataset}")
    dataset_config = get_config(args.dataset, "dataset")
    
    print(f"Loading processor config: {args.processor}")
    processor_config = get_config(args.processor, "processor")

    print("Initializing Dataset and Processor...")
    dataset = VisionPolicyDataset(dataset_config)
    processor = DataProcessor(processor_config)
    
    print("Fetching a batch...")
    # Use batch size 1 for visualization simplicity
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Generate infinite loop for visualization
    data_iter = iter(dataloader)
    while True:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        visualize(batch, processor, processor_config)
