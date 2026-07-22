"""
Run evaluation.

Usage:
    python run_eval.py \
           --processor [processor config] \
           --policy [policy config] \
           --wrapper [wrapper config] \
           --agent [agent config] \
           --agent_obs_keys [agent obs keys config name] \
           --agent_class [agent class] \
           --scheduler [scheduler config] \
           --ckpt_path [checkpoint path] \
           --visualize (optional, enable visualization) \
           --logger [log level] (optional, default: info)
"""
import time
import torch
import argparse
import numpy as np

from logger import logger, setup_logger, LoggerConfig
from configs import get_config, get_agent, get_agent_obs_keys
from runner.scheduler import build_scheduler
from runner.utils.data_recorder import DataRecorder
import os


def get_visualizer_params(agent_config, camera_name: str = "main", robot_name: str = None):
    """
    Extract visualizer parameters from agent config.
    
    Returns:
        dict with image_shm, intrinsic, T_world_camera, T_world_base, robot_name
        or None if not available
    """
    try:
        # Get camera shared memory
        if camera_name not in agent_config.cameras:
            logger.warning("Camera '{}' not found in agent config", camera_name)
            return None
        image_shm = agent_config.cameras[camera_name]
        
        # Get intrinsic
        if camera_name not in agent_config.intrinsics:
            logger.warning("Intrinsic for camera '{}' not found", camera_name)
            return None
        intrinsic = agent_config.intrinsics[camera_name]
        
        # Get extrinsic (T_world_camera)
        if camera_name not in agent_config.extrinsics:
            logger.warning("Extrinsic for camera '{}' not found", camera_name)
            return None
        T_world_camera = agent_config.extrinsics[camera_name]
        
        # Get robot name (first one if not specified)
        if robot_name is None:
            robot_names = list(agent_config.robots.keys())
            if len(robot_names) == 0:
                logger.warning("No robots found in agent config")
                return None
            robot_name = robot_names[0]
        
        # Get robot base pose
        if robot_name not in agent_config.robot_poses:
            T_world_base = np.eye(4, dtype=np.float32)
        else:
            T_world_base = agent_config.robot_poses[robot_name]
        
        return {
            "image_shm": image_shm,
            "intrinsic": intrinsic,
            "T_world_camera": T_world_camera,
            "T_world_base": T_world_base,
            "robot_name": robot_name
        }
    except Exception as e:
        logger.warning("Failed to extract visualizer params: {}", e)
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run evaluation.")
    parser.add_argument("--processor", type=str, required=True, help="processor config")
    parser.add_argument("--policy", type=str, required=True, help="policy config")
    parser.add_argument("--wrapper", type=str, required=True, help="wrapper config")
    parser.add_argument("--ckpt_path", type=str, help="checkpoint path")
    parser.add_argument("--agent", type=str, required=True, help="agent config")
    parser.add_argument("--agent_class", type=str, required=True, help="agent class")
    parser.add_argument("--agent_obs_keys", type=str, required=True, help="agent obs keys config name")
    parser.add_argument("--scheduler", type=str, required=True, help="scheduler config")
    parser.add_argument("--visualize", action="store_true", help="enable visualization")
    parser.add_argument("--vis_camera", type=str, default="main", help="camera name for visualization")
    parser.add_argument("--vis_robot", type=str, default=None, help="robot name for visualization")
    parser.add_argument("--vis_freq", type=int, default=20, help="visualization frequency")
    parser.add_argument("--logger", type=str, default="info", 
                        choices=["trace", "debug", "info", "success", "warning", "error", "critical"],
                        help="log level (default: info)")
    parser.add_argument("--fast_processor", type=str, default=None, help="fast processor config")
    parser.add_argument("--fast_policy", type=str, default=None, help="fast policy config")
    parser.add_argument("--fast_wrapper", type=str, default=None, help="fast wrapper config")
    parser.add_argument("--fast_ckpt_path", type=str, default=None, help="fast checkpoint path")
    parser.add_argument("--fast_agent_obs_keys", type=str, default=None, help="fast agent obs keys config name")
    parser.add_argument("--record", action="store_true", help="enable data recording")
    parser.add_argument("--record_visualizer", action="store_true", help="enable visualizer frame recording")
    parser.add_argument("--record_freq", type=int, default=30, help="recording frequency for cam and visualizer (default: 30)")
    args = parser.parse_args()

    # Setup logger with specified level
    setup_logger(LoggerConfig(level = args.logger.upper()))

    processor_config = get_config(args.processor, "processor")
    policy_config = get_config(args.policy, "policy")
    policy_wrapper_config = get_config(args.wrapper, "wrapper")
    agent = get_agent(args.agent, args.agent_class)
    agent_obs_keys = get_agent_obs_keys(args.agent, args.agent_obs_keys)
    scheduler_config = get_config(args.scheduler, "scheduler")

    scheduler_config.processor_config = processor_config
    scheduler_config.policy_config = policy_config
    scheduler_config.policy_wrapper_config = policy_wrapper_config
    scheduler_config.ckpt_path = args.ckpt_path

    kwargs = {}
    if scheduler_config.type == "slow_fast_scheduler":
        scheduler_config.fast_processor_config = get_config(args.fast_processor, "processor")
        scheduler_config.fast_policy_config = get_config(args.fast_policy, "policy")
        scheduler_config.fast_policy_wrapper_config = get_config(args.fast_wrapper, "wrapper")
        scheduler_config.fast_ckpt_path = args.fast_ckpt_path
        kwargs["fast_agent_obs_keys"] = get_agent_obs_keys(args.agent, args.fast_agent_obs_keys)

    scheduler = build_scheduler(
        scheduler_config, 
        agent=agent, 
        agent_obs_keys=agent_obs_keys,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **kwargs
    )

    # Setup recording paths
    scene_path = None
    if args.record or args.record_visualizer:
        try:
            data_base_dir = "eval_log_data" 
            os.makedirs(data_base_dir, exist_ok=True)
            
            existing_scenes = [d for d in os.listdir(data_base_dir) if d.startswith("scene_") and os.path.isdir(os.path.join(data_base_dir, d))]
            max_id = 0
            for d in existing_scenes:
                try:
                    sid = int(d.split("_")[1])
                    if sid > max_id:
                        max_id = sid
                except:
                    pass
            
            next_id = max_id + 1
            scene_path = os.path.join(data_base_dir, f"scene_{next_id:04d}")
            os.makedirs(scene_path, exist_ok=True)
            logger.info(f"Recording to {scene_path}")
        except Exception as e:
            logger.error(f"Failed to create recording directory: {e}")
            scene_path = None

    # Setup visualizer if requested
    visualizer = None
    if args.visualize:
        from runner.utils.visualizer import Visualizer
        
        vis_params = get_visualizer_params(
            agent.config,
            camera_name=args.vis_camera,
            robot_name=args.vis_robot
        )
        
        if vis_params is not None:
            # Prepare visualizer recording path if enabled
            vis_record_path = None
            if args.record_visualizer and scene_path is not None:
                vis_record_path = os.path.join(scene_path, "visualizer")
            
            visualizer = Visualizer(
                image_shm=vis_params["image_shm"],
                intrinsic=vis_params["intrinsic"],
                T_world_camera=vis_params["T_world_camera"],
                freq=args.vis_freq,
                record_path=vis_record_path,
                record_freq=args.record_freq if args.record_visualizer else None,
            )
            
            # Register agent and scheduler
            # Note: We register with generic names, the visualizer will iterate all available robots
            visualizer.register_agent(
                "agent", 
                agent, 
            )
            visualizer.register_scheduler(
                "scheduler", 
                scheduler, 
            )
            
            logger.info("Visualizer enabled for camera '{}', robot '{}'", 
                       args.vis_camera, vis_params["robot_name"])
        else:
            logger.warning("Failed to setup visualizer, continuing without it")

    # Setup Data Recorder (but don't start yet)
    recorder = None
    if args.record and scene_path is not None:
        try:
            logger.info(f"Initializing DataRecorder for {scene_path}")
            recorder = DataRecorder(agent, scene_path=scene_path, freq=args.record_freq)
        except Exception as e:
            logger.error(f"Failed to initialize DataRecorder: {e}")
            recorder = None

    # First input: Start visualizer and recorder
    logger.info("Press Enter to start recording...")
    input()
    
    # Start visualizer first (so we can see what's happening)
    if visualizer is not None:
        visualizer.run()
    
    # Start recorder
    if recorder is not None:
        recorder.start()
        logger.info("Recording started.")
    
    # Second input: Start scheduler (policy execution)
    logger.info("Press Enter to start scheduler...")
    input()
    
    # Start scheduler
    scheduler.start()
    logger.info("Scheduler started.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        scheduler.stop()
        if visualizer is not None:
            visualizer.stop()
        if recorder is not None:
            recorder.stop()
        print("Stopped.")
