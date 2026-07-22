import os
import time
import shutil
import multiprocessing
import numpy as np
import cv2
import h5py
from typing import Dict, Any, List

from logger import logger
from easyrobot.utils.shared_memory import SharedMemoryManager, DictSharedMemoryManager, generate_config_from_data


class DataRecorder(multiprocessing.Process):
    """
    Independent process to record data from Shared Memory to disk.
    
    Records:
    - Camera images (Color/Depth) -> PNG files
    - Low-dim data (Robot states) -> HDF5 file
    """
    def __init__(
        self, 
        agent,
        scene_path: str,
        freq: int = 30,
    ):
        """
        Args:
            agent: Agent instance to extract SHM configuration from.
            scene_path: Directory to save the recorded data (e.g. .../scene_0001)
            freq: Recording frequency in Hz.
        """
        super().__init__(name="DataRecorder")
        self.scene_path = scene_path
        self.freq = freq
        self.dt = 1.0 / freq
        
        # event to control the loop
        self._stop_event = multiprocessing.Event()
        
        self.camera_configs = {} # name -> {shm_name, dict_cfg (if dict), type}
        self.robot_configs = {}  # name -> {shm_name, dict_cfg (if dict), type}
        
        self._extract_shm_info(agent)
        
    def _extract_shm_info(self, agent):
        """Extract SHM names and configs from the agent."""
        
        # 1. Cameras
        if hasattr(agent.config, "cameras") and hasattr(agent.config, "camera_shm_names"):
            for name, cam_obj in agent.config.cameras.items():
                if name not in agent.config.camera_shm_names:
                    logger.warning(f"[DataRecorder] Camera '{name}' not found in camera_shm_names. Skipping.")
                    continue
                
                shm_name = agent.config.camera_shm_names[name]
                dict_cfg = cam_obj.dict_cfg if isinstance(cam_obj, DictSharedMemoryManager) else None
                
                self.camera_configs[name] = {
                    "shm_name": shm_name,
                    "dict_cfg": dict_cfg,
                    "type": 1
                }

        # 2. Robots
        if hasattr(agent.config, "robots"):
            for name, robot_obj in agent.config.robots.items():
                # Get shm_name from robot_shm_names config or robot object
                shm_name = None
                if hasattr(agent.config, "robot_shm_names") and name in agent.config.robot_shm_names:
                    shm_name = agent.config.robot_shm_names[name]
                elif hasattr(robot_obj, "shm_name"):
                    shm_name = robot_obj.shm_name
                
                if not shm_name:
                    logger.warning(f"[DataRecorder] Robot '{name}' has no shm_name. Skipping.")
                    continue
                
                # Generate dict_cfg from robot states
                dict_cfg = None
                try:
                    if hasattr(robot_obj, "get_states"):
                        robot_states = robot_obj.get_states()
                        dict_cfg = generate_config_from_data(robot_states, shm_name=shm_name)
                except Exception as e:
                    logger.warning(f"[DataRecorder] Failed to generate dict_cfg for robot '{name}': {e}")
                
                self.robot_configs[name] = {
                    "shm_name": shm_name,
                    "dict_cfg": dict_cfg,
                    "type": 1
                }

    def run(self):
        """Main process loop."""
        try:
            # --- Setup Output Directories ---
            cam_dir = os.path.join(self.scene_path, "cam")
            lowdim_dir = os.path.join(self.scene_path, "lowdim")
            
            os.makedirs(lowdim_dir, exist_ok=True)
            
            # Setup H5 file
            h5_path = os.path.join(lowdim_dir, "lowdim.h5")
            h5_file = h5py.File(h5_path, "w", libver="latest")
            
            # Define H5 Datasets (we'll create them lazily or upfront)
            # We'll use a dynamic approach: create dataset on first frame based on data keys
            datasets = {}
            
            # --- Re-connect to Shared Memory ---
            camera_managers = {}
            robot_managers = {}
            
            for name, cfg in self.camera_configs.items():
                try:
                    if cfg["type"] == 1:
                        # Use dict_cfg to recreate DictSharedMemoryManager
                        if cfg.get("dict_cfg") is not None:
                            manager = DictSharedMemoryManager(type=1, dict_cfg=cfg["dict_cfg"])
                        else:
                            logger.error(f"[DataRecorder] Camera '{name}' has no dict_cfg. Cannot recreate DictSharedMemoryManager.")
                            continue
                    else:
                        manager = SharedMemoryManager(shm_name=cfg["shm_name"])
                    
                    camera_managers[name] = manager
                    
                    # Create dirs for this camera
                    os.makedirs(os.path.join(cam_dir, name, "color"), exist_ok=True)
                    os.makedirs(os.path.join(cam_dir, name, "depth"), exist_ok=True)
                    
                except Exception as e:
                    logger.error(f"[DataRecorder] Failed to connect to camera SHM '{name}': {e}")

            for name, cfg in self.robot_configs.items():
                try:
                    # Robots are almost always Dict type in this codebase
                    # Use dict_cfg to recreate DictSharedMemoryManager
                    if cfg.get("dict_cfg") is not None:
                        manager = DictSharedMemoryManager(type=1, dict_cfg=cfg["dict_cfg"])
                        robot_managers[name] = manager
                        logger.info(f"[DataRecorder] Connected to robot SHM '{name}' with shm_name '{cfg.get('shm_name')}'")
                    else:
                        logger.warning(f"[DataRecorder] Robot '{name}' has no dict_cfg. Cannot recreate DictSharedMemoryManager. Low-dim data will not be recorded.")
                        continue
                except Exception as e:
                    logger.error(f"[DataRecorder] Failed to connect to robot SHM '{name}': {e}")

            if len(robot_managers) == 0:
                logger.warning(f"[DataRecorder] No robot managers connected. Low-dim data will not be recorded.")
            
            logger.info(f"[DataRecorder] Started recording to {self.scene_path}. Connected to {len(camera_managers)} cameras and {len(robot_managers)} robots.")
            
            # --- Recording Loop ---
            idx = 0
            while not self._stop_event.is_set():
                start_time = time.time()
                ts_int = int(start_time * 1000) # timestamp in ms
                
                # 1. Record Cameras
                for name, manager in camera_managers.items():
                    try:
                        data = manager.execute() # Read
                        if isinstance(data, dict):
                            # Extract color/depth
                            # Common keys: "rgb", "color", "depth"
                            color = None
                            depth = None
                            
                            # Try to find color
                            for k in ["color", "rgb", "image"]:
                                if k in data:
                                    color = data[k]
                                    break
                            
                            # Try to find depth
                            for k in ["depth"]:
                                if k in data:
                                    depth = data[k]
                                    break
                            
                            if color is not None:
                                # Convert RGB to BGR for cv2
                                if color.shape[-1] == 3:
                                    color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                                elif color.shape[-1] == 4:
                                    color = cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)
                                
                                path = os.path.join(cam_dir, name, "color", f"{ts_int}.png")
                                cv2.imwrite(path, color)
                                
                            if depth is not None:
                                # Save depth as png (16bit usually preferred if raw, but matching user req)
                                path = os.path.join(cam_dir, name, "depth", f"{ts_int}.png")
                                cv2.imwrite(path, depth)

                    except Exception as e:
                        # Don't spam logs too much in high freq loop
                        if idx % 100 == 0:
                            logger.error(f"[DataRecorder] Error recording camera '{name}': {e}")

                # 2. Record Low-dim (Robots)

                self._append_h5(h5_file, datasets, "timestamp", np.array([ts_int], dtype=np.int64), idx)
                
                for name, manager in robot_managers.items():
                    try:
                        data = manager.execute()
                        if isinstance(data, dict):
                            # Flatten dict to datasets
                            # e.g. robot "main" -> "tcp_pose_main", "q_main"
                            suffix = name if name else "0"
                            
                            # Key name mapping: map internal keys to desired output keys
                            key_mapping = {
                                "joint_pos": "q",  # joint_pos -> q
                                "joint_vel": "dq",  # joint_vel -> dq
                                # Keep other keys as-is: tcp_pose, tcp_vel, force_torque_tcp
                            }
                            
                            # Log available keys on first frame for debugging
                            if idx == 0:
                                logger.info(f"[DataRecorder] Robot '{name}' data keys: {list(data.keys())}")
                            
                            for key, val in data.items():
                                if val is None: continue
                                
                                # Convert to numpy if needed
                                val_np = np.asarray(val)
                                if val_np.size == 0: continue
                                
                                # Map key name if needed
                                output_key = key_mapping.get(key, key)
                                ds_name = f"{output_key}_{suffix}"
                                self._append_h5(h5_file, datasets, ds_name, val_np, idx)
                                
                    except Exception as e:
                        if idx % 100 == 0:
                            logger.error(f"[DataRecorder] Error recording robot '{name}': {e}")

                idx += 1
                
                # Maintain frequency
                elapsed = time.time() - start_time
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Cleanup
            h5_file.close()
            logger.info("[DataRecorder] Stopped and saved data.")

        except Exception as e:
            logger.exception(f"[DataRecorder] Process crashed: {e}")
            
    def _create_dataset(self, h5_file, datasets, name, shape, dtype):
        """Helper to create resizable h5 dataset."""
        # Chunk size: ~1 sec of data
        chunk_len = max(1, self.freq)
        maxshape = (None,) + shape[1:]
        chunks = (chunk_len,) + shape[1:]
        
        ds = h5_file.create_dataset(
            name, shape=shape, maxshape=maxshape, chunks=chunks,
            dtype=dtype, compression="gzip", compression_opts=4
        )
        datasets[name] = ds
        return ds

    def _append_h5(self, h5_file, datasets, name, data, idx):
        """Append data to HDF5 dataset, creating/resizing if needed."""
        data_np = np.asarray(data)
        
        # If scalar, resize to 1-dim array
        if data_np.ndim == 0:
            data_np = data_np.reshape(1)
            
        # If this is the first time we see this key, create dataset
        if name not in datasets:
            # Shape: (0, *data_shape)
            shape = (0,) + data_np.shape
            self._create_dataset(h5_file, datasets, name, shape, data_np.dtype)
            
        ds = datasets[name]
        
        # Resize if needed
        if idx >= ds.shape[0]:
            new_size = idx + 1
            ds.resize((new_size,) + ds.shape[1:])
            
        ds[idx] = data_np

    def stop(self):
        """Signal process to stop."""
        self._stop_event.set()
        self.join(timeout=2.0)
        if self.is_alive():
            self.terminate()
