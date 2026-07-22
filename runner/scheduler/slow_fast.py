"""
Slow-Fast Scheduler for robotic control.
Manages a Slow (Vision) Policy and a Fast (Force) Policy, switching between them based on Fast Policy feedback.
"""

import os
import time
import threading
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from easydict import EasyDict as edict

from logger import logger
from runner.scheduler.adaptive import AdaptiveScheduler, ASTask
from runner.configs.scheduler import SlowFastSchedulerConfig
from runner.agent.real_agent import RealAgent
from runner.configs.agent import AgentObsKeysConfig
from policy import get_policy, PolicyWrapper
from data_infra.processor import DataProcessor
from utils.common import set_seed
from runner.utils.scheduler_utils import same_axis
from utils.transforms.interpolation import resample_trajectory



class SlowFastScheduler(AdaptiveScheduler):
    """
    Scheduler that switches between a Slow Vision Policy and a Fast Force Policy.
    Inherits from AdaptiveScheduler to reuse vision policy trajectory generation and smoothing.
    
    Modes:
    - SLOW (Vision): Uses AdaptiveScheduler logic (aspredictor_thread + task_queue).
    - FAST (Force): Uses high-frequency control loop with local policy inference.
    """
    
    MODE_SLOW = "slow"
    MODE_FAST = "fast"

    def __init__(
        self,
        config: SlowFastSchedulerConfig,
        agent: RealAgent,
        agent_obs_keys: AgentObsKeysConfig,
        fast_agent_obs_keys: AgentObsKeysConfig,
        device: torch.device
    ) -> None:
        """ Initialization. """
        super(SlowFastScheduler, self).__init__(
            config = config,
            agent = agent,
            agent_obs_keys = agent_obs_keys,
            device = device
        )
        self.fast_agent_obs_keys = fast_agent_obs_keys
        
        # Fast policy and its wrapper
        self.fast_policy = get_policy(self.config.fast_policy_config).to(self.device)
        assert os.path.exists(self.config.fast_ckpt_path), "Checkpoint {} does not exist.".format(self.config.fast_ckpt_path)
        checkpoint = torch.load(self.config.fast_ckpt_path, map_location = self.device)
        self.fast_policy.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
        self.fast_policy.eval()
        self.fast_policy_wrapper = PolicyWrapper(self.fast_policy, self.config.fast_policy_wrapper_config)

        # Data processor for fast policy
        self.fast_processor = DataProcessor(self.config.fast_processor_config)

        # Switching logic state
        self.current_mode = self.MODE_SLOW
        self.switch_start_time = 0.0
        self.last_switch_trigger = False
        
        # Timing
        self.fast_dt = 1.0 / self.config.fast_frequency
        
        # Fast loop state
        self._last_force_frame = None
        self._fast_lock = threading.Lock()
        self._fast_result_lock = threading.Lock()
        self._fast_thread = threading.Thread(target = self._fast_inference_thread)
        self._fast_thread.daemon = True
        
        # Fast Execution Queue
        self.fast_task_queue = deque()

    def start(self):
        """ Start all threads. """
        super().start() # Starts planner (asplanner) and predictor (aspredictor)
        self._fast_thread.start()
        
    def stop(self):
        """ Stop all threads. """
        super().stop()
        if self._fast_thread.is_alive():
            self._fast_thread.join()
    
    def model_inference_fast(self) -> Dict[str, np.ndarray]:
        """ Run model inference for fast policy. """
        with torch.inference_mode():
            while self.last_inference is None:
                time.sleep(0.1)
            # create observation and inject slow policy's action into observation
            obs_raw, time_obs = self.agent.get_obs(self.fast_agent_obs_keys)
            aux_action = self.last_inference[2]
            for key, slow_key in self.config.fast_slow_key_mapping.items():
                obs_raw[key] = aux_action[slow_key]
            obs = self.agent.convert_obs(obs_raw, self.device)
            
            # model inference
            obs_dict = self.fast_processor(obs, enable_aug = False, process_type = "forward")
            action_dict = self.fast_policy_wrapper(obs_dict, action_dict = None, batch_size = 1)
            action = self.fast_processor(obs, action_dict, process_type = "backward")

            # convert back to agent action
            action_raw = self.agent.convert_action(action)
            action_raw = self.agent.to_agent(action_raw)

        # resample action to match fast frequency
        synced_action = {}
        aux_action = {}
        for key in action_raw.keys():
            if key in self.config.fast_sync_action_config.keys():
                config = self.config.fast_sync_action_config[key]
                synced_action[key] = resample_trajectory(
                    data = action_raw[key], 
                    source_freq = config.source_frequency, 
                    source_length = config.source_length,
                    target_freq = self.config.planner_freq,
                    target_length = self.config.fast_sync_horizon,
                    sampling_method = config.interpolation,
                    rotation_rep = config.rotation_rep,
                    convention = config.convention
                )
            else:
                aux_action[key] = action_raw[key]

        # save last inference [raw observation, synced_action, inference latency, obs time]
        self.last_fast_inference = (obs_raw, synced_action, aux_action, time.time() - time_obs, time_obs)
        # print(synced_action["robot/right/force_frame_mask"], synced_action["switch_signal"])
        # print(synced_action["robot/right/tcp_pose"])
        return synced_action, aux_action


    def _fast_inference_thread(self):
        """
        Background thread for fast policy.
        """
        while not self._stop_event.is_set():
            loop_start = time.time()

            # Get inference resutls
            self.model_inference_fast()
            obs_raw, synced_action, aux_action, latency, obs_time = self.last_fast_inference
            logger.debug("Fast policy inference completed in {:.2f}s", latency)
            # Dropout Waypoint
            dropout_idx = int(np.ceil(latency * self.config.planner_freq))
            dropout_idx = max(0, dropout_idx)

            for key in synced_action.keys():
                synced_action[key] = synced_action[key][dropout_idx:, ...]
            
            # Generate tasks
            tasks = []
            for i in range(self.config.fast_sync_horizon - dropout_idx):
                action = {}
                for key in synced_action.keys():
                    action[key] = synced_action[key][i]
                task = ASTask(
                    action = action,
                    waypoint_type = "raw"
                )
                tasks.append(task)

            # 3. Append into queue.
            self.fast_task_queue.clear()
            self.fast_task_queue.extend(tasks)
            
            loop_elapsed = time.time() - loop_start
            if loop_elapsed < self.fast_dt:
                time.sleep(self.fast_dt - loop_elapsed)


    def asplanner_thread(self):
        """
        Main Control Loop.
        Runs at fast_frequency (e.g. 50Hz).
        Continuously monitors Fast Policy Flag for immediate switching.
        """
        self._predictor_request_event.set() 
        
        while not self._stop_event.is_set():
            loop_start = time.time()

            if not self.task_queue and not self.is_infer_running:
                self._predictor_request_event.set()
            
            fast_action = None
            if self.fast_task_queue: # fast pop
                fast_action = self.fast_task_queue.popleft()
            
            if fast_action is None:
                time.sleep(0.01)
                continue
            
            # Extract Flag
            router_flag = bool(fast_action.action[self.config.switch_key])
            logger.debug(f"Router flag: {router_flag}")

            if router_flag:
                # Case A: FAST
                if self.current_mode != self.MODE_FAST:
                    if not self.last_switch_trigger:
                        self.switch_start_time = time.time()
                        self.last_switch_trigger = True
                    
                    elapsed = time.time() - self.switch_start_time
                    if elapsed >= self.config.switch_duration:
                        self.last_switch_trigger = False
                        self._switch_mode(self.MODE_FAST)
                        self.agent.action(fast_action.action, self.config.action_params) # fast action
                        if self.task_queue:
                            self.task_queue.popleft() # slow pop
                    else:
                        if self.task_queue:
                            self._step_task() # slow action
                
                else:
                    # Already in FAST mode
                    self.last_switch_trigger = False
                    self.agent.action(fast_action.action, self.config.action_params) # fast action
                    if self.task_queue:
                        self.task_queue.popleft() # slow pop
            
            else:
                # Case B: SLOW
                if self.current_mode == self.MODE_FAST:
                    if not self.last_switch_trigger:
                        self.switch_start_time = time.time()
                        self.last_switch_trigger = True
                    
                    elapsed = time.time() - self.switch_start_time
                    if elapsed >= self.config.switch_duration:
                        self.last_switch_trigger = False
                        self._switch_mode(self.MODE_SLOW)
                        # TODO: reset axis
                        if self.task_queue:
                            self._step_task() # slow action
                        # fast already popped
                    else:
                        self.agent.action(fast_action.action, self.config.action_params) # fast action
                        if self.task_queue:
                            self.task_queue.popleft() # slow pop
                    
                else:
                    self.last_switch_trigger = False
                    if self.task_queue:
                        self._step_task() # slow action
                    # fast already popped
            
            self._trigger_inference()

            loop_elapsed = time.time() - loop_start
            if loop_elapsed < self.planner_dt:
                time.sleep(self.planner_dt - loop_elapsed)


    def _switch_mode(self, mode):
        """ Transition from FAST to SLOW mode. """
        logger.info(f"Switching to {mode} mode.")
        self.current_mode = mode
        self.task_queue.clear()
        if mode == self.MODE_SLOW:
            self.last_switch_trigger = False
            self.switch_start_time = 0.0
        self._predictor_request_event.set()

    def get_queue_wrenches(self) -> Dict[str, Dict[str, Any]]:
        """
        Get predicted wrenches from the fast task queue.
        Used for visualization in Fast mode.
        Returns:
            Dict[str, Dict]: {robot_name: {tcp_wrench: ..., force_frame: ...}}
        """
        wrenches = {}
        if self.current_mode != self.MODE_FAST:
            return wrenches
        
        if not self.fast_task_queue:
            return wrenches
            
        # Peek at the next action
        next_task = self.fast_task_queue[0]
        action = next_task.action
        
        robot_names = self.platform_config.robot_names
        
        for robot_name in robot_names:
            tcp_wrench_key = f"{self.prefix_config.robot}/{robot_name}/tcp_wrench"
            force_frame_key = f"{self.prefix_config.robot}/{robot_name}/force_frame"
            
            info = {}
            if tcp_wrench_key in action:
                info["tcp_wrench"] = action[tcp_wrench_key]
            if force_frame_key in action:
                info["force_frame"] = action[force_frame_key]
                
            if info:
                wrenches[robot_name] = info
            
        return wrenches
