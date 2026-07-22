"""
Scheduler for robotic control with waypoint interpolation and dropout
"""

import math
import threading
import time
from collections import deque
from typing import Any, Dict, List

import numpy as np
import torch
from dataclasses import dataclass, field
from easydict import EasyDict as edict

from logger import logger

from runner.scheduler.base import SchedulerBase
from runner.agent.real_agent import RealAgent
from runner.configs.agent import AgentObsKeysConfig
from runner.configs.scheduler import AdaptiveSchedulerConfig
from runner.configs.lowdim_provider import LowdimRecorderConfig
from runner.utils.lowdim_provider import LowdimRecorder

from runner.utils.trajectory import build_indices, build_trajectory
from runner.utils.waypoint_dropout import waypoint_dropout


# ----------------------------- task container -----------------------------
@dataclass
class ASTask:
    action: Dict[str, Any]
    waypoint_type: str


# ----------------------------- adaptive scheduler -----------------------------
class AdaptiveScheduler(SchedulerBase):
    """
    Adaptive scheduler.
    """
    def __init__(
        self,
        config: AdaptiveSchedulerConfig,
        agent: RealAgent,
        agent_obs_keys: AgentObsKeysConfig,
        device: torch.device
    ) -> None:
        """ Initialization. """
        super(AdaptiveScheduler, self).__init__(
            config = config,
            agent = agent,
            agent_obs_keys = agent_obs_keys,
            device = device
        )
        
        self.lowdim_recorders = {}
        for robot_name in self.platform_config.robot_names:
            if robot_name in agent.config.robots:
                # Create a recorder config for TCP pose (used for waypoint dropout)
                tcp_pose_config = LowdimRecorderConfig(
                    device_ref = f"robot/{robot_name}",
                    device_func = "get_tcp_pose",
                    max_record_frequency = 200.0,
                    max_window_time = 10.0,
                )
                self.lowdim_recorders[robot_name] = LowdimRecorder(
                    config = tcp_pose_config,
                    device = agent.config.robots[robot_name],
                )
        
        self.planner_freq = config.planner_freq
        self.planner_dt = 1.0 / self.planner_freq

        # Control settings
        self.max_infer_dt = config.max_infer_dt
        self.pre_infer_time = config.pre_infer_time

        # Threading and state
        self.task_queue = deque()
        self._stop_event = threading.Event()
        self._predictor_request_event = threading.Event()
        
        # Runtime state
        self.is_infer_running = False
        self.infer_stop_time = 0

        # Threads
        self._planner_thread = None
        self._predictor_thread = None

        self.eval_finish = False
        self.last_ee_command = None


    def _estimate_queue_time(self) -> float:
        """
        Estimate remaining queue execution time.
        Simple estimation based on task count and planner_dt.
        """
        return len(self.task_queue) * self.planner_dt

    # ---------- lifecycle ----------
    def start(self):
        """Start all scheduler threads"""
        self._stop_event.clear()
        self._planner_thread = threading.Thread(
            target = self.asplanner_thread, 
            name = "asplanner_thread", 
            daemon = True
        )
        self._predictor_thread = threading.Thread(
            target = self.aspredictor_thread, 
            name = "aspredictor_thread", 
            daemon = True
        )
        self._planner_thread.start()
        self._predictor_thread.start()
    
    def stop(self):
        """Stop all scheduler threads and lowdim recorders"""
        self._stop_event.set()
        
        # Stop all lowdim recorders
        for recorder in self.lowdim_recorders.values():
            recorder.stop()
        
        # Wait for threads to finish
        if self._planner_thread is not None and self._planner_thread.is_alive():
            self._planner_thread.join(timeout=1.0)
        if self._predictor_thread is not None and self._predictor_thread.is_alive():
            self._predictor_thread.join(timeout=1.0)


    # ---------- planner loop ----------
    def _step_task(self):
        """ Step a single task. """
        task = self.task_queue.popleft()
        self.agent.action(task.action, self.config.action_params)


    def _trigger_inference(self):
        """ Trigger policy inference. """
        if not self.is_infer_running:
             if (self.max_infer_dt > 0 and time.time() - self.infer_stop_time > self.max_infer_dt):
                 self._predictor_request_event.set()
             elif (
                 self.pre_infer_time > 0
                 and self._estimate_queue_time() < self.pre_infer_time
             ):
                 self._predictor_request_event.set()


    def asplanner_thread(self):
        """Main planner thread that executes waypoint tasks"""
        self._predictor_request_event.set()

        while not self._stop_event.is_set():
            loop_start = time.time()

            if not self.task_queue:
                if not self.is_infer_running:
                    self._predictor_request_event.set()
                time.sleep(0.1)
                continue
            
            self._step_task()
            self._trigger_inference()
            
            loop_elapsed = time.time() - loop_start
            if loop_elapsed < self.planner_dt:
                time.sleep(self.planner_dt - loop_elapsed)
    

    # ---------- predictor loop ----------
    def construct_task_queue(
        self, 
        synced_action: Dict[str, np.ndarray]
    ) -> List[ASTask]:
        # currently only support tcp
        existing_robot_names = []
        for key in synced_action.keys():
            device_type, device_name, device_func = key.split("/")
            if device_type == self.prefix_config.robot and device_func == "tcp_pose":
                existing_robot_names.append(device_name)
        
        # DTW waypoint dropout
        pred_trajs = []
        ref_trajs = []

        for robot_name in existing_robot_names:
            pred_traj = synced_action[f"{self.prefix_config.robot}/{robot_name}/tcp_pose"]
            ref_traj = self.lowdim_recorders[robot_name].get_window(
                freq = self.config.sync_frequency, 
                length = self.config.sync_horizon, 
                is_reverse = False
            )
            pred_trajs.append(pred_traj)
            ref_trajs.append(ref_traj)
        
        pred_trajs = np.stack(pred_trajs, axis = 1)
        ref_trajs = np.stack(ref_trajs, axis = 1)

        tic = time.time()
        start_idx = waypoint_dropout(
            raw_tcps = pred_trajs,
            ref_traj = ref_trajs,
            dropout_cfg = self.config.waypoint_dropout,
            dt = 1.0 / self.config.sync_frequency,
        )
        dtw_latency = time.time() - tic
        dtw_latency_idx = int(dtw_latency / self.planner_dt)

        logger.debug(
            f"DTW dropout: {len(pred_trajs)} raw points -> start_idx = {start_idx} + {dtw_latency_idx} (dtw latency); keep {len(pred_trajs) - start_idx - dtw_latency_idx} points",
        )

        start_idx = start_idx + dtw_latency_idx

        for key in synced_action.keys():
            synced_action[key] = synced_action[key][start_idx:]

        # Generate waypoints and ASTask
        num_waypoints = np.inf
        for robot_name in existing_robot_names:
            # generate trajectory from current TCP
            robot_waypoints = build_trajectory(
                pred_tcp_poses = synced_action[f"{self.prefix_config.robot}/{robot_name}/tcp_pose"],
                cur_tcp_pose = self.agent.robot(robot_name).get_tcp_pose(),
                cur_tcp_vel = self.agent.robot(robot_name).get_tcp_vel(),
                source_freq = self.config.sync_frequency / self.config.time_scaling_factor,
                target_freq = self.planner_freq,
                interp_type = self.config.waypoint_interp_type
            )
            synced_action[f"{self.prefix_config.robot}/{robot_name}/tcp_pose"] = robot_waypoints
            num_waypoints = min(num_waypoints, len(robot_waypoints))
        
        # Generate other related action indices
        indices = build_indices(
            source_freq = self.config.sync_frequency / self.config.time_scaling_factor,
            target_freq = self.planner_freq,
            N_target = num_waypoints,
            N_source = self.config.sync_horizon - start_idx,
        )

        tasks = []
        for i in range(num_waypoints):
            action_step = {}
            for robot_name in existing_robot_names:
                robot_key = f"{self.prefix_config.robot}/{robot_name}/tcp_pose"
                action_step[robot_key] = synced_action[robot_key][i]
            
            if indices[i] != -1:
                idx = indices[i]
                for key in synced_action.keys():
                    if key not in action_step.keys():
                        action_step[key] = synced_action[key][idx]
                waypoint_type = "raw"
            else:
                waypoint_type = "interp"
            
            tasks.append(ASTask(action = action_step, waypoint_type = waypoint_type))
        
        return tasks
        

    def aspredictor_thread(self):
        """Prediction thread that runs model inference"""
        while not self._stop_event.is_set():
            if self._predictor_request_event.wait(timeout = 0.1):
                self._predictor_request_event.clear()
                self.aspredictor_infer()

    def aspredictor_infer(self):
        """Run full inference pipeline: observe -> predict -> generate waypoints"""
        self.is_infer_running = True

        # Run model inference
        self.model_inference()
        obs_raw, synced_action, aux_action, latency, time_obs = self.last_inference
        logger.info("Inference completed in {:.2f}s", latency)
        
        # Generate waypoints for each robot group
        tasks = self.construct_task_queue(synced_action)
        self.task_queue.clear()
        self.task_queue.extend(tasks)

        # Cleanup
        time.sleep(0.1)
        self.infer_stop_time = time.time()
        self.is_infer_running = False


    def get_queue_tcps(self, robot_name: str) -> List[np.ndarray]:
        """
        Get all TCP poses currently in the task queue.
        """
        pose_key = f"{self.prefix_config.robot}/{robot_name}/tcp_pose"
        return [
            (t.action[pose_key], t.waypoint_type)
            for t in self.task_queue if pose_key in t.action.keys()
        ]
    
    