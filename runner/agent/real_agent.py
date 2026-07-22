"""
Real Agent.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import time
import torch
import numpy as np

from easyrobot.arm.base import ArmBase
from easyrobot.gripper.base import GripperBase
from easyrobot.hand.base import HandBase
from easyrobot.camera.base import RGBDCameraBase

from runner.agent.base import BaseAgent
from runner.configs.agent import RealAgentConfig
from runner.configs.agent import AgentObsKeysConfig
from runner.utils.lowdim_provider import LowdimRecorder, get_lowdim_observation
from runner.utils.filter import OneEuroFilter, PoseOneEuroFilter


class RealAgent(BaseAgent):
    """
    Real agent.
    """
    def __init__(
        self,
        config: RealAgentConfig
    ) -> None:
        """ Initialization """
        super(RealAgent, self).__init__(config)

        self.lowdim_recorders: Dict[str, LowdimRecorder] = {}
        for recorder_key, recorder_config in self.config.lowdim_recorder_configs.items():
            device = self._get_device_from_ref(recorder_config.device_ref)
            self.lowdim_recorders[recorder_key] = LowdimRecorder(
                config = recorder_config,
                device = device
            )

        # Dict of callable functions for getting observation data
        self.lowdim_observations: Dict[str, callable] = {}
        for obs_key, obs_config in self.config.lowdim_observation_configs.items():
            recorder_key = obs_config.recorder_key
            if recorder_key not in self.lowdim_recorders:
                raise ValueError(
                    f"LowdimObservation '{obs_key}' references recorder '{recorder_key}' "
                )
            self.lowdim_observations[obs_key] = get_lowdim_observation(
                config = obs_config,
                recorder = self.lowdim_recorders[recorder_key]
            )

        # Action filters
        self.action_filters = {}
        for name, filter_config in self.config.pose_action_filter_configs.items():
            self.action_filters[name] = PoseOneEuroFilter(config = filter_config)
        for name, filter_config in self.config.vector_action_filter_configs.items():
            self.action_filters[name] = OneEuroFilter(config = filter_config)

        self.FUNC_DICT = {
            (self.prefix_config.robot, "tcp_pose"): "send_tcp_pose",
            (self.prefix_config.robot, "joint_pos"): "send_joint_pos",
            (self.prefix_config.gripper, "width"): "set_width"
        }

    def _get_device_from_ref(self, device_ref: str) -> Any:
        """
        Get device instance from reference string.
        """
        if "/" not in device_ref:
            raise ValueError(
                f"Invalid device_ref format: '{device_ref}'. "
                f"Expected format: 'type/name' (e.g., 'robot/main')"
            )
        
        device_type, device_name = device_ref.split("/", 1)
        
        device_dict = {
            "robot": self.config.robots,
            "gripper": self.config.grippers,
            "hand": self.config.hands,
        }

        if device_type not in device_dict:
            raise ValueError(
                f"Unknown device type '{device_type}' in device_ref '{device_ref}'. "
                f"Valid types: {list(device_dict.keys())}"
            )

        devices = device_dict[device_type]
        if device_name not in devices:
            raise ValueError(
                f"Device '{device_name}' not found in {device_type}s. "
                f"Available: {list(devices.keys())}"
            )

        return devices[device_name]

    def get_obs(
        self,
        obs_keys_config: AgentObsKeysConfig
    ) -> Dict[str, np.ndarray]:
        """ Get observation. """
        obs_dict = {}
        for cam_id, camera in self.config.cameras.items():
            if cam_id not in obs_keys_config.color_keys and cam_id not in obs_keys_config.depth_keys:
                continue
            cam_dict = camera.execute() if hasattr(camera, "execute") else camera.get_states() 
            if cam_id in obs_keys_config.color_keys:
                obs_dict[f"{self.prefix_config.color}/{cam_id}"] = self._process_image(cam_dict["rgb"])
            if cam_id in obs_keys_config.depth_keys:
                obs_dict[f"{self.prefix_config.depth}/{cam_id}"] = self._process_depth(cam_dict["depth"])
        
        for cam_id in obs_keys_config.intrinsic_keys:
            obs_dict[f"{self.prefix_config.intrinsic}/{cam_id}"] = self.config.intrinsics[cam_id]
        for cam_id in obs_keys_config.extrinsic_keys:
            obs_dict[f"{self.prefix_config.extrinsic}/{cam_id}"] = self.config.extrinsics[cam_id]
        for robot_name in obs_keys_config.robot_pose_keys:
            obs_dict[f"{self.prefix_config.robot}/{robot_name}"] = self.config.robot_poses[robot_name]
        
        for key in obs_keys_config.lowdim_provider_keys:
            obs_dict[key] = self.lowdim_observations[key]()
        
        return obs_dict, time.time()

    def stop_lowdim_recorders(self):
        """Stop all lowdim recorder threads."""
        for recorder in self.lowdim_recorders.values():
            recorder.stop()

    def key_action(self, key: str, action: np.ndarray, **kwargs):
        """ Key action. """
        if key in self.action_filters:
            action = self.action_filters[key](action)

        device_type, device_name, key_suffix = key.split('/', 2)
        device = getattr(self.config, f"{device_type}s")[device_name]
        device_func = self.FUNC_DICT[(device_type, key_suffix)]
        getattr(device, device_func)(action, **kwargs)

    def robot(self, robot_name: str) -> ArmBase:
        """ Get a robot by name. """
        return self.config.robots[robot_name]

    def get_force_torque_tcp(self, robot_name: str) -> np.ndarray:
        """ Get force torque tcp. """
        return self.robot(robot_name).get_force_torque_tcp()

    def gripper(self, gripper_name: str) -> GripperBase:
        """ Get a gripper by name. """
        return self.config.grippers[gripper_name]

    def hand(self, hand_name: str) -> HandBase:
        """ Get a hand by name. """
        return self.config.hands[hand_name]

    def action(self, action_dict: Dict[str, np.ndarray], action_params: Dict[str, Dict[str, Any]] = {}):
        """ Action. """
        robot_keys = [key for key in action_dict.keys() if key.startswith(self.prefix_config.robot)]
        gripper_keys = [key for key in action_dict.keys() if key.startswith(self.prefix_config.gripper)]
        hand_keys = [key for key in action_dict.keys() if key.startswith(self.prefix_config.hand)]
        
        for robot_name in self.platform_config.robot_names:
            # check tcp pose
            tcp_pose_key = f"{self.prefix_config.robot}/{robot_name}/tcp_pose"
            if tcp_pose_key in action_dict:
                force_frame_key = f"{self.prefix_config.robot}/{robot_name}/force_frame"
                tcp_wrench_key = f"{self.prefix_config.robot}/{robot_name}/tcp_wrench"
                force_frame_mask_key = f"{self.prefix_config.robot}/{robot_name}/force_frame_mask"

                tcp_pose = action_dict[tcp_pose_key]
                tcp_wrench = action_dict.get(tcp_wrench_key, None)
                force_frame = action_dict.get(force_frame_key, None)
                force_frame_mask = action_dict.get(force_frame_mask_key, None)

                kwargs = action_params[tcp_pose_key]
                if robot_name in self.config.force_control_robot_names:
                    # set force control frame
                    if force_frame is not None:
                        print("set force frame: ", force_frame)
                        self.robot(robot_name).set_force_control_frame('tcp', force_frame)
                    else:
                        print("reset force frame")
                        self.robot(robot_name).set_force_control_frame('world', [0, 0, 0, 1, 0, 0, 0])
                
                    # set force control axis
                    if force_frame_mask is not None:
                        if robot_name not in self.config.torque_control_robot_names:
                            force_frame_mask[3:] = False
                        print("set force control axis: ", force_frame_mask)
                        self.robot(robot_name).set_force_control_axis(
                            force_frame_mask,
                            kwargs.get("max_search_force_vel", [0.03, 0.03, 0.03])
                        )
                    
                    else:
                        self.robot(robot_name).set_force_control_axis(
                            [False, False, False, False, False, False], 
                            kwargs.get("max_search_force_vel", [0.03, 0.03, 0.03])
                        )
                    
                    # set wrench and max vel
                    if tcp_wrench is not None:
                        print("set tcp wrench: ", tcp_wrench)
                        kwargs["wrench"] = tcp_wrench
                
                if "max_search_force_vel" in kwargs:
                    del kwargs["max_search_force_vel"]

                print("send tcp pose: ", tcp_pose)
                self.robot(robot_name).send_tcp_pose(tcp_pose, **kwargs)
            
            
            # check joint pos
            joint_pos_key = f"{self.prefix_config.robot}/{robot_name}/joint_pos"
            if joint_pos_key in action_dict:
                self.robot(robot_name).send_joint_pos(
                    action_dict[joint_pos_key],
                    **action_params.get(joint_pos_key, {})
                )

        for gripper_name in self.platform_config.gripper_names:
            # check set width
            gripper_width_key = f"{self.prefix_config.gripper}/{gripper_name}/width"
            if gripper_width_key in action_dict:
                self.gripper(gripper_name).set_width(
                    action_dict[gripper_width_key],
                    **action_params.get(gripper_width_key, {})
                )
        
        for hand_name in self.platform_config.hand_names:
            # check hand joint state
            hand_joint_key = f"{self.prefix_config.hand}/{hand_name}/joint_pos"
            if hand_joint_key in action_dict:
                self.hand(hand_name).set_joint_pos(
                    action_dict[hand_joint_key],
                    **action_params.get(hand_joint_key, {})
                )
                
            