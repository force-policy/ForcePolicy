"""
Local Agent.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import time
import torch
import numpy as np

from PIL import Image

from runner.agent.base import BaseAgent
from runner.configs.agent import LocalAgentConfig


class LocalAgent(BaseAgent):
    """
    Local agent.
    """
    def __init__(
        self,
        config: LocalAgentConfig
    ) -> None:
        """ Initialization """
        super(LocalAgent, self).__init__(config)
    
    def ready(self) -> None:
        """ Agent ready. """
        pass

    def get_obs(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """ Get observation. """
        obs_dict = {}
        for cam_id, rgb in self.config.colors.items():
            obs_dict[f"{self.prefix_config.color}/{cam_id}"] = self._process_image(rgb)
        for cam_id, depth in self.config.depths.items():
            obs_dict[f"{self.prefix_config.depth}/{cam_id}"] = self._process_depth(depth)
        for cam_id, intrinsic in self.config.intrinsics.items():
            obs_dict[f"{self.prefix_config.intrinsic}/{cam_id}"] = intrinsic
        for cam_id, extrinsic in self.config.extrinsics.items():
            obs_dict[f"{self.prefix_config.extrinsic}/{cam_id}"] = extrinsic
        for robot_name, robot_pose in self.config.robot_poses.items():
            obs_dict[f"{self.prefix_config.robot}/{robot_name}"] = robot_pose
        for lowdim_name, lowdim in self.config.lowdim.items():
            obs_dict[lowdim_name] = lowdim
        return obs_dict, time.time()

    def key_action(self, key: str, action: np.ndarray, **kwargs):
        """ Key action. """
        pass
    
    def action(self, action_dict: Dict[str, np.ndarray], **kwargs):
        """ Action. """
        pass
