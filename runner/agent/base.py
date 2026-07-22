"""
Agent base.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np

from runner.configs.agent import BaseAgentConfig
from utils.common import to_tensor, to_device, sample_to_batch, batch_to_sample, to_numpy


class BaseAgent:
    """
    Base agent.
    """
    def __init__(
        self,
        config: BaseAgentConfig
    ) -> None:
        """ Initialization """
        self.config = config
        self.platform_config = self.config.platform_config
        self.prefix_config = self.config.prefix_config

    def ready(self) -> None:
        """ Agent ready. """
        raise NotImplementedError

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        """ Process image. """
        image = np.array(image, dtype = np.float32) / 255.0
        image = image.transpose([2, 0, 1])
        return image

    def _process_depth(self, depth: np.ndarray, depth_scale: float = 1000.0) -> np.ndarray:
        """ Process depth. """
        depth = np.array(depth, dtype = np.float32) / depth_scale
        return depth

    def get_obs(self) -> Dict[str, np.ndarray]:
        """ Get observation. """
        raise NotImplementedError
    
    def convert_obs(self, obs_dict: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
        """ Convert observation to tensor. """
        obs_dict = to_tensor(obs_dict)
        obs_dict = sample_to_batch(obs_dict)
        obs_dict = to_device(obs_dict, device)
        return obs_dict

    def to_agent(self, action_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """ Convert action dict to agent's format. """
        action_to_agent = {}
        for key, provider_key in self.config.agent_action_providers.items():
            if provider_key in action_dict:
                action_to_agent[key] = action_dict[provider_key]
        return action_to_agent

    def convert_action(self, action_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """ Convert action to numpy. """
        action_dict = to_device(action_dict, "cpu")
        action_dict = batch_to_sample(action_dict)
        action_dict = to_numpy(action_dict)
        return action_dict
    
    def key_action(self, key: str, action: np.ndarray, **kwargs):
        """ Key action. """
        raise NotImplementedError
    
    def action(self, action_dict: Dict[str, np.ndarray], **kwargs):
        """ Action. """
        raise NotImplementedError