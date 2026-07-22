import torch
import torch.nn as nn
from typing import Dict, Any, Union, List

from policy.configs.wrapper import *


class PolicyWrapper(nn.Module):
    def __init__(
        self,
        policy: nn.Module,
        config: PolicyWrapperConfig
    ) -> None:
        """ Initialization. """
        super().__init__()
        self.config = config
        self.policy = policy
    
    def _reorganize(
        self,
        data_dict: Dict[str, Any],
        config: ReorganizeConfig
    ) -> Dict[str, Any]:
        """ Reorganize keys. """
        if config.type == "pack":
            assert config.key not in data_dict
            data = []
            for key, dim in zip(config.key_list, config.dim_list):
                assert key in data_dict
                assert data_dict[key].shape[-1] == dim
                data.append(data_dict[key])
            data_dict[config.key] = torch.cat(data, dim = -1)
            return data_dict
        elif config.type == "unpack":
            assert config.key in data_dict
            data = torch.split(data_dict[config.key], config.dim_list, dim = -1)
            for key, tensor in zip(config.key_list, data):
                assert key not in data_dict
                data_dict[key] = tensor
            return data_dict
        elif config.type == "stack":
            assert config.key not in data_dict
            data = []
            target_shape = None
            for key in config.key_list:
                assert key in data_dict
                if target_shape is None:
                    target_shape = data_dict[key].shape
                else:
                    assert data_dict[key].shape == target_shape
                data.append(data_dict[key])
            data_dict[config.key] = torch.stack(data, dim = 1)
        else:
            raise ValueError(f"Invalid reorganize type: {config.type}")

    def _provide(
        self,
        data_dict: Dict[str, Any],
        config: DataProviderConfig
    ) -> Dict[str, Any]:
        """ Provide keys, including reorganizing and renaming. """
        for reorganize_config in config.reorganize_configs:
            data_dict = self._reorganize(data_dict, reorganize_config)
        return {new_key: data_dict[old_key] for new_key, old_key in config.provider_keys.items()}
    
    def forward(
        self,
        obs_dict: Dict[str, Any],
        action_dict: Dict[str, Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """ Policy call. """
        obs_dict = self._provide(obs_dict, self.config.obs_provider)

        if action_dict is None: # inference
            action_dict = self.policy(
                obs_dict = obs_dict, 
                action_dict = None, 
                **kwargs
            )
            action_dict = self._provide(action_dict, self.config.inference_action_provider)
            return action_dict
        else:
            action_dict = self._provide(action_dict, self.config.train_action_provider)
            return self.policy(
                obs_dict = obs_dict, 
                action_dict = action_dict, 
                **kwargs
            )
        
    def get_vision_feat(
        self,
        obs_dict: Dict[str, Any],
        **kwargs
    ):
        """ Get vision feature from the policy. """
        assert hasattr(self.policy, "get_vision_feat")
        obs_dict = self._provide(obs_dict, self.config.obs_provider)
        return self.policy.get_vision_feat(obs_dict, **kwargs)
    