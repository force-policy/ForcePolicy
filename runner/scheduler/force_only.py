"""
Force-only scheduler.
Runs force policy standalone with zero vision features (no slow/vision policy needed).
"""
import time
import torch
import numpy as np
from typing import Dict

from logger import logger
from runner.scheduler.vanilla import VanillaScheduler
from runner.configs.scheduler.force_only import ForceOnlySchedulerConfig
from runner.agent.real_agent import RealAgent
from runner.configs.agent import AgentObsKeysConfig
from utils.transforms.interpolation import resample_trajectory


class ForceOnlyScheduler(VanillaScheduler):
    """
    Scheduler that runs force policy alone, injecting zero vision features.
    Inherits VanillaScheduler for the eval loop; overrides model_inference
    to inject a zero vision_feat vector into the observation.
    """

    def __init__(
        self,
        config: ForceOnlySchedulerConfig,
        agent: RealAgent,
        agent_obs_keys: AgentObsKeysConfig,
        device: torch.device
    ) -> None:
        super().__init__(config=config, agent=agent, agent_obs_keys=agent_obs_keys, device=device)
        self.vision_feat_dim = config.vision_feat_dim

    def model_inference(self) -> Dict[str, np.ndarray]:
        with torch.inference_mode():
            obs_raw, time_obs = self.agent.get_obs(self.agent_obs_keys)
            obs_raw["vision_feat"] = np.zeros((1, self.vision_feat_dim), dtype=np.float32)
            obs = self.agent.convert_obs(obs_raw, self.device)

            obs_dict = self.processor(obs, enable_aug=False, process_type="forward")
            action_dict = self.policy_wrapper(obs_dict, action_dict=None, batch_size=1)
            action = self.processor(obs, action_dict, process_type="backward")

            action_raw = self.agent.convert_action(action)
            action_raw = self.agent.to_agent(action_raw)

        synced_action = {}
        aux_action = {}
        for key in action_raw.keys():
            if key in self.config.sync_action_config.keys():
                config = self.config.sync_action_config[key]
                synced_action[key] = resample_trajectory(
                    data=action_raw[key],
                    source_freq=config.source_frequency,
                    source_length=config.source_length,
                    target_freq=self.config.sync_frequency,
                    target_length=self.config.sync_horizon,
                    sampling_method=config.interpolation,
                    rotation_rep=config.rotation_rep,
                    convention=config.convention
                )
            else:
                aux_action[key] = action_raw[key]

        self.last_inference = (obs_raw, synced_action, aux_action, time.time() - time_obs, time_obs)
        return synced_action, aux_action
