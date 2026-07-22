"""
Local scheduler.
"""
import torch

from runner.scheduler.base import SchedulerBase
from runner.agent.local_agent import LocalAgent
from runner.configs.agent import AgentObsKeysConfig
from runner.configs.scheduler import LocalSchedulerConfig


class LocalScheduler(SchedulerBase):
    """
    Local scheduler for visualization.
    """
    def __init__(
        self,
        config: LocalSchedulerConfig,
        agent: LocalAgent,
        agent_obs_keys: AgentObsKeysConfig,
        device: torch.device
    ) -> None:
        """ Initialization """
        super(LocalScheduler, self).__init__(
            config = config, 
            agent = agent,
            agent_obs_keys = agent_obs_keys,
            device = device
        )
