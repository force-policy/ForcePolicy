import torch
from runner.agent import BaseAgent
from runner.configs.agent import AgentObsKeysConfig
from runner.configs.scheduler import *


def build_scheduler(
    config: SchedulerBaseConfig,
    agent: BaseAgent,
    agent_obs_keys: AgentObsKeysConfig,
    device: torch.device,
    **kwargs
):
    """ Build scheduler from config. """
    if config.type == "local_scheduler":
        from runner.scheduler.local import LocalScheduler
        return LocalScheduler(
            config = config, 
            agent = agent, 
            agent_obs_keys = agent_obs_keys, 
            device = device
        )
    elif config.type == "vanilla_scheduler":
        from runner.scheduler.vanilla import VanillaScheduler
        return VanillaScheduler(
            config = config, 
            agent = agent, 
            agent_obs_keys = agent_obs_keys, 
            device = device
        )
    elif config.type == "adaptive_scheduler":
        from runner.scheduler.adaptive import AdaptiveScheduler
        return AdaptiveScheduler(
            config = config, 
            agent = agent, 
            agent_obs_keys = agent_obs_keys, 
            device = device
        )
    elif config.type == "slow_fast_scheduler":
        from runner.scheduler.slow_fast import SlowFastScheduler
        assert config.fast_processor_config is not None, "Fast processor config must be provided for slow_fast_scheduler."
        assert config.fast_policy_config is not None, "Fast policy config must be provided for slow_fast_scheduler."
        assert config.fast_policy_wrapper_config is not None, "Fast policy wrapper config must be provided for slow_fast_scheduler."
        assert config.fast_ckpt_path is not None, "Fast policy checkpoint path must be provided for slow_fast_scheduler."
        assert "fast_agent_obs_keys" in kwargs, "Fast agent observation keys must be provided for slow_fast_scheduler."
        return SlowFastScheduler(
            config = config, 
            agent = agent, 
            agent_obs_keys = agent_obs_keys, 
            fast_agent_obs_keys = kwargs["fast_agent_obs_keys"],
            device = device
        )
    elif config.type == "force_only_scheduler":
        from runner.scheduler.force_only import ForceOnlyScheduler
        return ForceOnlyScheduler(
            config = config,
            agent = agent,
            agent_obs_keys = agent_obs_keys,
            device = device
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.type}")
