from typing import Any, Dict, List, Literal, Optional, Union
from dataclasses import dataclass, field

from data_infra.configs import DataProcessorConfig

from policy.configs import PolicyConfig
from policy.configs.wrapper import PolicyWrapperConfig 

from runner.configs.scheduler.base import SyncActionConfig
from runner.configs.scheduler.adaptive import AdaptiveSchedulerConfig


@dataclass(kw_only = True)
class SlowFastSchedulerConfig(AdaptiveSchedulerConfig):
    type: str = field(default = "slow_fast_scheduler", init = False)
    
    # Fast policy configuration
    fast_processor_config: Optional[DataProcessorConfig] = None
    fast_policy_config: Optional[PolicyConfig] = None
    fast_policy_wrapper_config: Optional[PolicyWrapperConfig] = None
    fast_ckpt_path: Optional[str] = None
    fast_slow_key_mapping: Dict[str, str] = field(default_factory = dict)
    fast_sync_action_config: Dict[str, SyncActionConfig] = field(default_factory = dict)
    fast_sync_horizon: int
    fast_frequency: float

    # mode switching
    switch_duration: float
    switch_key: str = "switch_signal"
    