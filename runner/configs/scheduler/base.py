from typing import Any, Dict, List, Literal, Optional, Union
from dataclasses import dataclass, field

from data_infra.configs import DataProcessorConfig

from policy.configs import PolicyConfig
from policy.configs.wrapper import PolicyWrapperConfig 

from utils.transforms.rotation import RotationType


@dataclass(kw_only = True)
class SyncActionConfig:
    source_frequency: int = 50
    source_length: Optional[int] = None 
    interpolation: Literal["linear", "nearest", "pose_linear"] = "linear"
    rotation_rep: Optional[RotationType] = None
    convention: Optional[str] = None
    

@dataclass(kw_only = True)
class SchedulerBaseConfig:
    type: Literal["local_scheduler", "vanilla_scheduler", "adaptive_scheduler", "force_only_scheduler"]
    seed: int

    processor_config: Optional[DataProcessorConfig] = None
    policy_config: Optional[PolicyConfig] = None
    policy_wrapper_config: Optional[PolicyWrapperConfig] = None
    ckpt_path: Optional[str] = None

    sync_action_config: Dict[str, SyncActionConfig] = field(default_factory = dict)
    sync_frequency: int
    sync_horizon: int
    
    action_params: Dict[str, Dict[str, Any]] = field(default_factory = dict)

    env_vars: Dict[str, str] = field(default_factory = dict)
