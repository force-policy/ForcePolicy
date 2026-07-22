from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from data_infra.configs import DatasetConfig
from data_infra.configs import DataProcessorConfig
from policy.configs.policy_base import PolicyConfig
from policy.configs.wrapper import PolicyWrapperConfig


@dataclass(kw_only = True)
class VisionFeatGeneratorConfig:
    seed: int
    dataset_config: DatasetConfig
    processor_config: DataProcessorConfig
    policy_config: PolicyConfig
    policy_wrapper_config: PolicyWrapperConfig
    
    batch_size: int = 1
    num_workers: int = 8
    ckpt_path: str

    file_prefix: str = "vision_feat"
    vision_feat_key: str = "vision_feat"
    
    env_vars: Dict[str, str] = field(default_factory = dict)
