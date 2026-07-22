from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from data_infra.configs import DatasetConfig
from data_infra.configs import DataProcessorConfig
from policy.configs.policy_base import PolicyConfig
from policy.configs.wrapper import PolicyWrapperConfig


@dataclass(kw_only = True)
class OptimizerConfig:
    type: str = "AdamW"
    kwargs: Dict[str, Any] = field(default_factory = dict)


@dataclass(kw_only = True)
class SchedulerConfig:
    type: str = "cosine"
    kwargs: Dict[str, Any] = field(default_factory = dict)


@dataclass(kw_only = True)
class TrainerConfig:
    seed: int
    dataset_config: Optional[DatasetConfig] = None
    processor_config: Optional[DataProcessorConfig] = None
    policy_config: Optional[PolicyConfig] = None
    policy_wrapper_config: Optional[PolicyWrapperConfig] = None
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig
    num_steps: int
    save_steps: int
    lr: float
    batch_size: int
    num_workers: int
    ckpt_dir: Optional[str] = None
    resume_ckpt: Optional[str] = None
    env_vars: Dict[str, str] = field(default_factory = dict)
