from typing import Any, Dict, List, Literal, Optional, Union
from dataclasses import dataclass, field

from data_infra.configs.point_mask import *
from data_infra.configs import DataProcessorConfig

from policy.configs import PolicyConfig
from policy.configs.wrapper import PolicyWrapperConfig 

from runner.configs.scheduler.base import SchedulerBaseConfig
from runner.configs.scheduler.visualization import VisualizationConfig
from runner.configs.ensemble_buffer import EnsembleBufferConfig

from utils.transforms.rotation import RotationType


@dataclass(kw_only = True)
class VanillaSchedulerConfig(SchedulerBaseConfig):
    type: str = field(default = "vanilla_scheduler", init = False)
    frequency: int
    num_inference_step: int
    max_steps: int

    ensemble_buffer_config: EnsembleBufferConfig
    visualization: Optional[VisualizationConfig] = None
