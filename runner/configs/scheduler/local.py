from typing import Optional
from dataclasses import dataclass, field

from runner.configs.scheduler.base import SchedulerBaseConfig
from runner.configs.scheduler.visualization import VisualizationConfig

@dataclass(kw_only = True)
class LocalSchedulerConfig(SchedulerBaseConfig):
    type: str = field(default = "local_scheduler", init = False)
    camera_name: str = "main"
    visualization: Optional[VisualizationConfig] = None

