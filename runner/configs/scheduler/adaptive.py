
from typing import Any, Dict, List, Literal, Optional, Union
from dataclasses import dataclass, field

from data_infra.configs.point_mask import *

from runner.configs.scheduler.base import SchedulerBaseConfig


@dataclass(kw_only = True)
class WaypointDropoutConfig:
    """DTW-based waypoint dropout configuration"""
    enable: bool = True
    max_dropout: int = 12  # Maximum number of waypoints that can be dropped
    weight_linear: float = 1.0
    weight_angular: float = 0.04
    weight_linear_vel: float = 1.0
    weight_angular_vel: float = 0.0


@dataclass(kw_only = True)
class AdaptiveSchedulerConfig(SchedulerBaseConfig):
    type: str = field(default = "adaptive_scheduler", init = False)
    planner_freq: int = 50
    max_infer_dt: float = 1.5
    pre_infer_time: float = 0.2

    waypoint_dropout: WaypointDropoutConfig = field(default_factory = WaypointDropoutConfig)

    waypoint_interp_type: Literal["velocity_continuous", "acceleration_continuous", "linear"] = "linear"
    time_scaling_factor: float = 1.0
    