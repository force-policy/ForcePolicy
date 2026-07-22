from typing import Literal
from dataclasses import dataclass, field


@dataclass(kw_only = True)
class FrameLabelerBaseConfig:
    name: Literal["vanilla", "advanced"] = "advanced"


@dataclass(kw_only = True)
class VanillaFrameLabelerConfig(FrameLabelerBaseConfig):
    name: str = field(default = "vanilla", init = False)
    thres_force: float = 3.0


@dataclass(kw_only = True)
class AdvancedFrameLabelerConfig(FrameLabelerBaseConfig):
    name: str = field(default = "advanced", init = False)
    thres_ang_vel: float = 0.15
    thres_lin_vel: float = 0.02
    thres_torque: float = 0.5
    thres_force: float = 3.0
    thres_is_parallel: float = 0.8
