from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass(kw_only = True)
class FrameIdentifierBaseConfig:
    name: Literal["force_only", "analytic", "optimization", "two_stage"] = "advanced"
    frame_origin_tcp: bool = True
    

@dataclass(kw_only = True)
class ForceOnlyFrameIdentifierConfig(FrameIdentifierBaseConfig):
    name: str = field(default = "force_only", init = False)
    frame_origin_tcp: bool = field(default = True, init = False)
    thres_parallel: float = 0.98


@dataclass(kw_only = True)
class WrenchOnlyFrameIdentifierConfig(FrameIdentifierBaseConfig):
    name: str = field(default = "wrench_only", init = False)
    frame_origin_tcp: bool = field(default = True, init = False)
    thres_parallel: float = 0.98
    weight_torque: float = 3.0


@dataclass(kw_only = True)
class LinearVelocityOnlyFrameIdentifierConfig(FrameIdentifierBaseConfig):
    name: str = field(default = "linear_velocity_only", init = False)
    frame_origin_tcp: bool = field(default = True, init = False)
    thres_parallel: float = 0.98


@dataclass(kw_only = True)
class TwistOnlyFrameIdentifierConfig(FrameIdentifierBaseConfig):
    name: str = field(default = "twist_only", init = False)
    frame_origin_tcp: bool = field(default = True, init = False)
    thres_parallel: float = 0.98
    weight_angular: float = 0.1


@dataclass(kw_only = True)
class TwistWrenchFrameIdentifierConfig(FrameIdentifierBaseConfig):
    name: str = field(default = "twist_wrench", init = False)
    frame_origin_tcp: bool = field(default = True, init = False)
    specify: Literal["auto", "twist", "wrench"] = "auto"
    thres_force: float = 3.0
    thres_torque: float = 0.5
    thres_lin_vel: float = 0.2
    thres_ang_vel: float = 0.02
    thres_parallel: float = 0.98
    weight_angular: float = 0.1
    weight_torque: float = 3.0


@dataclass(kw_only = True)
class AnalyticFrameIdentifierConfig(FrameIdentifierBaseConfig):
    name: str = field(default = "analytic", init = False)
    

@dataclass(kw_only = True)
class OptimizationFrameIdentifierConfig(FrameIdentifierBaseConfig):
    name: str = field(default = "optimization", init = False)
    num_steps: int = 100
    lr: float = 0.01
    loss_types: List[str] = field(default_factory = lambda: ["general_diagonal"])
    loss_weights: List[float] = field(default_factory = lambda: [1.0])


@dataclass(kw_only = True)
class TwoStageFrameIdentifierConfig(FrameIdentifierBaseConfig):
    name: str = field(default = "two_stage", init = False)
    num_steps: int = 100
    thres_force: float = 3.0
    lr: float = 0.01
    loss_types: List[str] = field(default_factory = lambda: ["general_diagonal"])
    loss_weights: List[float] = field(default_factory = lambda: [1.0])