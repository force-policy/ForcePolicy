from adaptor.configs.interaction_frame import  *
from adaptor.interaction_frame.frame_labeler import *
from adaptor.interaction_frame.frame_identifier import *

def get_frame_identifier(config: FrameIdentifierBaseConfig):
    if config.name == "force_only":
        return ForceOnlyFrameIdentifier(config)
    elif config.name == "wrench_only":
        return WrenchOnlyFrameIdentifier(config)
    elif config.name == "linear_velocity_only":
        return LinearVelocityOnlyFrameIdentifier(config)
    elif config.name == "twist_only":
        return TwistOnlyFrameIdentifier(config)
    elif config.name == "twist_wrench":
        return TwistWrenchFrameIdentifier(config)
    elif config.name == "analytic":
        return AnalyticFrameIdentifier(config)
    elif config.name == "optimization":
        return OptimizationFrameIdentifier(config)
    elif config.name == "two_stage":
        return TwoStageFrameIdentifier(config)
    else:
        raise ValueError(f"Unknown interaction frame identifier: {config.name}")


def get_frame_labeler(config: FrameLabelerBaseConfig):
    if config.name == "vanilla":
        return VanillaFrameLabeler(config)
    elif config.name == "advanced":
        return AdvancedFrameLabeler(config)
    else:
        raise ValueError(f"Unknown interaction frame labeler: {config.name}")
