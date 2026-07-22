from typing import Any, Dict, List, Union, Optional
from dataclasses import dataclass, field
from adaptor.configs.interaction_frame import FrameIdentifierBaseConfig, FrameLabelerBaseConfig


@dataclass(kw_only = True)
class DataAdaptorConfig:
    patch_size: int
    frame_identifier_config: FrameIdentifierBaseConfig
    frame_labeler_config: FrameLabelerBaseConfig
    calc_twist_from_pose: bool = False
    # Smoothing parameters for temporal consistency
    enable_smoothing: bool = True
    min_run_length: int = 10        # Runs shorter than this may be absorbed
    conf_threshold: float = 0.5     # Confidence below this may be absorbed  
    sim_threshold: float = 0.8      # Frame similarity threshold for boundary protection
    sim_window_size: int = 5        # Number of patches at run boundaries for frame averaging
