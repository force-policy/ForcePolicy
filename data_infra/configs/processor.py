"""
Data Processor configurations.
"""
from typing import Dict, Optional

from dataclasses import dataclass, field

from data_infra.configs.base import *
from data_infra.configs.data import *
from data_infra.configs.augmentation import *
from data_infra.configs.data_inference import *


@dataclass(kw_only = True)
class DataProcessorConfig:
    obs_data_configs: Dict[str, BaseTargetConfig]
    action_data_configs: Dict[str, BaseTargetConfig]
    prefix_config: PrefixConfig = field(default_factory = PrefixConfig)
    augmentation_configs: Dict[str, BaseAugmentationConfig] = field(default_factory = dict)
    action_data_reverse_configs: Dict[str, BaseInferenceConfig] = field(default_factory = dict)
    spatial_aug_frame: Optional[str] = None
