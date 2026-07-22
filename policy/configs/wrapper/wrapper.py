"""
Policy wrapper configurations.
"""

from typing import Any, List, Dict, Literal, Optional

from dataclasses import dataclass, field


@dataclass(kw_only = True)
class ReorganizeConfig:
    type: Literal["pack", "unpack"]
    key: str
    key_list: List[str] = field(default_factory = list)
    dim_list: List[int] = field(default_factory = list)


@dataclass(kw_only = True)
class DataProviderConfig:
    reorganize_configs: List[ReorganizeConfig] = field(default_factory = list)
    provider_keys: Dict[str, str] = field(default_factory = dict)


@dataclass(kw_only = True)
class PolicyWrapperConfig:
    obs_provider: DataProviderConfig
    train_action_provider: DataProviderConfig
    inference_action_provider: DataProviderConfig
