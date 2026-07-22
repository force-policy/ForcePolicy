"""
RISE configurations.
"""
from typing import Any, Dict
from dataclasses import dataclass, field

from policy.configs import PolicyConfig
from policy.configs.common import TransformerConfig
from policy.configs.common import ActionHeadConfig


@dataclass(kw_only=True)
class RISEConfig(PolicyConfig):
    name: str = field(default = "RISE", init = False)
    dim_hidden: int
    backbone: TransformerConfig
    disable_pcd_color: bool = False
    action_head: ActionHeadConfig
