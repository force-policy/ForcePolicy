"""
Diffusion policy configurations.
"""
from typing import Any, Dict, Tuple
from dataclasses import dataclass, field

from policy.configs import PolicyConfig
from policy.configs.common import ActionHeadConfig


@dataclass(kw_only = True)
class DiffusionPolicyVisionConfig:
    name: str
    dim_feat: int
    img_size: Tuple[int, int]
    num_images: int
    unified_encoder: bool = False
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class DiffusionPolicyCNNConfig(PolicyConfig):
    name: str = field(default = "diffusion_policy_cnn", init = False)
    action_head: ActionHeadConfig
    vision: DiffusionPolicyVisionConfig
