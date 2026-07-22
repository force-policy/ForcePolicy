"""
RISE-2 configurations.
"""
from typing import Any, Dict, Literal
from dataclasses import dataclass, field

import torch

from policy.configs import PolicyConfig
from policy.configs.common import TransformerConfig
from policy.configs.common import ActionHeadConfig


@dataclass(kw_only = True)
class DenseEncoderConfig:
    name: str
    dim_dense_feat: int
    dim_sparse_feat: int
    finetune: Literal["full", "lora", "none"]
    dtype: torch.dtype = torch.float32
    interp_fn_mode: Literal["naive", "custom"] = "custom"
    params: Dict[str, Any] = field(default_factory = dict)


@dataclass(kw_only = True)
class RISE2Config(PolicyConfig):
    name: str = field(default = "RISE2", init = False)
    dim_hidden: int
    action_head: ActionHeadConfig
    backbone: TransformerConfig
    dense_encoder: DenseEncoderConfig
    disable_pcd_color: bool = False
