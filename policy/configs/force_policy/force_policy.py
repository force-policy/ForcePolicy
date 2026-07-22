from typing import Any, List, Dict, Union, Literal, Optional
from dataclasses import dataclass, field

from policy.configs.policy_base import PolicyConfig
from policy.configs.common.seq_encoder import (
    SeqEncoderConfig, GRUSeqEncoderConfig, ConvSeqEncoderConfig, 
    TransformerSeqEncoderConfig, MLPSeqEncoderConfig
)
from policy.configs.common.action_head import ActionHeadConfig, MLPHeadConfig


@dataclass(kw_only = True)
class VisionEncoderConfig:
    type: Literal['resnet'] = 'resnet'
    use_film: bool = True
    dim_resnet: List[int] = field(default_factory = lambda: [64, 128, 256])
    kernel_resnet: List[int] = field(default_factory = lambda: [3, 3, 3])
    stride_resnet: List[int] = field(default_factory = lambda: [1, 2, 2])
    return_tokens: bool = False # If True, returns [B, N, C]. Else [B, C].


@dataclass(kw_only = True)
class FusionConfig:
    type: Literal['gated', 'cross_attn'] = 'gated'
    dim_hidden: int = 256
    # CrossAttn params
    dim_query: int = 256 # Usually same as global_emb
    num_heads: int = 4
    dim_feedforward: int = 1024
    p_dropout: float = 0.1
    pos_emb_enabled: Optional[List[bool]] = None
    type_emb_enabled: bool = True
    

@dataclass(kw_only = True)
class AugForceTorqueConfig:
    enable: bool = False
    std: float = 0.1
    prob: float = 0.2


@dataclass(kw_only = True)
class ForcePolicyConfig(PolicyConfig):
    name: str = field(default = "ForcePolicy", init = False)
    # Basics
    dim_vision_feat: Optional[int] = None
    dim_pred_action: int
    dim_pred_frame: int
    dim_pred_frame_force: int
    dim_pred_frame_mask: int
    mask_threshold: float = 0.8

    # Encoders
    vision_encoders: List[VisionEncoderConfig]
    camera_names: List[str] = field(default_factory=list)
    
    # Lowdim encoders
    separate_lowdim_encoders: bool = False
    lowdim_encoder: Union[SeqEncoderConfig, List[SeqEncoderConfig]]
    
    # Fusion
    fusion: FusionConfig
    fuse_global_vision_feat: bool = False
    
    # Heads
    one_step_reference_gt: bool = False
    pre_contact_steps: int = 20
    pre_release_steps: int = 20
    release_rho: float = 0.7

    separate_action_decoders: bool = False
    action_head: Union[ActionHeadConfig, List[ActionHeadConfig]]
    router_lookahead_steps: int = 0
    
    # Augmentation
    aug_force_torque: AugForceTorqueConfig = field(default_factory = AugForceTorqueConfig)
    
    # Loss weights
    loss_weights: Dict[str, float]
