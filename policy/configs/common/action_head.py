

from typing import Optional, Tuple, Any, Dict

from enum import Enum
from dataclasses import dataclass, field


class ActionHeadType(Enum):
    MLP = "mlp"
    GRU = "gru"
    DIFFUSION = "diffusion"
    MIP = "mip"


@dataclass(kw_only = True)
class ActionHeadConfig:
    type: ActionHeadType
    dim_action: int
    num_action: int


@dataclass(kw_only = True)
class MLPHeadConfig(ActionHeadConfig):
    type: ActionHeadType = field(default = ActionHeadType.MLP, init = False)
    dim_hidden: Tuple[int, ...] = ()
    loss_type: str = "l1_loss"


@dataclass(kw_only = True)
class GRUHeadConfig(ActionHeadConfig):
    type: ActionHeadType = field(default = ActionHeadType.GRU, init = False)
    dim_hidden: int
    num_layers: int
    dim_latent: Optional[int] = None
    norm_before_hidden: bool = True
    norm_before_input: bool = True
    loss_type: str = "l1_loss"


@dataclass(kw_only = True)
class DiffusionHeadConfig(ActionHeadConfig):
    type: ActionHeadType = field(default = ActionHeadType.DIFFUSION, init = False)
    num_inference_steps: Optional[int] = 20
    diffusion_step_embed_dim: int = 256
    down_dims: Tuple[int, ...] = (256, 512)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    clip_sample: bool = True
    noise_scheduler_params: Dict[str, Any] = field(default_factory = dict)


@dataclass(kw_only = True)
class MIPHeadConfig(ActionHeadConfig):    
    type: ActionHeadType = field(default = ActionHeadType.MIP, init = False)
    diffusion_step_embed_dim: int = 256
    down_dims: Tuple[int, ...] = (256, 512)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    t_star: float = 0.9
    fixed_scale_noising: bool = True
