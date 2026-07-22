from typing import Optional, Literal, Union, List
from dataclasses import dataclass, field


@dataclass(kw_only = True)
class SeqEncoderConfig:
    type: Literal['mlp', 'gru', 'conv', 'transformer']
    dim_input: int
    dim_cond: Optional[int] = None
    dim_hidden: Union[int, List[int]]
    patch_size: int = 1
    aggregate: bool = False
    num_input: Optional[int] = None # If set, pad/truncate to this length


@dataclass(kw_only = True)
class MLPSeqEncoderConfig(SeqEncoderConfig):
    type: Literal['mlp', 'gru', 'conv', 'transformer'] = field(default = 'mlp', init = False)
    dim_hidden: List[int]


@dataclass(kw_only = True)
class GRUSeqEncoderConfig(SeqEncoderConfig):
    type: Literal['mlp', 'gru', 'conv', 'transformer'] = field(default = 'gru', init = False)
    dim_hidden: int
    num_layers: int
    p_dropout: float = 0.1


@dataclass(kw_only = True)
class ConvSeqEncoderConfig(SeqEncoderConfig):
    type: Literal['mlp', 'gru', 'conv', 'transformer'] = field(default = 'conv', init = False)
    dim_hidden: List[int]
    kernel_size: int
    num_groups: int
    
    
@dataclass(kw_only = True)
class TransformerSeqEncoderConfig(SeqEncoderConfig):
    type: Literal['mlp', 'gru', 'conv', 'transformer'] = field(default = 'transformer', init = False)
    dim_hidden: int
    num_heads: int
    num_layers: int
    dim_feedforward: int
    p_dropout: float = 0.1
    activation: str = "gelu"
