from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass(kw_only = True)
class TransformerConfig:
    num_heads: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float

