from typing import Any, List, Tuple, Union, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from policy.common.pos_emb import make_1d_pos_emb, make_2d_pos_emb


class GatedFusion(nn.Module):
    """
    Gated Multimodal Unit for multiple sources.
    z = Softmax(Linear(cat([src1, src2, ...])))
    h = sum(z_i * src_i)
    """
    def __init__(
        self,
        dim_inputs: List[int],
        dim_hidden: int,
    ) -> None:
        super(GatedFusion, self).__init__()
        self.projections = nn.ModuleList([ 
            nn.Sequential(
                nn.Linear(dim, dim_hidden), 
                nn.LayerNorm(dim_hidden), 
                nn.GELU()
            ) for dim in dim_inputs 
        ])
        
        self.num_sources = len(dim_inputs)
        self.dim_hidden = dim_hidden
        self.gate_net = nn.Sequential(
            nn.Linear(dim_hidden * self.num_sources, dim_hidden),
            nn.LayerNorm(dim_hidden),
            nn.GELU(),
            nn.Linear(dim_hidden, self.num_sources)
        )
        self.norm = nn.LayerNorm(dim_hidden)

    def forward(self, inputs: List[torch.Tensor]):
        """
        inputs: List of [B, dim_i]
        """
        projected_feats = []
        for i, x in enumerate(inputs):
            projected_feats.append(self.projections[i](x))
        stacked_feats = torch.stack(projected_feats, dim = 1) # [B, N, dim_hidden]
        
        cat_feat = torch.cat(projected_feats, dim = -1) # [B, N * dim_hidden] 
        gates = self.gate_net(cat_feat) # [B, N]
        gates = F.softmax(gates, dim = 1) # [B, N]
        gates = gates.unsqueeze(-1) # [B, N, 1]
        
        out = torch.sum(gates * stacked_feats, dim = 1) # [B, dim_hidden]
        return self.norm(out)


class CrossAttnFusion(nn.Module):
    """
    Cross Attention Fusion for multiple sources.
    """
    def __init__(
        self,
        dim_inputs: List[int],
        dim_query: int,
        dim_hidden: int,
        num_heads: int,
        dim_feedforward: int,
        p_dropout: float = 0.1,
        type_emb_enabled: bool = True,
        pos_emb_enabled: Optional[List[bool]] = None
    ) -> None:
        super(CrossAttnFusion, self).__init__()
        self.dim_hidden = dim_hidden

        self.type_emb_enabled = type_emb_enabled
        if pos_emb_enabled is None:
            self.pos_emb_enabled = [True] * len(dim_inputs)
        else:
            self.pos_emb_enabled = pos_emb_enabled
            assert len(pos_emb_enabled) == len(dim_inputs)
        
        self.q_proj = nn.Linear(dim_query, dim_hidden)
        
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim_hidden) if dim != dim_hidden else nn.Identity(),
                nn.LayerNorm(dim_hidden)
            ) for dim in dim_inputs
        ])
        
        self.type_emb_enabled = type_emb_enabled
        if type_emb_enabled:
            self.type_embs = nn.ParameterList([
                nn.Parameter(torch.randn(1, 1, dim_hidden) * 0.02)
                for _ in dim_inputs
            ])

        self.attn = nn.MultiheadAttention(dim_hidden, num_heads, dropout = p_dropout, batch_first = True)

        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norm2 = nn.LayerNorm(dim_hidden)

        self.ffn = nn.Sequential(
            nn.Linear(dim_hidden, dim_feedforward),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(dim_feedforward, dim_hidden)
        )
        
        self.dropout = nn.Dropout(p_dropout)

    def _prepare_input(self, i: int, x: torch.Tensor) -> torch.Tensor:
        """ Prepare input (with pos emb). """
        feat = self.projections[i](x)
        pe = None
        
        if feat.dim() == 4:
            B, H, W, C = feat.shape
            if self.pos_emb_enabled[i]:
                 pe = make_2d_pos_emb(C, (H, W)) # [1, H, W, C]
                 pe = pe.to(dtype = feat.dtype, device = feat.device)
            feat = feat.flatten(1, 2)
        
        elif feat.dim() == 3:
            B, N, C = feat.shape
            if self.pos_emb_enabled[i]:
                 pe = make_1d_pos_emb(C, N) # [1, N, C]
                 pe = pe.to(dtype = feat.dtype, device = feat.device)

        elif feat.dim() == 2:
            feat = feat.unsqueeze(1)
            
        if pe is not None:
            feat = feat + pe
        if self.type_emb_enabled:
            feat = feat + self.type_embs[i]
        return feat

    def forward(
        self, 
        inputs: List[torch.Tensor], 
        query: torch.Tensor, 
        return_weights = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        inputs: List of tensors. 
            - [B, H, W, Di] (2D feature map)
            - [B, N, Di] (1D sequence)
            - [B, Di] (vector)
        query: [batch_size, dim_query]
        Returns: fused_feature, attn_weights
        """
        query_proj = self.q_proj(query).unsqueeze(1) # [B, 1, D]

        kv_tokens_list = [self._prepare_input(i, x) for i, x in enumerate(inputs)]
        kv_tokens = torch.cat(kv_tokens_list, dim = 1) # [B, Total_N, dim_hidden]

        query_norm = self.norm1(query_proj)
        
        attn_out, weights = self.attn(query_norm, kv_tokens, kv_tokens)

        x = query_proj + self.dropout(attn_out)
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        out = x + self.dropout(ffn_out)

        if return_weights:
            return out.squeeze(1), weights
        else:
            return out.squeeze(1)
