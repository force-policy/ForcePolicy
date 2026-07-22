"""
Sequential lowdim encoder.

Note: If not conditioned by configs, it can still accept cond, but it does not use it.
"""
from typing import List, Dict, Optional, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from policy.common.film import FiLMLinear, FiLMConv1dBlock
from policy.common.pos_emb import SinusoidalPositionalEncoding
from policy.configs.common.seq_encoder import (
    SeqEncoderConfig, MLPSeqEncoderConfig, GRUSeqEncoderConfig,
    ConvSeqEncoderConfig, TransformerSeqEncoderConfig
)


def _prepare_input(x: torch.Tensor, patch_size: int, num_input: Optional[int] = None) -> torch.Tensor:
    B, L, D = x.shape
    
    if num_input is not None:
        if L < num_input:
            pad_len = num_input - L
            padding = x[:, :1, :].repeat(1, pad_len, 1)
            x = torch.cat([padding, x], dim = 1)
        elif L > num_input:
            x = x[:, -num_input:, :]
        L = num_input

    if patch_size <= 1:
        return x

    rem = L % patch_size
    if rem > 0:
        pad_len = patch_size - rem
        padding = x[:, :1, :].repeat(1, pad_len, 1)
        x = torch.cat([padding, x], dim = 1)
        L = L + pad_len
    x = x.view(B, L // patch_size, patch_size, D).flatten(2)
    return x


class MLPSeqEncoder(nn.Module):
    """ MLP sequence encoder. """
    def __init__(self, config: MLPSeqEncoderConfig) -> None:
        """ Initialization. """
        super(MLPSeqEncoder, self).__init__()
        if config.aggregate and config.patch_size > 1:
            raise ValueError("MLPSeqEncoder: aggregate = True is only supported when patch_size = 1 (taking all tokens as flat input).")

        self.config = config   
        self.net = nn.ModuleList()
        dim_cur = 0
        if config.aggregate:
            if config.num_input is None:
                 raise ValueError("MLPSeqEncoder with aggregate=True requires num_input (fixed length) to be set.")
            dim_cur = config.dim_input * config.num_input
        else:
            dim_cur = config.dim_input * config.patch_size
        
        for dim_next in config.dim_hidden:
            if config.dim_cond is not None:
                self.net.append(FiLMLinear(dim_cur, dim_next, config.dim_cond))
            else:
                self.net.append(nn.Sequential(
                    nn.Linear(dim_cur, dim_next),
                    nn.LayerNorm(dim_next),
                    nn.Mish()
                ))
            dim_cur = dim_next

    def forward(
        self, 
        x: torch.Tensor, 
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: [batch_size, L, D]
        cond: [batch_size, D_cond]

        return: [batch_size, D_hidden]
        """
        if self.config.dim_cond is not None:
            assert cond is not None

        x = _prepare_input(x, self.config.patch_size, self.config.num_input)
        if self.config.aggregate:
            x = rearrange(x, "b n c -> b (n c)")
        
        for layer in self.net:
            if self.config.dim_cond is not None:
                x = layer(x, cond)
            else:
                x = layer(x)
                
        return x 


class GRUSeqEncoder(nn.Module):
    """ GRU sequence encoder. """
    def __init__(self, config: GRUSeqEncoderConfig) -> None:
        """ Initialization. """
        super(GRUSeqEncoder, self).__init__()
        self.config = config
        
        if config.dim_cond is not None:
            self.input_proj = FiLMLinear(config.dim_input * config.patch_size, config.dim_hidden, config.dim_cond)
            self.out_proj = FiLMLinear(config.dim_hidden, config.dim_hidden, config.dim_cond)
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(config.dim_input * config.patch_size, config.dim_hidden),
                nn.LayerNorm(config.dim_hidden),
                nn.Mish()
            )
            self.out_proj = nn.Sequential(
                nn.Linear(config.dim_hidden, config.dim_hidden),
                nn.LayerNorm(config.dim_hidden),
                nn.Mish()
            )
            
        self.gru = nn.GRU(
            config.dim_hidden, 
            config.dim_hidden, 
            num_layers = config.num_layers, 
            batch_first = True, 
            dropout = config.p_dropout
        )

    def forward(
        self, 
        x: torch.Tensor, 
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: [batch_size, L, D]
        cond: [batch_size, D_cond]

        return: [batch_size, L, D] or [batch_size, D]
        """
        if self.config.dim_cond is not None:
            assert cond is not None
        
        x = _prepare_input(x, self.config.patch_size, self.config.num_input)
            
        if self.config.dim_cond is not None:
            x = self.input_proj(x, cond)
        else:
            x = self.input_proj(x)
        
        gru_out, _ = self.gru(x)
        
        if self.config.dim_cond is not None:
            out = self.out_proj(gru_out, cond)
        else:
            out = self.out_proj(gru_out)
            
        if self.config.aggregate:
            return out[:, -1, :] # [B, D]
        else:
            return out # [B, L', D]


class Conv1dBlock(nn.Module):
    """ Conv1d block. """
    def __init__(
        self, 
        dim_input: int,
        dim_output: int,
        kernel_size: int = 3,
        num_groups: int = 8
    ):
        """ Initialization. """
        super().__init__()
        self.conv = nn.Conv1d(dim_input, dim_output, kernel_size, padding = kernel_size // 2, bias = False)
        self.norm = nn.GroupNorm(num_groups, dim_output)
        self.act = nn.Mish()
        
        self.shortcut = nn.Sequential()
        if dim_input != dim_output:
            self.shortcut = nn.Conv1d(dim_input, dim_output, 1)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out + self.shortcut(x)


class ConvSeqEncoder(nn.Module):
    """ Conv sequence encoder. """
    def __init__(self, config: ConvSeqEncoderConfig) -> None:
        """ Initialization. """
        super(ConvSeqEncoder, self).__init__()
        self.config = config
        
        self.input_proj = nn.Conv1d(config.dim_input * config.patch_size, config.dim_hidden[0], 1)
        self.layers = nn.ModuleList()
        dim_cur = config.dim_hidden[0]
        for dim_next in config.dim_hidden:
            if config.dim_cond is not None:
                self.layers.append(FiLMConv1dBlock(
                    dim_cur, dim_next, kernel_size = config.kernel_size, 
                    dim_cond = config.dim_cond, num_groups = config.num_groups
                ))
            else:
                self.layers.append(Conv1dBlock(
                    dim_cur, dim_next, kernel_size = config.kernel_size, 
                    num_groups = config.num_groups
                ))
            dim_cur = dim_next

    def forward(
        self, 
        x: torch.Tensor, 
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: [batch_size, L, D]
        cond: [batch_size, D_cond]

        return: [batch_size, L, D] or [batch_size, D]
        """
        if self.config.dim_cond is not None:
            assert cond is not None
        
        x = _prepare_input(x, self.config.patch_size, self.config.num_input)
            
        x = x.permute(0, 2, 1) # [batch_size, D, L]
        x = self.input_proj(x)
        for layer in self.layers:
            if self.config.dim_cond is not None:
                x = layer(x, cond)
            else:
                x = layer(x)
        
        if self.config.aggregate:
            x = torch.mean(x, dim = -1) # [B, D_hidden]
        else:
            x = x.permute(0, 2, 1) # [batch_size, L, D_hidden]
        return x


class TransformerSeqEncoder(nn.Module):
    """ Transformer sequence encoder. """
    def __init__(self, config: TransformerSeqEncoderConfig) -> None:
        super(TransformerSeqEncoder, self).__init__()
        self.config = config
        
        if config.dim_cond is not None:
            self.input_proj = FiLMLinear(config.dim_input * config.patch_size, config.dim_hidden, config.dim_cond)
        else:
            self.input_proj = nn.Linear(config.dim_input * config.patch_size, config.dim_hidden)
        
        self.pos_enc = SinusoidalPositionalEncoding(config.dim_hidden)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = config.dim_hidden, 
            nhead = config.num_heads, 
            dim_feedforward = config.dim_feedforward, 
            dropout = config.p_dropout, 
            activation = config.activation,
            batch_first = True, 
            norm_first = True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers = config.num_layers) 
        
        if config.dim_cond is not None:
            self.output_proj = FiLMLinear(config.dim_hidden, config.dim_hidden, config.dim_cond)
        else:
            self.output_proj = nn.Linear(config.dim_hidden, config.dim_hidden)

    def forward(
        self, 
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: [batch_size, L, D]
        cond: [batch_size, D_cond]

        return: [batch_size, L, D]
        """
        if self.config.dim_cond is not None:
            assert cond is not None
        
        x = _prepare_input(x, self.config.patch_size, self.config.num_input)
            
        if self.config.dim_cond is not None:
            x = self.input_proj(x, cond)
        else:
            x = self.input_proj(x)
        
        x = self.pos_enc(x)
        x = self.transformer(x)

        if self.config.dim_cond is not None:
            x = self.output_proj(x, cond)
        else:
            x = self.output_proj(x)
            
        if self.config.aggregate:
            x = torch.mean(x, dim = 1)
        
        return x


def get_seq_encoder(config: SeqEncoderConfig):
    if config.type == "mlp":
        return MLPSeqEncoder(config)
    elif config.type == "gru":
        return GRUSeqEncoder(config)
    elif config.type == "conv":
        return ConvSeqEncoder(config)
    elif config.type == "transformer":
        return TransformerSeqEncoder(config)
    else:
        raise ValueError(f"Unsupported seq encoder config type: {config.type}")
