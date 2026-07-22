from typing import Union, Tuple, List

import math
import torch
import torch.nn as nn


def make_1d_pos_emb(dim, length, temperature = 10000.0):
    """ Generate 1D grid emb. """
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(temperature) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0) # [1, L, D]


def make_2d_pos_emb(dim, grid_size, temperature = 10000.0):
    """ Generate 2D grid emb. """
    h, w = grid_size
    dim_h = dim // 2
    dim_w = dim - dim_h
    grid_h = make_1d_pos_emb(dim_h, h, temperature).squeeze(0) # [H, D/2]
    grid_w = make_1d_pos_emb(dim_w, w, temperature).squeeze(0) # [W, D/2]
    out_h = grid_h.unsqueeze(1).repeat(1, w, 1) 
    out_w = grid_w.unsqueeze(0).repeat(h, 1, 1) 
    pe = torch.cat([out_h, out_w], dim = -1)
    return pe # [1, H, W, D]


class SinusoidalPositionalEncoding(nn.Module):
    """ Sinusoidal Positional Encoding. Support both 1D and 2D. """
    def __init__(
        self, 
        d_model: int, 
        max_len: Union[int, Tuple[int, int], List[int]] = 5000, 
        temperature: float = 10000.0
    ) -> None:
        """ Initialization. """
        super(SinusoidalPositionalEncoding, self).__init__()
        if isinstance(max_len, int):
            pe = make_1d_pos_emb(d_model, max_len, temperature) # [1, L, D]
        elif isinstance(max_len, (tuple, list)) and len(max_len) == 2:
            pe = make_2d_pos_emb(d_model, max_len, temperature) # [1, H*W, D]
        else:
            raise ValueError("max_len must be int (1D) or tuple.")
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, L, D] or [batch_size, H, W, D]
        """
        if x.dim() == 3:
            return x + self.pe[:, :x.size(1), :]
        elif x.dim() == 4:
            return x + self.pe[:, :x.size(1), :x.size(2), :]
