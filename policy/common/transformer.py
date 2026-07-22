# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):
    """
    Transformer
    """
    def __init__(
        self, 
        dim_model = 512, 
        num_heads = 8, 
        num_encoder_layers = 6,
        num_decoder_layers = 6, 
        dim_feedforward = 2048, 
        dropout = 0.1,
        activation = "relu", 
        normalize_before = False,
        return_intermediate_dec = False
    ):
        super(Transformer, self).__init__()

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.return_intermediate_dec = return_intermediate_dec

        # Initialize encoder if num_encoder_layers > 0
        if num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(
                dim_model = dim_model,
                num_heads = num_heads,
                dim_feedforward = dim_feedforward,
                dropout = dropout,
                activation = activation,
                normalize_before = normalize_before
            )
            encoder_norm = nn.LayerNorm(dim_model) if normalize_before else None
            self.encoder = TransformerEncoder(
                encoder_layer, 
                num_layers = num_encoder_layers,
                norm = encoder_norm
            )
        else:
            self.encoder = None

        # Initialize decoder
        decoder_layer = TransformerDecoderLayer(
            dim_model = dim_model,
            num_heads = num_heads,
            dim_feedforward = dim_feedforward,
            dropout = dropout,
            activation = activation,
            normalize_before = normalize_before,
            tgt_self_attn = (num_encoder_layers > 0)  # if not decoder-only, use self-attention
        )
        decoder_norm = nn.LayerNorm(dim_model)
        self.decoder = TransformerDecoder(
            decoder_layer, 
            num_layers = num_decoder_layers, 
            norm = decoder_norm,
            return_intermediate = return_intermediate_dec
        )

        self.readout_emb = nn.Embedding(1, dim_model)

        self._reset_parameters()

        self.dim_model = dim_model
        self.num_heads = num_heads

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        attn_mask = None,
        padding_mask = None, 
        pos_emb = None
    ):
        assert len(src.shape) == 3
        
        if self.num_encoder_layers > 0:
            return self.forward_encoder_decoder(src, attn_mask, padding_mask, pos_emb)
        else:
            return self.forward_decoder_only(src, attn_mask, padding_mask, pos_emb)

    def forward_encoder_decoder(
        self, 
        src,
        attn_mask = None,
        padding_mask = None, 
        pos_emb = None
    ):
        """
        Forward pass for encoder-decoder mode.
        """
        bs = src.shape[0]
        src = src.permute(1, 0, 2)
        if pos_emb is not None:
            pos_emb = pos_emb.permute(1, 0, 2)
        readout_emb = self.readout_emb.weight.unsqueeze(1).repeat(1, bs, 1)

        readout_token = torch.zeros_like(readout_emb)
        memory = self.encoder(
            src, 
            attn_mask = attn_mask,
            padding_mask = padding_mask, 
            pos_emb = pos_emb
        )
        res = self.decoder(
            readout_token, 
            memory, 
            memory_attn_mask = attn_mask,
            memory_padding_mask = padding_mask,
            pos_emb = readout_emb, 
            memory_pos_emb = pos_emb
        )
        return res[0]
    
    def forward_decoder_only(
        self, 
        src, 
        attn_mask = None, 
        padding_mask = None, 
        pos_emb = None
    ):
        """
        Forward pass for decoder-only mode.
        """
        batch_size, len_src = src.size(0), src.size(1)

        src = src.permute(1, 0, 2)
        if pos_emb is not None:
            pos_emb = pos_emb.permute(1, 0, 2)
        else:
            pos_emb = torch.zeros_like(src)
        if padding_mask is None:
            padding_mask = torch.zeros([batch_size, len_src], dtype = torch.bool, device = src.device)
        
        readout_emb = self.readout_emb.weight.unsqueeze(1).repeat(1, batch_size, 1)
        readout_token = torch.zeros_like(readout_emb)
        readout_padding_mask = torch.zeros([batch_size, 1], dtype = torch.bool, device = src.device)
        all_src = torch.cat([src, readout_token], dim = 0)
        all_pos_emb = torch.cat([pos_emb, readout_emb], dim = 0)
        all_padding_mask = torch.cat([padding_mask, readout_padding_mask], dim = 1)

        all_attn_mask = torch.ones([len_src + 1, len_src + 1], dtype = torch.bool, device = src.device)
        all_attn_mask[:len_src, :len_src] = attn_mask if attn_mask is not None else False
        all_attn_mask[-1:, :] = False

        res = self.decoder(
            all_src,
            all_src,
            attn_mask = None,
            memory_attn_mask = all_attn_mask,
            padding_mask = None,
            memory_padding_mask = all_padding_mask,
            pos_emb = None,
            memory_pos_emb = all_pos_emb
        )
        return res[-1]


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    """
    def __init__(self, encoder_layer, num_layers, norm = None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                attn_mask = attn_mask,
                padding_mask = padding_mask,
                pos_emb = pos_emb
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, decoder_layer, num_layers, norm = None, return_intermediate = False):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        attn_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        memory_pos_emb: Optional[Tensor] = None
    ):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(
                tgt = output, 
                memory = memory,
                attn_mask = attn_mask,
                memory_attn_mask = memory_attn_mask,
                padding_mask = padding_mask,
                memory_padding_mask = memory_padding_mask,
                pos_emb = pos_emb,
                memory_pos_emb = memory_pos_emb
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    """
    def __init__(
        self,
        dim_model = 512,
        num_heads = 8,
        dim_feedforward = 2048,
        dropout = 0.1,
        activation = "relu",
        normalize_before = False
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, num_heads, dropout = dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_emb(self, tensor, pos_emb: Optional[Tensor] = None):
        return tensor if pos_emb is None else tensor + pos_emb

    def forward_post(
        self,
        src: Tensor,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None
    ):
        q = k = self.with_pos_emb(src, pos_emb)
        
        attn = self.self_attn(
            query = q, key = k, value = src, 
            attn_mask = attn_mask,
            key_padding_mask = padding_mask
        )[0]

        src = src + self.dropout1(attn)
        src = self.norm1(src)

        residual = self.linear2(self.dropout(self.activation(self.linear1(src))))

        src = src + self.dropout2(residual)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src: Tensor,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None
    ):
        norm_src = self.norm1(src)
        q = k = self.with_pos_emb(norm_src, pos_emb)
        
        attn = self.self_attn(
            query = q, key = k, value = norm_src, 
            attn_mask = attn_mask,
            key_padding_mask = padding_mask
        )[0]

        src = src + self.dropout1(attn)
        
        residual = self.norm2(src)
        residual = self.linear2(self.dropout(self.activation(self.linear1(residual))))

        src = src + self.dropout2(residual)
        return src

    def forward(
        self,
        src: Tensor,
        attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None
    ):
        if self.normalize_before:
            return self.forward_pre(src, attn_mask, padding_mask, pos_emb)
        return self.forward_post(src, attn_mask, padding_mask, pos_emb)


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    """
    def __init__(
        self, 
        dim_model = 512, 
        num_heads = 8, 
        dim_feedforward = 2048, 
        dropout = 0.1, 
        activation = "relu", 
        normalize_before = False,
        tgt_self_attn = True
    ):
        super(TransformerDecoderLayer, self).__init__()

        self.tgt_self_attn = tgt_self_attn
        if tgt_self_attn:
            self.self_attn = nn.MultiheadAttention(dim_model, num_heads, dropout = dropout)
            self.dropout = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(dim_model)
        
        self.multihead_attn = nn.MultiheadAttention(dim_model, num_heads, dropout = dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_emb(self, tensor, pos_emb: Optional[Tensor] = None):
        return tensor if pos_emb is None else tensor + pos_emb

    def forward_post(
        self,
        tgt,
        memory,
        attn_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        memory_pos_emb: Optional[Tensor] = None
    ):
        if self.tgt_self_attn:
            q = k = self.with_pos_emb(tgt, pos_emb)
            self_attn = self.self_attn(
                query = q, key = k, value = tgt,
                attn_mask = attn_mask,
                key_padding_mask = padding_mask
            )[0]
            tgt = tgt + self.dropout(self_attn)
            tgt = self.norm1(tgt)

        attn = self.multihead_attn(
            query = self.with_pos_emb(tgt, pos_emb),
            key = self.with_pos_emb(memory, memory_pos_emb),
            value = memory,
            attn_mask = memory_attn_mask,
            key_padding_mask = memory_padding_mask
        )[0]
        tgt = tgt + self.dropout1(attn)
        tgt = self.norm2(tgt)
        
        residual = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout3(residual)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        attn_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        memory_pos_emb: Optional[Tensor] = None
    ):
        if self.tgt_self_attn:
            norm_tgt = self.norm1(tgt)
            q = k = self.with_pos_emb(norm_tgt, pos_emb)
            self_attn = self.self_attn(
                query = q, key = k, value = tgt,
                attn_mask = attn_mask,
                padding_mask = padding_mask
            )[0]
            tgt = tgt + self.dropout(self_attn)
        
        norm_tgt = self.norm2(tgt)
        attn = self.multihead_attn(
            query = self.with_pos_emb(norm_tgt, pos_emb),
            key = self.with_pos_emb(memory, memory_pos_emb),
            value = memory,
            attn_mask = memory_attn_mask,
            key_padding_mask = memory_padding_mask
        )[0]
        tgt = tgt + self.dropout(attn)

        residual = self.norm3(tgt)
        residual = self.linear2(self.dropout(self.activation(self.linear1(residual))))
        
        tgt = tgt + self.dropout3(residual)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        attn_mask: Optional[Tensor] = None,
        memory_attn_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
        pos_emb: Optional[Tensor] = None,
        memory_pos_emb: Optional[Tensor] = None
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, attn_mask, memory_attn_mask, padding_mask, memory_padding_mask, pos_emb, memory_pos_emb)
        return self.forward_post(tgt, memory, attn_mask, memory_attn_mask, padding_mask, memory_padding_mask, pos_emb, memory_pos_emb)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
