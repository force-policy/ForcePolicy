"""
RISE Policy (https://github.com/rise-policy/RISE).
"""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from policy.configs.vision_policy.RISE import RISEConfig

from policy.configs.common.action_head import *
from policy.common.transformer import Transformer
from policy.common.action_head import get_action_head
from policy.policy_modules.RISE.sparse_modules.encoder import SparseEncoder


class RISE(nn.Module):
    def __init__(self, config: RISEConfig):
        super(RISE, self).__init__()
        self.config = config
        # sparse 3D encoder
        self.encoder = SparseEncoder(
            dim_input = 3 if self.config.disable_pcd_color else 6,
            dim_output = self.config.dim_hidden,
            dense_encoder = None
        )
        # transformer backbone
        self.transformer = Transformer(
            dim_model = self.config.dim_hidden, 
            num_heads = self.config.backbone.num_heads,
            num_encoder_layers = self.config.backbone.num_encoder_layers,
            num_decoder_layers = self.config.backbone.num_decoder_layers,
            dim_feedforward = self.config.backbone.dim_feedforward, 
            dropout = self.config.backbone.dropout
        )
        # diffusion action head
        if self.config.action_head.type not in [ActionHeadType.DIFFUSION]:
            raise AttributeError("Unsupported action head type: {}.".format(self.config.action_head.type))
        self.action_head = get_action_head(
            config = self.config.action_head,
            dim_cond = self.config.dim_hidden
        )

    def get_vision_feat(
        self,
        obs_dict: Dict[str, Any],
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        src, src_pos_emb, src_padding_mask = self.encoder(obs_dict["cloud"], batch_size = batch_size)
        cond = self.transformer(src, padding_mask = src_padding_mask, pos_emb = src_pos_emb) 
        return cond
    
    def forward(
        self, 
        obs_dict: Dict[str, Any], 
        action_dict: Optional[Dict[str, Any]] = None, 
        batch_size: int = 1, 
        **kwargs
    ) -> Dict[str, Any]:
        cond = self.get_vision_feat(obs_dict, batch_size = batch_size)
        if action_dict is None:
            with torch.no_grad():
                return {
                    "action": self.action_head.predict_action(cond, **kwargs),
                    "vision_feat": cond
                }
        else:
            loss_action = self.action_head.compute_loss(cond, actions = action_dict["action"])
            return {"loss": loss_action}
