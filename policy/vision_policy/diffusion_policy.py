"""
Diffusion Policy (CNN) (https://github.com/real-stanford/diffusion_policy).
"""
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from policy.configs.vision_policy.diffusion_policy import DiffusionPolicyCNNConfig

from policy.configs.common.action_head import *
from policy.common.action_head import get_action_head
from policy.policy_modules.diffusion_policy.vision import DiffusionPolicyVisionEncoder


class DiffusionPolicyCNN(nn.Module):
    def __init__(self, config: DiffusionPolicyCNNConfig):
        super(DiffusionPolicyCNN, self).__init__()
        self.config = config
        self.image_keys = self.config.vision.image_keys
        self.num_images = len(self.image_keys)
        
        # image encoders
        self.vision_backbones = [DiffusionPolicyVisionEncoder(
            name = self.config.vision.name,
            dim_feat = self.config.vision.dim_feat,
            img_size = self.config.vision.img_size,
            **self.config.vision.params
        )]
        if not self.config.vision.unified_encoder:
            for i in range(self.num_images - 1):
                self.vision_backbones.append(DiffusionPolicyVisionEncoder(
                    name = self.config.vision.name,
                    dim_feat = self.config.vision.dim_feat,
                    img_size = self.config.vision.img_size,
                    **self.config.vision.params
                ))
        self.num_encoders = len(self.vision_backbones)

        # diffusion action head
        if self.config.action_head.type not in [ActionHeadType.DIFFUSION]:
            raise AttributeError("Unsupported action head type: {}.".format(self.config.action_head.type))
        self.action_head = get_action_head(
            config = self.config.action_head,
            dim_cond = self.config.vision.dim_feat * self.num_images
        )

    def get_vision_feat(
        self,
        obs_dict: Dict[str, Any],
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        cond = []
        for idx in range(self.num_images):
            cond.append(self.vision_backbones[idx % self.num_encoders](obs_dict[self.image_keys[idx]]))
        cond = torch.cat(cond, dim = -1)
        return cond
    
    def forward(
        self, 
        obs_dict: Dict[str, Any], 
        action_dict: Optional[Dict[str, Any]] = None, 
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