import torch
import torch.nn as nn
from torch.amp import autocast

from policy.policy_modules.RISE.dense_modules.resnet import ResNetEncoder
from policy.policy_modules.RISE.dense_modules.dino import DINOEncoder
from policy.policy_modules.RISE.dense_modules.paligemma import PaliGemmaEncoder


class DenseEncoder(nn.Module):
    def __init__(
        self,
        name,
        dim_output = 512,
        finetune = "full",
        dtype = torch.float32,
        **kwargs
    ):
        super(DenseEncoder, self).__init__()
        
        if name.startswith("resnet"):
            Encoder = ResNetEncoder
        elif name.startswith("dinov2") or name.startswith("dinov3"):
            Encoder = DINOEncoder
        elif name.startswith("paligemma"):
            Encoder = PaliGemmaEncoder
        else:
            raise ValueError(f"Unsupported encoder name: {name}. Supported encoders: resnet*, dinov2*, paligemma*")
            
        self.dense_encoder = Encoder(
            name = name,
            dim_output = dim_output, 
            finetune = finetune,
            dtype = dtype,
            **kwargs
        )

        self.image_enc_dtype = dtype
        self.dim_output = dim_output
        self.encoder_name = name
    
    def forward(
        self,
        image,
        lang = None
    ):
        with autocast(
            device_type = image.device.type, 
            dtype = self.image_enc_dtype if image.device.type == 'cuda' else torch.float32
        ):
            # Handle different encoder requirements
            if self.encoder_name.startswith("paligemma"):
                if lang is None:
                    raise ValueError("PaliGemma encoder requires language input")
                image_feat = self.dense_encoder(image, lang = lang)
            else:
                # ResNet and DINO don't use lang parameter
                image_feat = self.dense_encoder(image)
        
        if self.image_enc_dtype != torch.float32:
            image_feat = image_feat.to(torch.float32)

        return image_feat
