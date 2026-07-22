import os
import torch
import torchvision

from torch import nn
from einops import rearrange
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


class DINOEncoder(nn.Module):
    """DINOv2/v3 backbone with optional LoRA fine-tuning."""
    def __init__(
        self, 
        name: str = "dinov2-base", 
        dim_output: int = 512,
        finetune: str = "lora", 
        dtype = torch.float32,
        lora_rank: int = 16, 
        lora_dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        assert finetune in ["full", "lora", "none"], "finetune parameter should be one of [full, lora, none]."
        
        try:
            dino = AutoModel.from_pretrained(os.path.join("./weights", name), dtype = dtype)
        except Exception as e:
            dino = AutoModel.from_pretrained(os.path.join("./weights", name), torch_dtype = dtype)
        
        self.num_register_tokens = 0 if name.startswith("dinov2") else dino.config.num_register_tokens

        if finetune == "lora":
            dino.requires_grad_(False)
            config = LoraConfig(
                r              = lora_rank,
                lora_alpha     = lora_rank,
                target_modules = ['projection', 'query', 'key', 'value', 'dense', 'fc1', 'fc2'] if name.startswith("dinov2") else ['patch_embeddings', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj'],
                lora_dropout   = lora_dropout,
                bias           = 'none',
                use_rslora     = True,
            )
            dino = get_peft_model(dino, config)
            # convert LoRA parameters to float32
            for name, param in dino.named_parameters():
                if "lora_" in name:
                    param.data = param.data.float()
        elif finetune == "none":
            dino.requires_grad_(False)
        
        self.model = dino

        self.patch_size = dino.config.patch_size
        hidden_size = dino.config.hidden_size
        if hidden_size != dim_output:
            self.proj = nn.Linear(hidden_size, dim_output)
        else:
            self.proj = nn.Identity()
        self.num_channels = dim_output

    def forward(self, img, **kwargs):
        H, W = img.shape[-2:]
        grid_H, grid_W = H // self.patch_size, W // self.patch_size
        feats = self.model(img).last_hidden_state[:, 1 + self.num_register_tokens:] # B, L, hidden_size
        feats = self.proj(feats)    # B, L, num_channels
        feats = feats.reshape(-1, grid_H, grid_W, self.num_channels).permute(0, 3, 1, 2)

        return feats
    