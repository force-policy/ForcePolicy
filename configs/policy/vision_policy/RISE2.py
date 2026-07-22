import torch

from policy.configs.common import *
from policy.configs.vision_policy import RISE2Config, DenseEncoderConfig


policy_config = RISE2Config(
    dim_hidden = 512,
    disable_pcd_color = True,
    dense_encoder = DenseEncoderConfig(
        name = "dinov2-base",
        dim_dense_feat = 128,
        dim_sparse_feat = 128,
        finetune = "lora",
        dtype = torch.bfloat16,
        interp_fn_mode = "custom"
    ),
    backbone = TransformerConfig(
        num_heads = 8,
        num_encoder_layers = 0,
        num_decoder_layers = 4,
        dim_feedforward = 2048,
        dropout = 0.1
    ),
    action_head = DiffusionHeadConfig(
        dim_action = 10,
        num_action = 50,
        num_inference_steps = 20,
        diffusion_step_embed_dim = 256,
        down_dims = (256, 512),
        kernel_size = 5,
        n_groups = 8,
        cond_predict_scale = False
    )
)