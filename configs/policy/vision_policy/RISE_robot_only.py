from policy.configs.common import *
from policy.configs.vision_policy import RISEConfig


policy_config = RISEConfig(
    dim_hidden = 512,
    disable_pcd_color = False,
    backbone = TransformerConfig(
        num_heads = 8,
        num_encoder_layers = 4,
        num_decoder_layers = 1,
        dim_feedforward = 2048,
        dropout = 0.1
    ),
    action_head = DiffusionHeadConfig(
        dim_action = 9,
        num_action = 50,
        num_inference_steps = 20,
        diffusion_step_embed_dim = 256,
        down_dims = (256, 512),
        kernel_size = 5,
        n_groups = 8,
        cond_predict_scale = False
    )
)