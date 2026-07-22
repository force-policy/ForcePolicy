from policy.configs.common.seq_encoder import GRUSeqEncoderConfig
from policy.configs.common.action_head import MLPHeadConfig, MIPHeadConfig
from policy.configs.force_policy.force_policy import ForcePolicyConfig, VisionEncoderConfig, FusionConfig, AugForceTorqueConfig


policy_config = ForcePolicyConfig(
    vision_encoders = [
        VisionEncoderConfig(
            type = 'resnet',
            use_film = True,
            dim_resnet = [64, 128, 256],
            kernel_resnet = [3, 3, 3],
            stride_resnet = [1, 2, 2],
            return_tokens = False
        )
    ],
    camera_names = ['wrist'],
    
    # Lowdim
    separate_lowdim_encoders = False,
    lowdim_encoder = GRUSeqEncoderConfig(
        dim_input = 15, 
        num_input = 100,
        dim_cond = 512,
        dim_hidden = 128,
        num_layers = 2,
        patch_size = 1,
        p_dropout = 0.1,
        aggregate = True
    ),
    
    # Fusion
    fusion = FusionConfig(
        type = 'gated',
        dim_hidden = 128
    ),
    fuse_global_vision_feat = True,
    
    # Heads
    separate_action_decoders = True,
    one_step_reference_gt = True,
    # pre_contact_steps / pre_release_steps: the "look-ahead" hyper-params that control WHEN the
    # force policy switches stages -- free-space -> contact and contact -> release (see will_contact /
    # will_release in get_one_step_reference_gt). This is the most critical timing knob for the force
    # policy: switching too late causes contact impact / pushing into empty space after release, while
    # switching too early releases the force prematurely. If stages are not switching at the proper
    # time, tune these two values first.
    pre_contact_steps = 40,
    pre_release_steps = 40,
    release_rho = 0.9,
    action_head = [
        MIPHeadConfig(
            dim_action = 9,
            num_action = 50,
            diffusion_step_embed_dim = 128,
            down_dims = (128, 256),
            kernel_size = 5,
            n_groups = 8,
            cond_predict_scale = False
        ),
        MLPHeadConfig(
            dim_action = 6,
            num_action = 1,
            dim_hidden = (128, 64),
            loss_type = "l1_loss"
        ), 
        MLPHeadConfig(
            dim_action = 9,
            num_action = 1,
            dim_hidden = (128, 64),
            loss_type = "l1_loss"
        ),
        MLPHeadConfig(
            dim_action = 6,
            num_action = 1,
            dim_hidden = (128, 64),
            loss_type = "binary_cross_entropy_with_logits"
        )
    ],
    
    router_lookahead_steps = 0,
    mask_threshold = 0.5,
    
    # Loss weights
    loss_weights = {
        "action": 1.0,
        "frame": 1.0,
        "force": 1.0,
        "mask": 0.1
    },

    aug_force_torque = AugForceTorqueConfig(
        enable = False,
        std = 0.1,
        prob = 0.2
    ),
    
    # Global Embedding
    dim_vision_feat = 512,
    dim_pred_action = 9,
    dim_pred_frame = 9,
    dim_pred_frame_force = 6,
    dim_pred_frame_mask = 6
)
