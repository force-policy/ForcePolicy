from typing import Any, List, Dict, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.common.resnet import ResNetEncoder
from policy.common.film import ResNetFiLMEncoder
from policy.common.seq_encoder import get_seq_encoder
from policy.common.action_head import get_action_head
from policy.force_policy.fusion import GatedFusion, CrossAttnFusion
from policy.configs.force_policy.force_policy import ForcePolicyConfig, SeqEncoderConfig, VisionEncoderConfig, FusionConfig


def get_one_step_reference_gt(
    mask: torch.Tensor,
    frame: torch.Tensor,
    force: torch.Tensor,
    pre_contact_steps: int = 4,
    pre_release_steps: int = 6, 
    release_rho: float = 0.5
):
    """
    Get one-step reference ground truth for force policy.
    """
    device = mask.device
    B, T, _ = mask.shape

    c = (mask.amax(dim = -1) > 0.5)
    is_contact = c[:, 0]

    has_contact_any = c.any(dim = 1)
    t_contact = torch.argmax(c.to(torch.int64), dim = 1)
    t_contact = torch.where(has_contact_any, t_contact, torch.full_like(t_contact, T))

    will_contact = (~is_contact) & c[:, :pre_contact_steps].any(dim = 1)
    will_release = is_contact & (c[:, :pre_release_steps].to(torch.float32).mean(dim = 1) < release_rho)

    empty_mask = torch.zeros((B, 6), device = device, dtype = mask.dtype)
    empty_force = torch.zeros((B, 6), device = device, dtype = force.dtype)
    empty_frame = torch.tensor([[0, 0, 0, 1., 0., 0., 0., 1., 0.]], device = device, dtype = frame.dtype).expand(B, 9)

    # helper: gather x at per-sample idx
    def gather_at(idx_1d: torch.Tensor, x: torch.Tensor):
        idx = torch.clamp(idx_1d, 0, T - 1)
        return x[torch.arange(B, device = device), idx]

    in_idx = torch.zeros((B,), device = device, dtype = torch.long)
    in_mask = gather_at(in_idx, mask)
    in_frame = gather_at(in_idx, frame)
    in_force = gather_at(in_idx, force)

    tc_valid = (t_contact < T)
    pre_mask = gather_at(t_contact, mask)
    pre_frame = gather_at(t_contact, frame)
    pre_force = gather_at(t_contact, force)

    use_in = (is_contact & (~will_release)).unsqueeze(-1)
    ref_mask = torch.where(use_in, in_mask, empty_mask)
    ref_frame = torch.where(use_in, in_frame, empty_frame)
    ref_force = torch.where(use_in, in_force, empty_force)

    use_pre = (will_contact & tc_valid).unsqueeze(-1)
    ref_mask = torch.where(use_pre, pre_mask, ref_mask)
    ref_frame = torch.where(use_pre, pre_frame, ref_frame)
    ref_force = torch.where(use_pre, pre_force, ref_force)

    return ref_mask, ref_frame, ref_force


class ForcePolicy(nn.Module):
    def __init__(self, config: ForcePolicyConfig):
        super(ForcePolicy, self).__init__()
        self.config = config
        self.has_vision_feat = config.dim_vision_feat is not None 
        
        if not self.has_vision_feat:
            if config.fuse_global_vision_feat:
                raise ValueError("dim_vision_feat is None, but fuse_global_vision_feat is True.")
            for vis_cfg in config.vision_encoders:
                if vis_cfg.use_film:
                    raise ValueError("dim_vision_feat is None, but use_film is True.")
            if config.fusion.type == 'cross_attn':
                raise ValueError("dim_vision_feat is None, but fusion type is 'cross_attn' (requires global vision feature as query).")

        # 1. Vision Encoder
        if len(config.vision_encoders) != len(config.camera_names):
            raise ValueError(f"Number of vision encoders ({len(config.vision_encoders)}) does not match number of camera names ({len(config.camera_names)}).")
        
        self.visual_encoders = nn.ModuleList()
        self.dim_vis_feats = []
        for vis_cfg in config.vision_encoders:
            enc, dim_out = self._build_vision_encoder(vis_cfg)
            self.visual_encoders.append(enc)
            self.dim_vis_feats.append(dim_out)

        # 2. Sequential Encoder
        self.enc_lowdim = None
        self.enc_proprio = None
        self.enc_force_torque = None
        self.dim_lowdim_feats = []

        if not config.separate_lowdim_encoders:
            self.enc_lowdim, dims = self._build_lowdim_encoder(config.lowdim_encoder, separate = False)
        else:
            self.enc_proprio, self.enc_force_torque, dims = self._build_lowdim_encoder(config.lowdim_encoder, separate = True)
        self.dim_lowdim_feats.extend(dims)

        # 3. Fusion
        self.fusion = self._build_fusion(config.fusion)

        # 4. Heads
        if not config.separate_action_decoders:
            assert config.action_head.dim_action == config.dim_pred_action + config.dim_pred_frame_force + config.dim_pred_frame + config.dim_pred_frame_mask
            self.action_head = get_action_head(config.action_head, dim_cond = config.fusion.dim_hidden)
        else:
            assert config.action_head[0].dim_action == config.dim_pred_action
            assert config.action_head[1].dim_action == config.dim_pred_frame_force
            assert config.action_head[2].dim_action == config.dim_pred_frame
            assert config.action_head[3].dim_action == config.dim_pred_frame_mask
            self.action_head = nn.ModuleList([
                get_action_head(config.action_head[0], dim_cond = config.fusion.dim_hidden),
                get_action_head(config.action_head[1], dim_cond = config.fusion.dim_hidden),
                get_action_head(config.action_head[2], dim_cond = config.fusion.dim_hidden),
                get_action_head(config.action_head[3], dim_cond = config.fusion.dim_hidden)
            ])


    def _build_vision_encoder(self, vis_cfg: VisionEncoderConfig):
        """ Build vision encoder from configurations. """
        if vis_cfg.use_film:
             enc = ResNetFiLMEncoder(
                dim_cond = self.config.dim_vision_feat,
                img_channels = 3, 
                dim_resnet = vis_cfg.dim_resnet,
                kernel_resnet = vis_cfg.kernel_resnet,
                stride_resnet = vis_cfg.stride_resnet,
                return_tokens = vis_cfg.return_tokens
            )
        else:
             enc = ResNetEncoder(
                img_channels = 3,
                dim_resnet = vis_cfg.dim_resnet,
                kernel_resnet = vis_cfg.kernel_resnet,
                stride_resnet = vis_cfg.stride_resnet,
                return_tokens = vis_cfg.return_tokens
            )
        return enc, vis_cfg.dim_resnet[-1]


    def _build_lowdim_encoder(self, encoder_cfg: Union[SeqEncoderConfig, List[SeqEncoderConfig]], separate: bool):
        """ Build lowdim encoder from configurations. """
        if not separate:
            if isinstance(encoder_cfg, list):
                raise ValueError("separate_lowdim_encoders = False, but lowdim_encoder is a list.")

            enc = get_seq_encoder(encoder_cfg)

            d = encoder_cfg.dim_hidden
            if isinstance(d, list): d = d[-1]

            return enc, [d]

        else:
             if not isinstance(encoder_cfg, list):
                 raise ValueError("separate_lowdim_encoders = True, but lowdim_encoder is not a list.")
             if len(encoder_cfg) != 2:
                 raise ValueError("separate_lowdim_encoders = True, lowdim_encoder list must have length 2 (proprio, force_torque).")
             
             proprio_cfg = encoder_cfg[0]
             force_torque_cfg = encoder_cfg[1]
             
             enc_proprio = get_seq_encoder(proprio_cfg)
             enc_force_torque = get_seq_encoder(force_torque_cfg)
             
             dims = []
             d1 = proprio_cfg.dim_hidden
             if isinstance(d1, list): d1 = d1[-1]
             dims.append(d1)
             
             d2 = force_torque_cfg.dim_hidden
             if isinstance(d2, list): d2 = d2[-1]
             dims.append(d2)
             
             return enc_proprio, enc_force_torque, dims


    def _build_fusion(self, fusion_cfg: FusionConfig):
        """ Build fusion module from configurations. """
        fusion_input_dims = []
        fusion_input_dims.extend(self.dim_vis_feats)
        fusion_input_dims.extend(self.dim_lowdim_feats)

        if self.config.fuse_global_vision_feat:
            fusion_input_dims.append(self.config.dim_vision_feat)

        if fusion_cfg.type == 'gated':
            return GatedFusion(
                dim_inputs = fusion_input_dims,
                dim_hidden = fusion_cfg.dim_hidden
            )

        elif fusion_cfg.type == 'cross_attn':
            return CrossAttnFusion(
                dim_inputs = fusion_input_dims,
                dim_query = fusion_cfg.dim_query,
                dim_hidden = fusion_cfg.dim_hidden,
                num_heads = fusion_cfg.num_heads,
                dim_feedforward = fusion_cfg.dim_feedforward,
                p_dropout = fusion_cfg.p_dropout,
                type_emb_enabled = fusion_cfg.type_emb_enabled,
                pos_emb_enabled = fusion_cfg.pos_emb_enabled
            )
        
        else:
             raise ValueError(f"Unknown fusion type: {fusion_cfg.type}")

    
    def _encode(
        self,
        obs_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Encodes inputs and fuses them into a single feature vector.
        """
        global_vision_feat = obs_dict.get('vision_feat', None)
        if self.has_vision_feat:
            if global_vision_feat is None:
                raise ValueError("Model expects 'vision_feat' in obs_dict, but it is missing.")
            if global_vision_feat.ndim == 3:
                global_vision_feat = global_vision_feat.squeeze(1)
            global_vision_feat = global_vision_feat.float()
        proprio = obs_dict['proprio']
        force_torque = obs_dict['force_torque']
        
        # 1. Vision Features
        vis_feats = []
        for i, (name, encoder) in enumerate(zip(self.config.camera_names, self.visual_encoders)):
            img = obs_dict.get(f"image_{name}", None)
            if img is None:
                raise ValueError(f"Camera input '{name}' not found in obs_dict.")
            
            if isinstance(encoder, ResNetFiLMEncoder):
                feat = encoder(img, global_vision_feat)
            else:
                feat = encoder(img)
            vis_feats.append(feat)
        
        # 2. Lowdim Features
        lowdim_feats = []
        if not self.config.separate_lowdim_encoders:
            feat = self.enc_lowdim(
                x = torch.cat([proprio, force_torque], dim = -1), 
                cond = global_vision_feat if self.has_vision_feat else None
            )
            lowdim_feats.append(feat)
        else:
            p_feat = self.enc_proprio(
                x = proprio, 
                cond = global_vision_feat if self.has_vision_feat else None
            )
            ft_feat = self.enc_force_torque(
                force_torque,
                cond = global_vision_feat if self.has_vision_feat else None
            )
            lowdim_feats.append(p_feat)
            lowdim_feats.append(ft_feat)
        
        # 3. Fusion Inputs
        inputs = vis_feats + lowdim_feats
        
        if self.config.fuse_global_vision_feat:
             inputs.append(global_vision_feat)
        
        if isinstance(self.fusion, CrossAttnFusion):
            assert self.has_vision_feat, "CrossAttnFusion requires a global vision feature as query."
            fused_feat = self.fusion(inputs, global_vision_feat)
        else:
            fused_feat = self.fusion(inputs)

        return fused_feat

    
    def predict_action(
        self, 
        obs_dict: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """ 
        Predict actions for inference.
        """
        fused_feat = self._encode(obs_dict)

        if self.config.separate_action_decoders:
            pred_action = self.action_head[0].predict_action(fused_feat)
            pred_force = self.action_head[1].predict_action(fused_feat)
            pred_frame = self.action_head[2].predict_action(fused_feat)
            pred_mask_prob = self.action_head[3].predict_action(fused_feat)
        else:
            action_combined = self.action_head.predict_action(fused_feat)
            pred_action = action_combined[..., : self.config.dim_pred_action]
            pred_force = action_combined[..., self.config.dim_pred_action: self.config.dim_pred_action + self.config.dim_pred_frame_force]
            pred_frame = action_combined[..., self.config.dim_pred_action + self.config.dim_pred_frame_force : self.config.dim_pred_action + self.config.dim_pred_frame_force + self.config.dim_pred_frame]
            pred_mask_prob = action_combined[..., self.config.dim_pred_action + self.config.dim_pred_frame_force + self.config.dim_pred_frame:]

        pred_mask = (pred_mask_prob > self.config.mask_threshold).bool()
        switch_signal = pred_mask[:, self.config.router_lookahead_steps:, :].any(dim = -1).any(dim = -1).unsqueeze(-1)

        return {
            "action": pred_action,
            "force": pred_force,
            "frame": pred_frame,
            "mask": pred_mask,
            "switch_signal": switch_signal
        }


    def compute_loss(
        self,
        obs_dict: Dict[str, torch.Tensor],
        action_dict: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """ 
        Compute loss for training.
        """
        if self.config.aug_force_torque.enable:
            force_input = obs_dict['force_torque']
            rand_probs = torch.rand(force_input.shape[0], 1, 1, device = force_input.device)
            noise_data = torch.randn_like(force_input) * self.config.aug_force_torque.std
            mask = rand_probs < self.config.aug_force_torque.prob
            obs_dict['force_torque'] = torch.where(mask, noise_data, force_input)
        
        fused_feat = self._encode(obs_dict)
        
        if self.config.one_step_reference_gt:
            mask_gt, frame_gt, force_gt = get_one_step_reference_gt(
                action_dict["mask"],
                action_dict["frame"],
                action_dict["force"],
                self.config.pre_contact_steps,
                self.config.pre_release_steps,
                self.config.release_rho
            )
            mask_gt = mask_gt.unsqueeze(1)
            frame_gt = frame_gt.unsqueeze(1)
            force_gt = force_gt.unsqueeze(1)
            action_gt = action_dict["action"]
        else:
            mask_gt = action_dict["mask"]
            frame_gt = action_dict["frame"]
            force_gt = action_dict["force"]
            action_gt = action_dict["action"]

        loss_dict = {}
        
        if self.config.separate_action_decoders:
            action_gt = action_gt[:, :self.config.action_head[0].num_action, :]
            force_gt = force_gt[:, :self.config.action_head[1].num_action, :]
            frame_gt = frame_gt[:, :self.config.action_head[2].num_action, :]
            mask_gt = mask_gt[:, :self.config.action_head[3].num_action, :]
            loss_dict["action"] = self.action_head[0].compute_loss(fused_feat, action_gt, batch_reduction = True)
            loss_dict["force"] = self.action_head[1].compute_loss(fused_feat, force_gt, batch_reduction = True)
            loss_dict["frame"] = self.action_head[2].compute_loss(fused_feat, frame_gt, batch_reduction = True)
            loss_dict["mask"] = self.action_head[3].compute_loss(fused_feat, mask_gt, batch_reduction = True)
        else:
            action_gt = action_gt[:, :self.config.action_head.num_action, :]
            force_gt = force_gt[:, :self.config.action_head.num_action, :]
            frame_gt = frame_gt[:, :self.config.action_head.num_action, :]
            mask_gt = mask_gt[:, :self.config.action_head.num_action, :]
            loss_dict = self.action_head.compute_loss(
                fused_feat,
                torch.cat([action_gt, force_gt, frame_gt, mask_gt], dim = -1),
                seperate_loss_keys = {
                    "action": self.config.dim_pred_action,
                    "force": self.config.dim_pred_frame_force,
                    "frame": self.config.dim_pred_frame,
                    "mask": self.config.dim_pred_frame_mask
                },
                batch_reduction = True
            )

        total_loss = sum(
            self.config.loss_weights[key] * loss_dict[key]
            for key in loss_dict
        )

        loss_dict["loss"] = total_loss
        return loss_dict
    

    def forward(
        self, 
        obs_dict: Dict[str, Any],
        action_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if action_dict is None:
            with torch.no_grad():
                return self.predict_action(obs_dict)
        else:
            return self.compute_loss(obs_dict, action_dict)
