"""
Action Heads.

Supports:
   - MLP head;
   - GRU head;
   - Diffusion head;
   - Flow-matching head.
"""
from typing import Union, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import reduce, rearrange
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from policy.configs.common.action_head import *
from policy.policy_modules.diffusion_policy.diffusion.conditional_unet1d import ConditionalUnet1D


def _compute_separate_loss(loss: torch.Tensor, seperate_loss_keys: Optional[Dict[str, int]], batch_reduction: bool = True) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Helper function to compute loss with optional separate loss keys.
    
    Args:
        loss: Loss tensor with shape [batch, ..., dim] where dim is the concatenated dimension
        seperate_loss_keys: Optional dictionary mapping loss keys to their dimensions
        
    Returns:
        If seperate_loss_keys is provided: Dictionary mapping keys to scalar loss values
        Otherwise: Scalar loss value
    """
    if seperate_loss_keys is not None:
        loss_dict = {}
        start_idx = 0
        for key, dim in seperate_loss_keys.items():
            end_idx = start_idx + dim
            loss_key = reduce(loss[..., start_idx:end_idx], 'b ... -> b', 'mean')
            loss_dict[key] = loss_key.mean() if batch_reduction else loss_key
            start_idx = end_idx
        return loss_dict
    else:
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        return loss.mean() if batch_reduction else loss


class MLPHead(nn.Module):
    def __init__(
        self,
        config: MLPHeadConfig,
        dim_cond: int
    ):
        super(MLPHead, self).__init__()
        self.config = config
        self.dim_action = config.dim_action
        self.dim_cond = dim_cond
        self.num_action = config.num_action
        self.dim_hidden = config.dim_hidden
        self.loss_func = getattr(F, config.loss_type)

        layers = []
        all_dims = [dim_cond] + list(self.dim_hidden) + [self.dim_action * self.num_action]
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
            if i < len(all_dims) - 2:
                layers.append(nn.LayerNorm(all_dims[i + 1]))
                layers.append(nn.Mish())
        self.mlps = nn.Sequential(*layers)

    def predict_action(self, cond):
        bs = cond.shape[0]
        action = self.mlps(cond).reshape(bs, self.num_action, self.dim_action)
        if self.config.loss_type == "binary_cross_entropy_with_logits":
            action = F.sigmoid(action)
        return action
    
    def compute_loss(self, cond, actions, seperate_loss_keys: Optional[Dict[str, int]] = None, batch_reduction: bool = True):
        bs = cond.shape[0]
        pred_action = self.mlps(cond).reshape(bs, self.num_action, self.dim_action)
        loss = self.loss_func(pred_action, actions, reduction='none')
        return _compute_separate_loss(loss, seperate_loss_keys, batch_reduction)


class GRUHead(nn.Module):
    def __init__(
        self,
        config: GRUHeadConfig,
        dim_cond: int
    ):
        super(GRUHead, self).__init__()
        self.config = config
        self.dim_action = config.dim_action
        self.dim_cond = dim_cond
        self.num_action = config.num_action
        self.dim_hidden = config.dim_hidden
        self.num_layers = config.num_layers
        self.with_latent = config.dim_latent is not None
        self.loss_func = getattr(F, config.loss_type)
        
        # hidden state projection
        hidden_proj = [nn.Linear(config.dim_latent if self.with_latent else dim_cond, self.dim_hidden * self.num_layers)]
        if config.norm_before_hidden:
            hidden_proj.append(nn.LayerNorm(self.dim_hidden * self.num_layers))
        self.hidden_proj = nn.Sequential(*hidden_proj)

        # gru decoder
        self.gru = nn.GRU(
            input_size = dim_cond,
            hidden_size = self.dim_hidden, 
            num_layers = self.num_layers,
            batch_first = True
        )
        self.cond_norm = nn.LayerNorm(dim_cond) if config.norm_before_input else nn.Identity() 
        
        # action projection
        self.action_proj = nn.Linear(self.dim_hidden, self.dim_action)
    
    def predict_action(self, cond, gru_hidden = None, latent = None, return_hidden = False, one_step = False):
        # determine condition type
        multiple_cond = (len(cond.shape) == 3)
        if multiple_cond:
            assert not one_step, "One-step prediction only supports single condition."
            assert cond.shape[1] == self.num_action, "Please provide the correct number of conditions (expect {}, found {}).".format(self.num_action, cond.shape[1])

        # process initial state
        if gru_hidden is None:
            if self.with_latent:
                assert latent is not None
                gru_hidden = self.hidden_proj(latent)
            else:
                gru_hidden = self.hidden_proj(cond[:, 0, :] if multiple_cond else cond)
            gru_hidden = rearrange(gru_hidden, "b (n d) -> n b d", n = self.num_layers)
            gru_hidden = gru_hidden.contiguous()
        else:
            assert gru_hidden.shape[0] == self.num_layers and gru_hidden.shape[2] == self.dim_hidden
        
        # process inputs
        cond = self.cond_norm(cond)
        if not multiple_cond:
            _cond = cond.unsqueeze(1)
            if not one_step:
                _cond = _cond.repeat(1, self.num_action, 1)
        else:
            _cond = cond
        
        gru_out, gru_hidden = self.gru(_cond, gru_hidden)
        action = self.action_proj(gru_out)
        
        # process outputs
        if one_step:
            action = action.squeeze(1)

        # return action(s) and hidden state
        if return_hidden:
            return action, gru_hidden
        else:
            return action

    def compute_loss(self, cond, actions, latent = None, seperate_loss_keys: Optional[Dict[str, int]] = None, batch_reduction: bool = True):
        pred_action = self.predict_action(cond, latent = latent)
        loss = self.loss_func(pred_action, actions, reduction='none')
        return _compute_separate_loss(loss, seperate_loss_keys, batch_reduction)


class DiffusionHead(nn.Module):
    def __init__(
        self,
        config: DiffusionHeadConfig,
        dim_cond: int
    ):
        super(DiffusionHead, self).__init__()
        self.config = config
        self.num_action = config.num_action
        self.dim_action = config.dim_action
        self.noise_scheduler_params = config.noise_scheduler_params

        self.model = ConditionalUnet1D(
            input_dim = self.dim_action,
            local_cond_dim = None,
            global_cond_dim = dim_cond,
            diffusion_step_embed_dim = config.diffusion_step_embed_dim,
            down_dims = config.down_dims,
            kernel_size = config.kernel_size,
            n_groups = config.n_groups,
            cond_predict_scale = config.cond_predict_scale
        )

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps = 100,
            beta_start = 0.0001,
            beta_end = 0.02,
            beta_schedule = "squaredcos_cap_v2",
            clip_sample = config.clip_sample,
            set_alpha_to_one = True,
            steps_offset = 0,
            prediction_type = "epsilon"
        )

        if config.num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            num_inference_steps = config.num_inference_steps
        self.num_inference_steps = num_inference_steps
    
    def conditional_sample(
        self,
        traj_shape,
        cond,
        num_inference_steps = None,
        generator = None,
        **kwargs
    ):
        trajectory = torch.randn(size = traj_shape, dtype = cond.dtype, device = cond.device, generator = generator)

        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        self.noise_scheduler.set_timesteps(num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            model_output = self.model(
                trajectory, 
                t, 
                local_cond = None, 
                global_cond = cond
            )
            trajectory = self.noise_scheduler.step(
                model_output, 
                t, 
                trajectory, 
                generator = generator,
                **kwargs
            ).prev_sample
            
        return trajectory

    def predict_action(self, cond):
        bs = cond.shape[0]
        return self.conditional_sample(
            traj_shape = (bs, self.num_action, self.dim_action), 
            cond = cond.reshape(bs, -1),
            **self.noise_scheduler_params
        )

    def compute_loss(self, cond, actions, seperate_loss_keys: Optional[Dict[str, int]] = None, batch_reduction: bool = True): 
        assert actions.shape[1] == self.num_action and actions.shape[2] == self.dim_action, \
            "Action shape mismatch (expect [{}, {}], found [{}, {}])".format(
                self.num_action, self.dim_action, actions.shape[1], actions.shape[2]
            )
        bs = cond.shape[0]
        noise = torch.randn(actions.shape, device = actions.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs, ), device = actions.device).long()
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        pred = self.model(
            noisy_actions, 
            timesteps, 
            local_cond = None, 
            global_cond = cond.reshape(bs, -1)
        )

        gt = noise if self.noise_scheduler.config.prediction_type == 'epsilon' else actions
        loss = F.mse_loss(pred, gt, reduction = 'none')
        return _compute_separate_loss(loss, seperate_loss_keys, batch_reduction)


class MIPHead(nn.Module):
    """
    Minimal Iterative Policy (MIP) Head using Diffusion UNet backbone.
    """
    def __init__(
        self, 
        config: MIPHeadConfig,
        dim_cond: int
    ):
        super(MIPHead, self).__init__()
        self.config = config
        self.dim_cond = dim_cond
        self.dim_action = config.dim_action
        self.num_action = config.num_action
        self.t_star = config.t_star
        self.coeff = 1.0 if self.config.fixed_scale_noising else self.t_star
        
        self.model = ConditionalUnet1D(
            input_dim = self.dim_action,
            local_cond_dim = None,
            global_cond_dim = self.dim_cond,
            diffusion_step_embed_dim = config.diffusion_step_embed_dim,
            down_dims = config.down_dims,
            kernel_size = config.kernel_size,
            n_groups = config.n_groups,
            cond_predict_scale = config.cond_predict_scale
        )

    def compute_loss(self, cond, actions, seperate_loss_keys: Optional[Dict[str, int]] = None, batch_reduction: bool = True):
        """
        Computes the MIP training loss.
        """
        assert actions.shape[1] == self.num_action and actions.shape[2] == self.dim_action, \
            "Action shape mismatch (expect [{}, {}], found [{}, {}])".format(
                self.num_action, self.dim_action, actions.shape[1], actions.shape[2]
            )
        bs = cond.shape[0]
        
        pred_0 = self.model(
            torch.zeros_like(actions), 
            torch.zeros((bs, ), device = actions.device), 
            local_cond = None, 
            global_cond = cond
        )
        loss_0 = F.mse_loss(pred_0, actions, reduction = 'none')

        pred_t_star = self.model(
            self.coeff * actions + (1 - self.t_star) * torch.randn_like(actions), 
            torch.full((bs, ), self.t_star, device = actions.device), 
            local_cond = None, 
            global_cond = cond
        )
        loss_t_star = F.mse_loss(pred_t_star, actions, reduction = 'none')
        
        loss = loss_0 + loss_t_star
        return _compute_separate_loss(loss, seperate_loss_keys, batch_reduction)

    
    def predict_action(self, cond):
        """
        MIP Inference.
        """
        bs = cond.shape[0]

        action_0 = self.model(
            torch.zeros((bs, self.num_action, self.dim_action), device = cond.device), 
            torch.zeros((bs, ), device = cond.device), 
            local_cond = None, 
            global_cond = cond
        )
        action = self.model(
            self.coeff * action_0, 
            torch.full((bs, ), self.t_star, device = cond.device), 
            local_cond = None, 
            global_cond = cond
        )
        
        return action


def get_action_head(config: ActionHeadConfig, dim_cond: int):
    if config.type == ActionHeadType.MLP:
        return MLPHead(config, dim_cond)
    elif config.type == ActionHeadType.GRU:
        return GRUHead(config, dim_cond)
    elif config.type == ActionHeadType.DIFFUSION:
        return DiffusionHead(config, dim_cond)
    elif config.type == ActionHeadType.MIP:
        return MIPHead(config, dim_cond)
    else:
        raise ValueError(f"Unsupported action head config type: {type(config)}")
