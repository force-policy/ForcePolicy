import os
import torch
import torchvision
import torch.nn as nn

from torchvision.models._utils import IntermediateLayerGetter



class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class ResNetEncoder(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str = "resnet18", dim_output: int = 512, finetune: str = "full", **kwargs):
        super().__init__()
        backbone = getattr(torchvision.models, name)(weights = "IMAGENET1K_V1", norm_layer = FrozenBatchNorm2d)

        if finetune == "none":
            backbone.requires_grad_(False)

        self.body = IntermediateLayerGetter(backbone, return_layers={'layer4': "0"})
        
        enc_dim_output = 512 if name in ('resnet18', 'resnet34') else 2048
        if enc_dim_output != dim_output:
            self.proj = nn.Conv2d(enc_dim_output, dim_output, 1)
        else:
            self.proj = nn.Identity()
        self.num_channels = dim_output

    def forward(self, img, **kwargs):
        feats = self.body(img)["0"]
        feats = self.proj(feats)
        return feats