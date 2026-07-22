"""
ResNet-based vision encoder of diffusion policy.
"""

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from typing import Callable


class DiffusionPolicyVisionEncoder(nn.Module):
    """
    Vision encoder of diffusion policy.
    """
    def __init__(
        self,
        dim_feat,
        img_size,
        name: str = "renset18",
        pretrained_weights: str = None,
        img_crop: bool = True,
        img_crop_random: bool = True,
        img_crop_ratio: float = 0.875,
        use_group_norm: bool = True,
        use_spatial_softmax: bool = True,
        num_keypoints: int = 32,
    ):
        super(DiffusionPolicyVisionEncoder, self).__init__()
        
        assert len(img_size) == 2

        self.img_crop = img_crop
        
        assert not use_group_norm or pretrained_weights is None, "Cannot replace the normalization layers in the pre-trained vision encoder."
        
        backbone_model = getattr(torchvision.models, name)(weights = pretrained_weights)
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        
        if img_crop:
            crop_size = (int(img_size[0] * img_crop_ratio), int(img_size[1] * img_crop_ratio))
            self.train_crop = T.RandomCrop(crop_size) if img_crop_random else T.CenterCrop(crop_size)
            self.eval_crop = T.CenterCrop(crop_size)

        if use_group_norm:
            self.backbone = replace_submodules(
                root_module = self.backbone,
                predicate = lambda x: isinstance(x, nn.BatchNorm2d),
                func = lambda x: nn.GroupNorm(
                    num_groups = x.num_features // 16,
                    num_channels = x.num_features
                )
            )

        # dummy run to determine the arguments.
        dummy_size = crop_size if img_crop else img_size
        dummy_img = torch.zeros((1, 3, *dummy_size), dtype = torch.float32)
        with torch.inference_mode():
            dummy_feat = self.backbone(dummy_img)

        if use_spatial_softmax:
            self.pool = SpatialSoftmax(
                input_shape = dummy_feat.shape[1:],
                num_kp = num_keypoints
            )
            dim_feat_after_pool = num_keypoints * 2
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            dim_feat_after_pool = dummy_feat.shape[1]
        
        self.out = nn.Sequential(
            nn.Linear(dim_feat_after_pool, dim_feat),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Args:
        - x: B * C * H * W

        Returns:
        - x': B * D_feat
        """
        if self.img_crop:
            x = self.train_crop(x) if self.training else self.eval_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim = 1)
        return self.out(x)


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(
        self,
        input_shape,
        num_kp = None,
        temperature = 1.,
        learnable_temperature = False,
        output_variance = False,
        noise_std = 0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not use spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self._in_w),
            np.linspace(-1., 1., self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None


    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial 
        probability distribution is created using a softmax, where the support is the 
        pixel locations. This distribution is used to compute the expected value of 
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim = 1, keepdim = True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim = 1, keepdim = True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim = 1, keepdim = True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints


def replace_submodules(
    root_module: nn.Module, 
    predicate: Callable[[nn.Module], bool], 
    func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module