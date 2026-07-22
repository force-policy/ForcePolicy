"""
Augmentation operations and context.
"""
from typing import Dict, Any, List, Tuple, Optional, Callable, Union

import sys
import torch
import numpy as np

try:
    import torchvision.transforms.v2 as T
    import torchvision.transforms.v2.functional as TF
except ImportError:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF


from data_infra.configs import (
    DataType,
    RotationType,
    AugmentationType,
    BaseAugmentationConfig,
    RandomTransformAugmentationConfig,
    ColorJitterAugmentationConfig
)

import utils.transforms.projection as projection_utils
from utils.transforms.generator import trans_mat, rot_trans_mat
from utils.transforms.pose import xyz_rot_to_mat, mat_to_xyz_rot, transform_matmul


class AugmentationContext:
    """
    Context to manage shared augmentation parameters.
    """
    def __init__(self):
        self._params: Dict[str, Any] = {}

    def get_op_param(
        self, 
        aug_name: str, 
        generator_fn: Callable[[], Any]
    ) -> Any:
        """
        Get or generate an augmentation parameter.
        """
        if aug_name not in self._params:
            self._params[aug_name] = generator_fn()
        return self._params[aug_name]


def generate_random_transform(
    config: RandomTransformAugmentationConfig, 
    batch_size: int, 
    device: torch.device
) -> torch.Tensor:
    """ 
    Generate (B, 4, 4) transform matrices.
    Batchified generation.
    """
    def _get_angle(r, b):
        return (torch.rand(b, device = device) * (r[1] - r[0]) + r[0]) * np.pi / 180.0

    rx = _get_angle(config.rot_x_range, batch_size)
    ry = _get_angle(config.rot_y_range, batch_size)
    rz = _get_angle(config.rot_z_range, batch_size)
    angles = torch.stack([rx, ry, rz], dim = -1)

    def _get_trans(r, b):
        return torch.rand(b, device = device) * (r[1] - r[0]) + r[0]

    tx = _get_trans(config.trans_x_range, batch_size)
    ty = _get_trans(config.trans_y_range, batch_size)
    tz = _get_trans(config.trans_z_range, batch_size)
    offsets = torch.stack([tx, ty, tz], dim = -1)

    return rot_trans_mat(offsets, angles, convention = "XYZ")


def augment_random_transform(
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    aug_groups: List[str],
    aug_configs: Dict[str, BaseAugmentationConfig],
    aug_ctx: Optional[AugmentationContext] = None,
    centroids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Spatial transformation augmentation.
    """
    T = torch.eye(4, device = device, dtype = dtype).expand(batch_size, -1, -1) 

    if centroids is not None:
        T_c = trans_mat(centroids)
        T_invc = trans_mat(-centroids)

    for aug_name in aug_groups:
        if aug_configs[aug_name].type == AugmentationType.RANDOM_TRANSFORM:
            def generator_fn():
                T_this = generate_random_transform(
                    aug_configs[aug_name],
                    batch_size,
                    device
                )
                return transform_matmul(T_c, transform_matmul(T_this, T_invc)) if centroids is not None else T_this
        
            T = transform_matmul(T, aug_ctx.get_op_param(aug_name, generator_fn))
    
    return T


def augment_color_jitter(
    data: Union[torch.Tensor, List[torch.Tensor]],
    data_type: DataType,
    aug_name: str,
    aug_configs: ColorJitterAugmentationConfig,
    aug_ctx: AugmentationContext,
    **kwargs
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Color Jitter Augmentation.
    """
    if data_type != DataType.IMAGE:
        return data

    if data.ndim == 3:
        data = data.unsqueeze(0)
        is_batch = False
    else:
        is_batch = True

    def gen_params():
        jitter = T.ColorJitter(
            brightness = aug_configs.brightness,
            contrast = aug_configs.contrast,
            saturation = aug_configs.saturation,
            hue = aug_configs.hue
        )
        return jitter.get_params(
            jitter.brightness,
            jitter.contrast,
            jitter.saturation,
            jitter.hue
        )

    params = aug_ctx.get_op_param(aug_name, gen_params)
    fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = params

    for i in range(4):
        idx = fn_idx[i]
        if idx == 0:
            data = TF.adjust_brightness(data, brightness_factor)
        elif idx == 1:
            data = TF.adjust_contrast(data, contrast_factor)
        elif idx == 2:
            data = TF.adjust_saturation(data, saturation_factor)
        elif idx == 3:
            data = TF.adjust_hue(data, hue_factor)

    if not is_batch:
        data = data.squeeze(0)
    return data



def apply_source_augmentation(
    data: torch.Tensor,
    data_type: DataType,
    aug_groups: List[str],
    aug_configs: Dict[str, BaseAugmentationConfig],
    aug_ctx: Optional[AugmentationContext]
) -> torch.Tensor:
    """
    Apply augmentation to source data.
    """
    for aug_name in aug_groups:
        assert aug_configs[aug_name].type != AugmentationType.RANDOM_TRANSFORM
        current_module = sys.modules[__name__]
        augment_fn = getattr(current_module, f"augment_{aug_configs[aug_name].type.value}")
        data = augment_fn(
            data = data,
            data_type = data_type,
            aug_name = aug_name,
            aug_configs = aug_configs[aug_name],
            aug_ctx = aug_ctx
        )
    return data
