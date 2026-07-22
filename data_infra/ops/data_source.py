"""
Fetch data from data sources.
"""
from typing import Dict, List, Union, Optional

import torch

from data_infra.configs import (
    DataType,
    PrefixConfig,
    BaseSourceConfig,
    ImageSourceConfig,
    DepthSourceConfig,
    BaseAugmentationConfig,
    Depth2PointSourceConfig
)

from data_infra.ops.point_cloud import get_coordinates
from data_infra.ops.point_cloud import generate_points
from data_infra.ops.augmentation import AugmentationContext
from data_infra.ops.augmentation import apply_source_augmentation


def get_lowdim(
    data_dict: Dict[str, torch.Tensor],
    source_config: BaseSourceConfig,
    prefix_config: PrefixConfig,
    enable_aug: bool,
    aug_configs: Dict[str, BaseAugmentationConfig],
    aug_ctx: Optional[AugmentationContext] = None
) -> torch.Tensor:
    """
    Get lowdim data (including augmentation)
    """
    data = data_dict[source_config.input_key]
 
    if enable_aug:
        data = apply_source_augmentation(
            data = data,
            data_type = source_config.type,
            aug_groups = source_config.aug_groups,
            aug_configs = aug_configs,
            aug_ctx = aug_ctx
        )

    return data


def get_image(
    data_dict: Dict[str, torch.Tensor],
    source_config: Union[ImageSourceConfig, DepthSourceConfig],
    prefix_config: PrefixConfig,
    enable_aug: bool,
    aug_configs: Dict[str, BaseAugmentationConfig],
    aug_ctx: Optional[AugmentationContext] = None
):
    """
    Get image (including augmentation)
    """
    if source_config.type == DataType.IMAGE:
        prefix = prefix_config.color
    elif source_config.type == DataType.DEPTH:
        prefix = prefix_config.depth
    else:
        raise ValueError(f"Invalid image type: {source_config.type}")
    
    color = data_dict[f"{prefix}/{source_config.camera_name}"]
 
    if enable_aug:
        color = apply_source_augmentation(
            data = color,
            data_type = DataType.IMAGE,
            aug_groups = source_config.aug_groups,
            aug_configs = aug_configs,
            aug_ctx = aug_ctx
        )

    return color


def get_depth_points(
    data_dict: Dict[str, torch.Tensor],
    source_config: Depth2PointSourceConfig,
    prefix_config: PrefixConfig,
    enable_aug: bool,
    aug_configs: Dict[str, BaseAugmentationConfig],
    aug_ctx: Optional[AugmentationContext]
) -> List[torch.Tensor]:
    """
    Get points from depths (including augmentation)
    """
    assert source_config.frame in ["world", f"camera/{source_config.camera_name}"]

    depth = data_dict[f"{prefix_config.depth}/{source_config.camera_name}"]
    intrinsic = data_dict[f"{prefix_config.intrinsic}/{source_config.camera_name}"]
    extrinsic = data_dict[f"{prefix_config.extrinsic}/{source_config.camera_name}"]

    if source_config.image_source_config:
        color = get_image(
            data_dict = data_dict,
            source_config = source_config.image_source_config,
            prefix_config = prefix_config,
            enable_aug = enable_aug,
            aug_configs = aug_configs,
            aug_ctx = aug_ctx
        )
    else:
        color = None

    if enable_aug:
        depth = apply_source_augmentation(
            data = depth,
            data_type = DataType.DEPTH,
            aug_groups = source_config.aug_groups,
            aug_configs = aug_configs,
            aug_ctx = aug_ctx
        )
    
    return generate_points(
        depth = depth,
        intrinsic = intrinsic,
        extrinsic = extrinsic,
        frame = source_config.frame,
        color = color,
        size = source_config.size,
        fill_hole = source_config.fill_hole,
        fill_hole_kernel_size = source_config.fill_hole_kernel_size,
        flatten = source_config.flatten,
        camera_mask_config = source_config.camera_mask_config,
        world_mask_config = source_config.world_mask_config,
        pooling_size = source_config.pooling_size,
        pooling_interp_mode = source_config.pooling_interp_mode
    )