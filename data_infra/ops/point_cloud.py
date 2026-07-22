
from typing import List, Dict, Tuple, Optional, Union, Literal

import torch
import torch.nn.functional as F

from einops import rearrange

from data_infra.configs import (
    PointMaskConfig, 
    BaseSourceConfig,
    GroupPointMaskConfig
)

from utils.transforms.projection import apply_mat_to_point


def fill_depth_hole(grid_points, mask, kernel_size = [60, 64]):
    """
    Batch version of hole filling.
    
    Args:
        grid_points: (B, H, W, 3) 
        mask: (B, H, W) Boolean Tensor
    Returns:
        grid_points: (B, H, W, 3)
    """
    B, H, W, C = grid_points.shape
    h, w = H // kernel_size[0], W // kernel_size[1]
    while H % h != 0: h += 1
    while W % w != 0: w += 1
    k_h, k_w = H // h, W // w

    grid_view = grid_points.view(B, h, k_h, w, k_w, C)
    grid_view = grid_view.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, h, w, -1, C)
    mask_view = mask.view(B, h, k_h, w, k_w)
    mask_view = mask_view.permute(0, 1, 3, 2, 4).contiguous().view(B, h, w, -1)

    valid_count = mask_view.float().sum(dim = -1, keepdim = True)
    points_mean = grid_view.sum(dim = -2) / (valid_count + 1e-6)
    points_mean_expanded = points_mean.unsqueeze(-2).expand(-1, -1, -1, k_h * k_w, -1)

    grid_view[~mask_view] = points_mean_expanded[~mask_view]
    grid_points = grid_view.view(B, h, w, k_h, k_w, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return grid_points


def get_coordinates(
    depth: torch.Tensor, 
    intrinsic: torch.Tensor, 
    fill_hole: bool = True,
    img_size: Optional[Tuple[int, int]] = None,
    fill_hole_kernel_size: Tuple[int, int] = (60, 64)
) -> torch.Tensor:
    """
    Depth-based coordinate generation.
    
    Args:
        depth: (B, H, W) or (H, W) Tensor
        intrinsic: (B, 3, 3) Tensor (supports (3, 3) via broadcasting)
    Returns:
        points: (B, H, W, 3)
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
    if depth.dim() == 3:
        depth = depth.unsqueeze(1)
        
    B, _, H_orig, W_orig = depth.shape
    device = depth.device

    if intrinsic.dim() == 2:
        intrinsic = intrinsic.unsqueeze(0).expand(B, -1, -1)

    if img_size is not None and (img_size[0] != H_orig or img_size[1] != W_orig):
        depth = F.interpolate(depth, size = img_size, mode = 'nearest')
        H, W = img_size
        rescale_h = H / H_orig
        rescale_w = W / W_orig
    else:
        H, W = H_orig, W_orig
        rescale_h, rescale_w = 1.0, 1.0
    
    depth = depth.squeeze(1)

    fx = (intrinsic[:, 0, 0] * rescale_w).view(B, 1, 1)
    fy = (intrinsic[:, 1, 1] * rescale_h).view(B, 1, 1)
    cx = (intrinsic[:, 0, 2] * rescale_w).view(B, 1, 1)
    cy = (intrinsic[:, 1, 2] * rescale_h).view(B, 1, 1)

    y_range = torch.arange(H, device = device, dtype = torch.float32)
    x_range = torch.arange(W, device = device, dtype = torch.float32)
    xmap, ymap = torch.meshgrid(x_range, y_range, indexing = 'xy')
    
    xmap = xmap.unsqueeze(0)
    ymap = ymap.unsqueeze(0)

    points_z = depth
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    points = torch.stack([points_x, points_y, points_z], dim = -1)  # (B, H, W, 3)

    if fill_hole:
        depth_mask = depth > 0  # (B, H, W)
        points = fill_depth_hole(points, depth_mask)

    return points


def generate_points(
    depth: torch.Tensor, 
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    frame: Literal["world", "camera"] = "world",
    color: Optional[torch.Tensor] = None,
    size: Optional[Tuple[int, int]] = None,
    fill_hole: bool = False,
    fill_hole_kernel_size: Optional[Tuple[int, int]] = None,
    flatten: bool = True,
    camera_mask_config: Optional[Union[PointMaskConfig, GroupPointMaskConfig, List[PointMaskConfig]]] = None,
    world_mask_config: Optional[Union[PointMaskConfig, GroupPointMaskConfig, List[PointMaskConfig]]] = None,
    pooling_size: Optional[Tuple[int, int]] = None,
    pooling_interp_mode: str = "area"
) -> List[torch.Tensor]:
    """
    Generate point cloud from depth.
    
    Args:
        depth: (B, H, W) or (H, W)
        intrinsic: (B, 3, 3) or (3, 3)
        extrinsic: (B, 4, 4) or (4, 4), Transformation from world to camera (T_world_cam).
        frame: "world" or "camera", points in world frame or camera frame.
        color: (B, C, H, W) or (C, H, W)
        size: target point resolution.
        fill_hole: whether to fill holes in depth images
        fill_hole_kernel_size: kernel size during hole filling
        flatten: bool, whether to flatten points to (..., N, 3)
        camera_mask_config: point mask applied in camera frame.
        world_mask_config: point mask applied in world frame.
        pooling_size: pooling target size.
        pooling_interp_mode: pooling interpolation mode.
    """
    if depth.ndim == 2:
        no_batch = True
        depth = depth.unsqueeze(0)
    else:
        no_batch = False
    B, H, W = depth.shape
    device = depth.device

    # Extrinsics
    if extrinsic.ndim == 2:
        extrinsic = extrinsic.unsqueeze(0).expand(B, -1, -1)

    # Color processing
    if color is not None:
        if color.ndim == 3: # (C, H, W) -> (1, C, H, W)
            color = color.unsqueeze(0)
        if size is not None and size != (color.shape[2], color.shape[3]):
            color = F.interpolate(color, size = size, mode = 'bilinear')  # (B, C, H, W)
        color = rearrange(color, 'b c h w -> b h w c')

    # Get depth points
    points = get_coordinates(
        depth = depth,
        intrinsic = intrinsic,
        img_size = size,
        fill_hole = fill_hole,
        fill_hole_kernel_size = fill_hole_kernel_size
    ) # (B, H, W, 3) in camera frame
    
    if not flatten:
        # Retain original (B, H, W, ...) shape

        # Pooling if any
        if pooling_size and pooling_size != (points.shape[2], points.shape[3]):
            assert color is None  # If pooling, the shape must mismatch.
            points = F.interpolate(
                rearrange(points, 'b h w c -> b c h w'),
                size = pooling_size,
                mode = pooling_interp_mode
            )
            points = rearrange(points, 'b c h w -> b h w c')

        # Frame projection
        if frame == "world":
            points = apply_mat_to_point(points, extrinsic, coord_first = False) # (B, H, W, 3)
        
        # Concatenate with color
        if color is not None:
            points = torch.cat([points, color], dim = -1) # (B, H, W, 3 + 3)
        
        return points.squeeze(0) if no_batch else points

    else:
        # Flatten (H, W) into N = H * W points, and also apply masks.
        points = rearrange(points, 'b h w c -> b (h w) c')
        color_flat = None if color is None else rearrange(color, 'b h w c -> b (h w) c')

        # Resolve point mask configurations
        def _resolve_point_mask_config(cfg):
            if cfg is None:
                return PointMaskConfig(mask_type = "inside")
            if isinstance(cfg, list):
                return GroupPointMaskConfig(children = cfg, operation = "and")
            return cfg

        # Valid point mask and camera frame point mask 
        cam_mask = _resolve_point_mask_config(camera_mask_config)   
        valid_mask = (points[..., 2] > 1e-4) & (torch.isfinite(points[..., 2]))
        valid_mask = valid_mask & cam_mask.get_mask(points)

        # World space mask
        if world_mask_config:
            world_mask = _resolve_point_mask_config(world_mask_config)
            points_world = apply_mat_to_point(points, extrinsic, coord_first = False)
            valid_mask = valid_mask & world_mask.get_mask(points_world)
        
        # Frame projection
        if frame == "world":
            if world_mask_config:
                points = points_world
            else:
                points = apply_mat_to_point(points, extrinsic, coord_first = False)

        # Filtering
        out_points_list = []
        out_colors_list = [] if color is not None else None

        for b in range(B):
            mask = valid_mask[b]
            if color_flat is not None:
                out_points_list.append(torch.cat([points[b][mask], color_flat[b][mask]], dim = -1))
            else:
                out_points_list.append(points[b][mask])

        return out_points_list[0] if no_batch else out_points_list


def voxel_downsample(
    pcd: torch.Tensor, 
    voxel_size: float
) -> torch.Tensor:
    """
    Voxel downsampling point clouds.
    """
    points = pcd[..., :3]
    feats = pcd[..., 3:]

    min_bound = points.min(dim = 0).values
    quantized = torch.floor((points - min_bound) / voxel_size).long()
    _, inverse_indices = torch.unique(quantized, dim = 0, return_inverse = True, sorted = False)
    M = torch.max(inverse_indices) + 1

    pooled_points = torch.zeros((M, 3), device = points.device, dtype = points.dtype)
    pooled_features = torch.zeros((M, feats.shape[-1]), device = points.device, dtype = feats.dtype)
    counts = torch.zeros((M, 1), device = points.device, dtype = points.dtype)

    pooled_points.index_add_(0, inverse_indices, points)
    pooled_features.index_add_(0, inverse_indices, feats)
    counts.index_add_(0, inverse_indices, torch.ones((points.shape[0], 1), device = points.device))

    counts = counts.clamp(min = 1.0)
    return torch.cat([pooled_points / counts, pooled_features / counts], dim = -1)


def _farthest_point_sampling(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Naive Iterative FPS.
    TODO: Replace with more efficient implementation.
    """
    N, _ = points.shape
    device = points.device
    
    if N <= n_samples:
        return torch.arange(N, device = device)
        
    centroids = torch.zeros(n_samples, dtype = torch.long, device = device)
    distance = torch.ones(N, device = device) * 1e10
    farthest = torch.randint(0, N, (1,), dtype = torch.long, device = device).item()
    
    for i in range(n_samples):
        centroids[i] = farthest
        centroid = points[farthest, :].view(1, 3)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, -1).item()
        
    return centroids


def fixed_number_downsample(
    pcd: torch.Tensor, 
    fixed_number: int,
    sampling_method: Literal['fps', 'random'] = "random"
) -> torch.Tensor:
    """
    Fixed number downsampling point clouds.
    """
    points = pcd[..., :3]
    N = pcd.shape[0]
    
    if N >= fixed_number:
        if sampling_method == 'fps':
            indices = _farthest_point_sampling(points, fixed_number)
        else:
            indices = torch.randperm(N, device = pcd.device)[:fixed_number]
        return pcd[indices]
    
    else:
        if N > 0:
            indices = torch.randint(0, N, (fixed_number - N,), device = pcd.device)
            pcd_pad = pcd[indices]
            return torch.cat([pcd, pcd_pad], dim = 0)
        else:
            return torch.zeros((fixed_number, pcd.shape[1]), device = pcd.device)
