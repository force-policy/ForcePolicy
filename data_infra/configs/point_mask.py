"""
Point Mask.
"""
from typing import Literal, Tuple, List, Optional, Union
from dataclasses import dataclass, field

import torch


@dataclass(kw_only = True)
class PointMaskConfig:
    """ Base Configuration """
    mask_type: Literal["inside", "outside"] = "inside"

    def get_mask(self, points: torch.Tensor) -> torch.Tensor:
        if self.mask_type == "inside":
            return torch.ones(points.shape[:-1], dtype = torch.bool, device = points.device)
        else:
            return torch.zeros(points.shape[:-1], dtype = torch.bool, device = points.device)


@dataclass(kw_only = True)
class CubePointMaskConfig(PointMaskConfig):
    """ Cube Mask """
    min_bounds: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    max_bounds: Tuple[float, float, float] = ( 1.0,  1.0,  1.0)

    def get_mask(self, points: torch.Tensor) -> torch.Tensor:
        x, y, z = points[..., 0], points[..., 1], points[..., 2]
        in_bound = (x >= self.min_bounds[0]) & (x <= self.max_bounds[0]) & \
                   (y >= self.min_bounds[1]) & (y <= self.max_bounds[1]) & \
                   (z >= self.min_bounds[2]) & (z <= self.max_bounds[2])
        return in_bound if self.mask_type == "inside" else ~in_bound

@dataclass(kw_only = True)
class SpherePointMaskConfig(PointMaskConfig):
    """ Sphere Mask """
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0

    def get_mask(self, points: torch.Tensor) -> torch.Tensor:
        center = torch.tensor(self.center, device = points.device, dtype = points.dtype)
        dist_sq = torch.sum((points - center) ** 2, dim = -1)
        in_bound = dist_sq <= (self.radius ** 2)
        return in_bound if self.mask_type == "inside" else ~in_bound


@dataclass(kw_only = True)
class GroupPointMaskConfig(PointMaskConfig):
    children: List[PointMaskConfig] = field(default_factory = list)
    operation: Literal["and", "or"] = "and"

    def get_mask(self, points: torch.Tensor) -> torch.Tensor:
        if not self.children:    
            if mask_type == "inside":
                return torch.ones(points.shape[:-1], dtype = torch.bool, device = points.device)
            else:
                return torch.zeros(points.shape[:-1], dtype = torch.bool, device = points.device)

        final_mask = self.children[0].get_mask(points)
        for child in self.children[1:]:
            child_mask = child.get_mask(points)
            if self.operation == "and":
                final_mask = final_mask & child_mask
            elif self.operation == "or":
                final_mask = final_mask | child_mask
        
        return final_mask if self.mask_type == "inside" else ~final_mask
