"""
FiLM-conditioned encoders.
"""
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class FiLMLinear(nn.Module):
    """ Linear layer with FiLM. """
    def __init__(
        self, 
        dim_input: int,
        dim_output: int,
        dim_cond: int
    ):
        """ Initialization. """
        super(FiLMLinear, self).__init__()
        self.linear = nn.Linear(dim_input, dim_output)
        self.norm = nn.LayerNorm(dim_output)
        self.act = nn.Mish()
        self.film_gen = nn.Linear(dim_cond, dim_output * 2)

    def forward(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [batch_size, ..., dim_input]
        cond: [batch_size, dim_cond]
        """
        out = self.linear(x)
        out = self.norm(out)
        
        params = self.film_gen(cond)
        gamma, beta = torch.chunk(params, 2, dim = -1)
        
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        
        out = (1 + gamma) * out + beta
        return self.act(out)


class FiLMConv1dBlock(nn.Module):
    """ Conv1d block with FiLM. """
    def __init__(
        self, 
        dim_input: int,
        dim_output: int,
        kernel_size: int = 3,
        dim_cond: int = 512,
        num_groups: int = 8
    ):
        """ Initialization. """
        super().__init__()
        self.conv = nn.Conv1d(dim_input, dim_output, kernel_size, padding = kernel_size // 2, bias = False)
        self.norm = nn.GroupNorm(num_groups, dim_output)
        self.act = nn.Mish()
        self.film_gen = nn.Linear(dim_cond, dim_output * 2)
        
        self.shortcut = nn.Sequential()
        if dim_input != dim_output:
            self.shortcut = nn.Conv1d(dim_input, dim_output, 1)

    def forward(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [batch_size, ..., dim_input]
        cond: [batch_size, dim_cond]
        """
        out = self.conv(x)
        out = self.norm(out)
        
        params = self.film_gen(cond)
        gamma, beta = torch.chunk(params, 2, dim=1)
        
        gamma = gamma.unsqueeze(2)
        beta = beta.unsqueeze(2)
        
        out = (1 + gamma) * out + beta
        out = self.act(out)
        
        return out + self.shortcut(x)



class FiLMResBlock(nn.Module):
    """ Resnet block with FiLM. """
    def __init__(
        self, 
        dim_input: int, 
        dim_output: int, 
        stride: int = 1, 
        kernel_size: int = 3,
        dim_cond: int = 512
    ) -> None:
        """ Initialization. """
        super(FiLMResBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim_input, dim_output, kernel_size = kernel_size, stride = stride, padding = kernel_size // 2, bias = False)
        self.bn1 = nn.BatchNorm2d(dim_output)
        self.conv2 = nn.Conv2d(dim_output, dim_output, kernel_size = kernel_size, stride = 1, padding = kernel_size // 2, bias = False)
        self.bn2 = nn.BatchNorm2d(dim_output)
        self.shortcut = nn.Sequential()
        if stride != 1 or dim_input != dim_output:
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim_input, dim_output, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(dim_output)
            )
        self.film_gen = nn.Linear(dim_cond, dim_output * 2)

    def forward(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [batch_size, C, H, W]
        cond: [batch_size, dim_cond]
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        film_params = self.film_gen(cond)
        gamma, beta = torch.chunk(film_params, 2, dim = -1)

        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)

        out = (1 + gamma) * out + beta
        out += self.shortcut(x)

        return F.relu(out)


class ResNetFiLMEncoder(nn.Module):
    """ Resnet encoder with FiLM. """
    def __init__(
        self, 
        dim_cond: int = 512, 
        img_channels: int = 3, 
        dim_resnet: List[int] = [64, 128, 256],
        kernel_resnet: List[int] = [3, 3, 3],
        stride_resnet: List[int] = [1, 2, 2],
        return_tokens: bool = False
    ) -> None:
        """ Initialization. """
        super().__init__()
        self.conv1 = nn.Conv2d(img_channels, dim_resnet[0], kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(dim_resnet[0])
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layers = nn.ModuleList()
        assert len(dim_resnet) == len(kernel_resnet) == len(stride_resnet)
        for i in range(len(dim_resnet)):
            in_dim = dim_resnet[0] if i == 0 else dim_resnet[i - 1]
            out_dim = dim_resnet[i]
            self.layers.append(FiLMResBlock(
                in_dim, 
                out_dim, 
                stride = stride_resnet[i], 
                kernel_size = kernel_resnet[i],
                dim_cond = dim_cond
            ))
        
        self.return_tokens = return_tokens
        if not return_tokens:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(
        self, 
        x: torch.Tensor, 
        cond: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [batch_size, C, H, W]
        cond: [batch_size, dim_cond]
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        for layer in self.layers:
            out = layer(out, cond)

        if self.return_tokens:
            return rearrange(out, 'b c h w -> b (h w) c')
        else:
            return self.avgpool(out).flatten(1)

