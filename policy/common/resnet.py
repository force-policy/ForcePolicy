import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from einops import rearrange


class ResBlock(nn.Module):
    """ Resnet block. """
    def __init__(
        self, 
        dim_input: int, 
        dim_output: int, 
        stride: int = 1, 
        kernel_size: int = 3,
    ) -> None:
        """ Initialization. """
        super(ResBlock, self).__init__()
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

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [batch_size, C, H, W]
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)

        return F.relu(out)


class ResNetEncoder(nn.Module):
    """ Resnet encoder. """
    def __init__(
        self, 
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
            self.layers.append(ResBlock(
                in_dim, 
                out_dim, 
                stride = stride_resnet[i], 
                kernel_size = kernel_resnet[i]
            ))
        
        self.return_tokens = return_tokens
        if not return_tokens:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [batch_size, C, H, W]
        return: [batch_size, D] or [batch_size, N, D]
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        for layer in self.layers:
            out = layer(out)

        if self.return_tokens:
            return rearrange(out, 'b c h w -> b (h w) c')
        else:
            return self.avgpool(out).flatten(1)
