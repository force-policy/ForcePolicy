"""
Common utilities.
"""
from typing import Any, List, Dict, Union, Optional

import os
import torch
import random
import numpy as np
import torch.nn as nn


def set_seed(seed: int) -> None:
    """ Set seed. """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def to_device(data: Any, device: Union[torch.device, str]) -> Any:
    """ Move tensors to device. """
    if torch.is_tensor(data):
        return data.to(device)
    
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device) for x in data)
    
    return data


def to_tensor(data: Any) -> Any:
    """ Convert to tensor. """
    if isinstance(data, dict):
        return {k: to_tensor(v) for k, v in data.items()} 
    
    elif isinstance(data, (list, tuple)):
        return type(data)(to_tensor(x) for x in data) 

    else:
        if isinstance(data, torch.Tensor) or data is None:
            return data
        return torch.as_tensor(data)


def to_numpy(data: Any) -> Any:
    """ Convert to numpy. """
    if isinstance(data, dict):
        return {k: to_numpy(v) for k, v in data.items()} 
    
    elif isinstance(data, (list, tuple)):
        return type(data)(to_numpy(x) for x in data) 

    else:
        if isinstance(data, torch.Tensor):
            return data.numpy()
        return data


def sample_to_batch(data: Any) -> Any:
    """ Convert sample to batch. """
    if torch.is_tensor(data):
        return data.unsqueeze(0)
    
    elif isinstance(data, dict):
        return {k: sample_to_batch(v) for k, v in data.items()}
    
    elif isinstance(data, (list, tuple)):
        return type(data)(sample_to_batch(x) for x in data)
    
    return data


def batch_to_sample(data: Any) -> Any:
    """ Convert batch to sample. """
    if torch.is_tensor(data):
        return data.squeeze(0)
    
    elif isinstance(data, dict):
        return {k: batch_to_sample(v) for k, v in data.items()}
    
    elif isinstance(data, (list, tuple)):
        return type(data)(batch_to_sample(x) for x in data)
    
    return data


def num_trainable_params(module: nn.Module) -> int:
    """ Number of trainable parameters. """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def num_params(module: nn.Module) -> int:
    """ Number of parameters. """
    return sum(p.numel() for p in module.parameters())