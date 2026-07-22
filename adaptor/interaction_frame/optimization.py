"""
Optimization for Interaction Frame Selection

Reference: 
- Overbeek, Alex Harm Gert-Jan, et al. "Identifying Physical Interactions in Contact-Based Robot Manipulation for Learning from Demonstration." Advanced Robotics Research (2025): e202500109.
"""
from typing import List, Union, Optional

import sys
import torch
import numpy as np
import torch.nn as nn

from logger import logger
from utils.transforms.rotation import RotationType
from utils.transforms.projection import apply_mat_to_twist, apply_mat_to_wrench
from utils.transforms.pose import rotation_transform, mat_to_xyz_rot, xyz_rot_to_mat


def log_power_sum_loss(wrench, twist):
    """ Compute log power loss for given if_frame. """
    N = len(wrench)
    power_if = torch.mul(wrench, twist)
    power_if_segment = torch.linalg.norm(power_if, ord = 2, dim = 0) * (N ** -0.5)
    full_power_if_segment = torch.log(power_if_segment).sum()
    return full_power_if_segment


def power_prod_loss(wrench, twist):
    """ Compute power product loss for given if_frame. """
    N = len(wrench)
    power_if = torch.mul(wrench, twist)
    power_if_segment = torch.linalg.norm(power_if, ord = 2, dim = 0) * (N ** -0.5)
    full_power_if_segment = power_if_segment.prod()
    return full_power_if_segment


def power_l1_loss(wrench, twist):
    """ Compute power L1 loss for given if_frame. """
    N = len(wrench)
    power_if = torch.mul(wrench, twist)
    power_magnitude = torch.linalg.norm(power_if, dim = 0)  * (N ** -0.5)
    return torch.sum(torch.abs(power_magnitude))


def general_diagonal_loss(wrench, twist):
    """ Compute general diagonal loss for given if_frame. """
    Cw = wrench.T @ wrench
    Ct = twist.T @ twist
    mask_w = torch.ones_like(Cw)
    mask_t = torch.ones_like(Ct)
    mask_w.diagonal().fill_(0)
    mask_t.diagonal().fill_(0)
    return torch.sum(torch.abs(Cw) * mask_w) + torch.sum(torch.abs(Ct) * mask_t)
    

def disentanglement_loss(twist, wrench):
    """ Compute disentanglement loss. """
    weights = {'v_align_x': 1000.0, 'f_align_z': 5.0}

    v = twist[:, :3]
    f = wrench[:, :3]
    w = twist[:, 3:]
    
    loss_v_align = torch.mean(v[:, 1]**2 + v[:, 2]**2)

    v_norm = torch.norm(v, dim=1, keepdim=True) + 1e-6
    v_dir = v / v_norm
    
    f_friction_mag = torch.sum(f * v_dir, dim=1, keepdim=True)
    f_friction = f_friction_mag * v_dir
    f_constraint = f - f_friction
    
    loss_f_align = torch.mean(f_constraint[:, 0]**2 + f_constraint[:, 1]**2)


    return weights['v_align_x'] * loss_v_align + \
           weights['f_align_z'] * loss_f_align


def optimize_interaction_frame(
    wrench: np.ndarray,
    twist: np.ndarray,
    tcp: np.ndarray,
    with_origin: bool = False,
    num_steps: int = 100,
    lr: float = 0.01,
    init_guess: Optional[np.ndarray] = None,
    loss_types: List[str] = ["general_diagonal"],
    loss_weights: List[float] = [1.0],
) -> np.ndarray:
    """
    Optimize interaction frame for given wrench and twist using power.

    Inputs:
        wrench (np.ndarray): [N, 6] wrench in tcp frame.
        twist (np.ndarray): [N, 6] twist in tcp frame.
        tcp (np.ndarray): [N, 4, 4] tcp in base frame.
        with_origin (bool): Whether to optimize the origin of the interaction frame.
        init_guess (np.ndarray): [4, 4] initial guess for interaction frame.
        loss_types (List of str): 
        loss_weights (List of float):
    Outputs:
        if_frame (np.ndarray): [4, 4] interaction frame in TCP.
    """
    N = len(wrench)
    wrench = torch.from_numpy(wrench).to(torch.float32)
    twist = torch.from_numpy(twist).to(torch.float32)
    tcp = torch.from_numpy(tcp).to(torch.float32)

    def generate_transform_matrix(x):
        """ Generate matrix from rotation. """
        if with_origin:
            res = xyz_rot_to_mat(x, rotation_rep = RotationType.ROTATION_6D)
        else:
            rot_mat = rotation_transform(x, from_rep = RotationType.ROTATION_6D, to_rep = RotationType.MATRIX)
            res = torch.eye(4, device = rot_mat.device, dtype = rot_mat.dtype)
            res[:3, :3] = rot_mat            
        return res
    
    current_module = sys.modules[__name__]

    def loss_func(params):
        """ Compute loss for given params and loss types. """
        T_tcp_if = generate_transform_matrix(params)
        wrench_if = apply_mat_to_wrench(wrench, mat = torch.linalg.inv(T_tcp_if), rotation_only = False)
        twist_if = apply_mat_to_twist(twist, mat = torch.linalg.inv(T_tcp_if), rotation_only = False)
        loss = 0
        loss_dict = {}
        for loss_type, weight in zip(loss_types, loss_weights):
            loss_dict[loss_type] = getattr(current_module, f"{loss_type}_loss")(wrench_if, twist_if)
            loss = loss + loss_dict[loss_type] * weight
        return loss, loss_dict

    if init_guess is None:
        if with_origin:
            init_guess = torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype = torch.float32)
        else:
            init_guess = torch.tensor([1, 0, 0, 0, 1, 0], dtype = torch.float32)
    else:
        init_guess = mat_to_xyz_rot(
            init_guess, 
            rotation_rep = RotationType.ROTATION_6D
        )
        if not with_origin:
            init_guess = init_guess[3:]
        init_guess = torch.tensor(init_guess, dtype = torch.float32)

    init_guess.requires_grad = True
    params = nn.Parameter(init_guess)
    optimizer = torch.optim.Adam([params], lr = lr)

    for i in range(num_steps):
        optimizer.zero_grad()
        loss, loss_dict = loss_func(params)
        info = f"step = {i}, loss = {loss.item():.4f}"
        for loss_type, val in loss_dict.items():
            info += f", {loss_type} = {val.item():.4f}"
        logger.debug(info)
        loss.backward()
        optimizer.step()
    
    if_frame = generate_transform_matrix(params.detach())
    return if_frame.numpy()
