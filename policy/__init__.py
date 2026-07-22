from typing import Type

import torch.nn as nn

from policy.wrapper.wrapper import PolicyWrapper
from policy.configs import PolicyConfig

from policy.vision_policy.RISE import RISE
from policy.vision_policy.RISE2 import RISE2
from policy.vision_policy.diffusion_policy import DiffusionPolicyCNN

from policy.force_policy import ForcePolicy

POLICY_MAP = {
    "RISE": RISE,
    "RISE2": RISE2,
    "DiffusionPolicyCNN": DiffusionPolicyCNN,
    "ForcePolicy": ForcePolicy
}


def get_policy(config: PolicyConfig) -> nn.Module:
    if config.name not in POLICY_MAP:
        raise ValueError(f"Policy {config.name} not found in POLICY_MAP. Available policies: {list(POLICY_MAP.keys())}")
    policy_cls = POLICY_MAP[config.name]
    return policy_cls(config)
