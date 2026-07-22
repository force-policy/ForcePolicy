from typing import Any, Dict, List, Union
from dataclasses import dataclass, field

import numpy as np

from runner.configs.agent.base import BaseAgentConfig


@dataclass(kw_only = True)
class LocalAgentConfig(BaseAgentConfig):
    colors: Dict[str, np.ndarray] = field(default_factory = dict)
    depths: Dict[str, np.ndarray] = field(default_factory = dict)
    lowdim: Dict[str, np.ndarray] = field(default_factory = dict)
