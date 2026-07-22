"""
Ensemble buffer configuration
"""
from typing import Any, Dict
from dataclasses import dataclass, field


@dataclass(kw_only = True)
class EnsembleBufferConfig:
    params_dict: Dict[str, Dict[str, Any]] = field(default_factory = dict)
