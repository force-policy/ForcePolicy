from dataclasses import dataclass


@dataclass(kw_only = True)
class PolicyConfig:
    name: str
