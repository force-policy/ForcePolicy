from dataclasses import dataclass, field

from runner.configs.scheduler.vanilla import VanillaSchedulerConfig


@dataclass(kw_only = True)
class ForceOnlySchedulerConfig(VanillaSchedulerConfig):
    type: str = field(default = "force_only_scheduler", init = False)
    vision_feat_dim: int = 512
