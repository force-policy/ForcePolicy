from policy.configs.wrapper import *


wrapper_config = PolicyWrapperConfig(
    obs_provider = DataProviderConfig(
        reorganize_configs = [],
        provider_keys = {
            "cloud": "point_cloud",
            "image": "image",
            "image_coord": "image_coord"
        }
    ),
    train_action_provider = DataProviderConfig(
        reorganize_configs = [],
        provider_keys = {
            "action": "action_tcp"
        }
    ),
    inference_action_provider = DataProviderConfig(
        reorganize_configs = [],
        provider_keys = {
            "action_tcp": "action",
            "vision_feat": "vision_feat"
        }
    )
)
