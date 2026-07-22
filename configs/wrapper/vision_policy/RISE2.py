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
        reorganize_configs = [
            ReorganizeConfig(
                type = "pack",
                key = "action",
                key_list = ["action_tcp", "action_ee_command"],
                dim_list = [9, 1]
            )
        ],
        provider_keys = {
            "action": "action"
        }
    ),
    inference_action_provider = DataProviderConfig(
        reorganize_configs = [
            ReorganizeConfig(
                type = "unpack",
                key = "action",
                key_list = ["action_tcp", "action_ee_command"],
                dim_list = [9, 1]
            )
        ],
        provider_keys = {
            "action_tcp": "action_tcp",
            "action_ee_command": "action_ee_command",
            "vision_feat": "vision_feat"
        }
    )
)
