from policy.configs.wrapper import *


wrapper_config = PolicyWrapperConfig(
    obs_provider = DataProviderConfig(
        reorganize_configs = [],
        provider_keys = {
            "vision_feat": "vision_feat",
            "image_wrist": "image_wrist",
            "proprio": "proprio_tcp",
            "force_torque": "force_torque"
        }
    ),
    train_action_provider = DataProviderConfig(
        provider_keys = {
            "action": "action_tcp",
            "force": "action_force_torque",
            "frame": "action_frame",
            "mask": "action_frame_mask"
        }
    ),
    inference_action_provider = DataProviderConfig(
        provider_keys = {
            "action_tcp": "action",
            "action_force_torque": "force",
            "action_frame": 'frame',
            "action_frame_mask": "mask",
            "switch_signal": "switch_signal"
        }
    )
)
