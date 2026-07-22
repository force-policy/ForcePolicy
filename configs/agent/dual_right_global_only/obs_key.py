from runner.configs.agent import AgentObsKeysConfig


obs_key_configs = {
    "vision_RISE2": AgentObsKeysConfig(
        color_keys = ["main"],
        depth_keys = ["main"],
        intrinsic_keys = ["main"],
        extrinsic_keys = ["main"],
        robot_pose_keys = ["right"],
        lowdim_provider_keys = []
    )
}

