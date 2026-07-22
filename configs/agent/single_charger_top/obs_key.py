from runner.configs.agent import AgentObsKeysConfig


obs_key_configs = {
    "vision_RISE2": AgentObsKeysConfig(
        color_keys = ["main"],
        depth_keys = ["main"],
        intrinsic_keys = ["main"],
        extrinsic_keys = ["main"],
        robot_pose_keys = ["main"],
        lowdim_provider_keys = []
    ),
    "force_policy": AgentObsKeysConfig(
        color_keys = ["wrist"],
        depth_keys = [],
        intrinsic_keys = [],
        extrinsic_keys = [],
        robot_pose_keys = ["main"],
        lowdim_provider_keys = ["tcp_pose", "force_torque", "aux_tcp_pose"]
    )
}
