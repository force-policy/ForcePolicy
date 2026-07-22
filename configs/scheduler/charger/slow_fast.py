from runner.configs.scheduler import (
    SyncActionConfig,
    WaypointDropoutConfig,
    SlowFastSchedulerConfig
)
from data_infra.configs.point_mask import CubePointMaskConfig
from utils.transforms.rotation import RotationType


scheduler_config = SlowFastSchedulerConfig(
    seed = 233,
    
    planner_freq = 50,
    max_infer_dt = 1.5,
    pre_infer_time = 0.2,
    
    waypoint_dropout = WaypointDropoutConfig(
        enable = True,
        max_dropout = 20,
        weight_linear = 1.0,
        weight_angular = 0.1,
        weight_linear_vel = 0.1,
        weight_angular_vel = 0.0
    ),

    waypoint_interp_type = "linear",
    time_scaling_factor = 1.0,

    sync_action_config = {
        "robot/main/tcp_pose": SyncActionConfig(
            source_frequency = 10,
            source_length = 50,
            interpolation = "pose_linear",
            rotation_rep = RotationType.QUATERNION
        )
    },
    sync_frequency = 10,
    sync_horizon = 50,

    action_params = {
        "robot/main/tcp_pose": {
            "max_vel": 0.5,
            "max_acc": 1.0,
            "max_angular_vel": 1.0,
            "max_angular_acc": 2.0,
            "max_search_force_vel": [0.02, 0.02, 0.02]
        }
    },

    fast_slow_key_mapping = {
        "vision_feat": "vision_feat"
    },
    fast_sync_action_config = {
        "robot/main/tcp_pose": SyncActionConfig(
            source_frequency = 50,
            source_length = 50,
            interpolation = "pose_linear",
            rotation_rep = RotationType.QUATERNION
        ),
        "robot/main/tcp_wrench": SyncActionConfig(
            source_frequency = 50,
            source_length = 50,
            interpolation = "linear",
        ),
        "robot/main/force_frame": SyncActionConfig(
            source_frequency = 50,
            source_length = 1,
            interpolation = "pose_linear",
            rotation_rep = RotationType.QUATERNION
        ),
        "robot/main/force_frame_mask": SyncActionConfig(
            source_frequency = 50,
            source_length = 1,
            interpolation = "nearest",
        ),
        "switch_signal": SyncActionConfig(
            source_frequency = 50,
            source_length = 1,
            interpolation = "nearest",
        )
    },
    fast_sync_horizon = 50,
    fast_frequency = 50,

    switch_duration = 0.5,
    switch_key = "switch_signal"
)

