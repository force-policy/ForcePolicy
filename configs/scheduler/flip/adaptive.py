"""
AdaptiveScheduler configuration for cable5 task.
Features: DTW waypoint dropout, dynamic motion, time-profiled trajectory.
"""
from runner.configs.scheduler import (
    SyncActionConfig,
    AdaptiveSchedulerConfig,
    WaypointDropoutConfig,
)
from data_infra.configs.point_mask import CubePointMaskConfig
from utils.transforms.rotation import RotationType


scheduler_config = AdaptiveSchedulerConfig(
    seed = 233,
    
    planner_freq = 100,
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

    waypoint_interp_type = "acceleration_continuous",
    time_scaling_factor = 1.0,

    sync_action_config = {
        "robot/right/tcp_pose": SyncActionConfig(
            source_frequency = 10,
            source_length = 50,
            interpolation = "pose_linear",
            rotation_rep = RotationType.QUATERNION
        )
    },
    sync_frequency = 10,
    sync_horizon = 50,

    action_params = {
        "robot/right/tcp_pose": {
            "max_vel": 0.5,
            "max_acc": 1.0,
            "max_angular_vel": 1.0,
            "max_angular_acc": 2.0
        }
    },
)

