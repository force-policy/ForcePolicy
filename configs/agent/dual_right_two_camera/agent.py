import numpy as np

from easyrobot.arm.flexiv import FlexivArm
from easyrobot.gripper.flexiv import FlexivGripper
from easyrobot.camera.realsense import RealSenseRGBDCamera
from easyrobot.utils.shared_memory import DictSharedMemoryManager, generate_config_from_data

from utils.transforms.rotation import RotationType
from runner.configs.agent import PlatformConfig
from runner.configs.agent.real_agent import RealAgentConfig
from runner.configs.filter import PoseOneEuroFilterConfig, OneEuroFilterConfig, RotationOneEuroFilterConfig
from runner.configs.lowdim_provider import LowdimRecorderConfig, LowdimObservationConfig

# Define shm_names as variables for reuse
robot_shm_name = "Rizon4R-062046"
camera_main_shm_name = "camera_main"
camera_wrist_right_shm_name = "camera_wrist_right"
robot = FlexivArm("Rizon4R-062046", shm_freq = 100, shm_name = robot_shm_name)
gripper = FlexivGripper(robot, shm_freq = 100, gripper_name='Flexiv-GN01', is_init = False)
camera_main = RealSenseRGBDCamera("104122060902", streaming_freq = 15, shm_freq = 15, shm_name = camera_main_shm_name)
camera_wrist_right = RealSenseRGBDCamera("104122064489", streaming_freq = 15, shm_freq = 15, shm_name = camera_wrist_right_shm_name)

camera_shm_main = DictSharedMemoryManager(
    type = 1, 
    dict_cfg = generate_config_from_data(
        camera_main.get_states(), 
        shm_name = camera_main_shm_name
    )
) 
camera_shm_wrist_right = DictSharedMemoryManager(
    type = 1, 
    dict_cfg = generate_config_from_data(
        camera_wrist_right.get_states(), 
        shm_name = camera_wrist_right_shm_name
    )
)

agent_config = RealAgentConfig(
    platform_config = PlatformConfig(
        camera_names = ["main","wrist"],
        robot_names = ["right"],
        gripper_names = ["right"]
    ),
    robots = {"right": robot},
    grippers = {"right": gripper},
    cameras = {"main": camera_shm_main, "wrist": camera_shm_wrist_right},
    camera_shm_names = {"main": camera_main_shm_name, "wrist": camera_wrist_right_shm_name},
    colors = ["main","wrist"],
    depths = ["main"],
    intrinsics = {"main": camera_main.get_intrinsic(return_mat = True)},
    extrinsics = {
        "main": np.array([
            # [ 0.03710599, -0.80679006 , 0.58967189 , 0.08142478],
            # [-0.99929998, -0.03277055 , 0.01804574 , 0.20469146],
            # [ 0.00476475, -0.58992871 ,-0.80744128 , 0.34889845],
            # [ 0.         , 0.         , 0.         , 1.        ]
            [ 0.02424299, -0.80195241,  0.59689581,  0.07783932],   # new correct calib
            [-0.99968406, -0.01548308,  0.01980016,  0.20788143],
            [-0.006637  , -0.59718724, -0.8020744 ,  0.34723684],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ], dtype = np.float32)
    },
    robot_poses = {"right": np.eye(4, dtype = np.float32)},
    force_control_robot_names = ["right"],
    torque_control_robot_names = [],
    agent_action_providers = {
        "robot/right/tcp_pose": "action_tcp",
        "gripper/right/width": "action_ee_command",
        "robot/right/tcp_wrench": "action_force_torque",
        "robot/right/force_frame": "action_frame",
        "robot/right/force_frame_mask": "action_frame_mask",
        "vision_feat": "vision_feat",
        "switch_signal": "switch_signal"
    },
    pose_action_filter_configs = {
        "robot/right/tcp_pose": PoseOneEuroFilterConfig(
            trans = OneEuroFilterConfig(mincutoff = 1.0, beta = 0.5, dcutoff = 1.0),
            rot = RotationOneEuroFilterConfig(mincutoff = 1.0, beta = 0.5, dcutoff = 1.0, rotation_rep = RotationType.QUATERNION)
        )
    },
    vector_action_filter_configs = {
        "gripper/right/width": OneEuroFilterConfig(
            mincutoff = 1.0,
            beta = 0.0,
            dcutoff = 1.0
        )
    },

    lowdim_recorder_configs = {
        "tcp_pose": LowdimRecorderConfig(
            device_ref = "robot/right",
            device_func = "get_tcp_pose",
            max_record_frequency = 200.0,
            max_window_time = 10.0
        ),
        "gripper_width": LowdimRecorderConfig(
            device_ref = "gripper/right",
            device_func = "get_width",
            max_record_frequency = 200.0,
            max_window_time = 10.0
        ),
        "force_torque": LowdimRecorderConfig(
            device_ref = "robot/right",
            device_func = "get_force_torque_tcp",
            max_record_frequency = 200.0,
            max_window_time = 10.0
        )
    },

    lowdim_observation_configs = {
        "tcp_pose": LowdimObservationConfig(
            recorder_key = "tcp_pose",
            freq = 50,
            length = 100,
            time_reversed = False
        ),
        "gripper_width": LowdimObservationConfig(
            recorder_key = "gripper_width",
            freq = 50,
            length = 1,
            time_reversed = False
        ),
        "aux_tcp_pose": LowdimObservationConfig(
            recorder_key = "tcp_pose",
            freq = 50,
            length = 1,
            time_reversed = False
        ),
        "force_torque": LowdimObservationConfig(
            recorder_key = "force_torque",
            freq = 50,
            length = 100,
            time_reversed = False
        )
    } 

)

