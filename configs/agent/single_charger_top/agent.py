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
robot_shm_name = "Rizon4-063047"
camera_shm_name = "camera"
wrist_camera_shm_name = "wrist_camera"

robot = FlexivArm("Rizon4-063047", shm_freq = 100, shm_name = robot_shm_name)
camera = RealSenseRGBDCamera("035622060973", streaming_freq = 15, shm_freq = 15, shm_name = camera_shm_name)
wrist_camera = RealSenseRGBDCamera("104122062823", streaming_freq = 15, shm_freq = 15, shm_name = wrist_camera_shm_name)

camera_shm = DictSharedMemoryManager(
    type = 1, 
    dict_cfg = generate_config_from_data(
        camera.get_states(), 
        shm_name = camera_shm_name
    )
) 
wrist_camera_shm = DictSharedMemoryManager(
    type = 1,
    dict_cfg = generate_config_from_data(
        wrist_camera.get_states(),
        shm_name = wrist_camera_shm_name
    )
)

agent_config = RealAgentConfig(
    platform_config = PlatformConfig(
        camera_names = ["main", "wrist"],
        robot_names = ["main"]
    ),
    robots = {"main": robot},
    grippers = {},
    cameras = {"main": camera_shm, "wrist": wrist_camera_shm},
    camera_shm_names = {"main": camera_shm_name, "wrist": wrist_camera_shm_name},
    intrinsics = {
        "main": camera.get_intrinsic(return_mat = True),
        "wrist": wrist_camera.get_intrinsic(return_mat = True)
    },
    extrinsics = {
        "main": np.array([
            [-9.89660068e-03, -9.99687396e-01,  2.29601309e-02, -7.38618282e-01],
            [-9.99950547e-01,  9.87146858e-03, -1.20768216e-03,  5.37614720e-02],
            [ 9.80654426e-04, -2.29709474e-02, -9.99735652e-01,  1.07856919e+00],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ], dtype = np.float32)
    },
    robot_poses = {"main": np.eye(4, dtype = np.float32)},
    force_control_robot_names = ["main"],
    torque_control_robot_names = [],
    agent_action_providers = {
        "robot/main/tcp_pose": "action_tcp",
        "robot/main/tcp_wrench": "action_force_torque",
        "robot/main/force_frame": "action_frame",
        "robot/main/force_frame_mask": "action_frame_mask",
        "vision_feat": "vision_feat",
        "switch_signal": "switch_signal"
    },
    pose_action_filter_configs = {
        "robot/main/tcp_pose": PoseOneEuroFilterConfig(
            trans = OneEuroFilterConfig(mincutoff = 1.0, beta = 0.5, dcutoff = 1.0),
            rot = RotationOneEuroFilterConfig(mincutoff = 1.0, beta = 0.5, dcutoff = 1.0, rotation_rep = RotationType.QUATERNION)
        )
    },
    vector_action_filter_configs = {},

    lowdim_recorder_configs = {
        "tcp_pose": LowdimRecorderConfig(
            device_ref = "robot/main",
            device_func = "get_tcp_pose",
            max_record_frequency = 200.0,
            max_window_time = 10.0
        ),
        "force_torque": LowdimRecorderConfig(
            device_ref = "robot/main",
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
        "force_torque": LowdimObservationConfig(
            recorder_key = "force_torque",
            freq = 50,
            length = 100,
            time_reversed = False
        ),
        "aux_tcp_pose": LowdimObservationConfig(
            recorder_key = "tcp_pose",
            freq = 50,
            length = 1,
            time_reversed = False
        )
    }
)


