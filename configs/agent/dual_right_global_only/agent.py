import numpy as np

from easyrobot.arm.flexiv import FlexivArm
from easyrobot.gripper.flexiv import FlexivGripper
from easyrobot.camera.realsense import RealSenseRGBDCamera
from easyrobot.utils.shared_memory import DictSharedMemoryManager, generate_config_from_data

from utils.transforms.rotation import RotationType
from runner.configs.agent import PlatformConfig
from runner.configs.agent.real_agent import RealAgentConfig
from runner.configs.filter import PoseOneEuroFilterConfig, OneEuroFilterConfig, RotationOneEuroFilterConfig


# Define shm_names as variables for reuse
robot_shm_name = "Rizon4R-062046"
camera_shm_name = "camera"

robot = FlexivArm("Rizon4R-062046", shm_freq = 100, shm_name = robot_shm_name)
gripper = FlexivGripper(robot, shm_freq = 100, gripper_name='Flexiv-GN01', is_init = False)
camera = RealSenseRGBDCamera("104122060902", streaming_freq = 15, shm_freq = 15, shm_name = camera_shm_name)

camera_shm = DictSharedMemoryManager(
    type = 1, 
    dict_cfg = generate_config_from_data(
        camera.get_states(), 
        shm_name = camera_shm_name
    )
) 

agent_config = RealAgentConfig(
    platform_config = PlatformConfig(
        camera_names = ["main"],
        robot_names = ["right"],
        gripper_names = ["right"]
    ),
    robots = {"right": robot},
    grippers = {"right": gripper},
    cameras = {"main": camera_shm},
    camera_shm_names = {"main": camera_shm_name},
    colors = ["main"],
    depths = ["main"],
    intrinsics = {"main": camera.get_intrinsic(return_mat = True)},
    extrinsics = {
        "main": np.array([
            [ 0.02424299, -0.80195241,  0.59689581,  0.07783932],
            [-0.99968406, -0.01548308,  0.01980016,  0.20788143],
            [-0.006637  , -0.59718724, -0.8020744 ,  0.34723684],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ], dtype = np.float32)
    },
    robot_poses = {"right": np.eye(4, dtype = np.float32)},
    force_control_robot_names = ["right"],
    torque_control_robot_names = ["right"],
    agent_action_providers = {
        "robot/right/tcp_pose": "action_tcp",
        "gripper/right/width": "action_ee_command"
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
    }
)


