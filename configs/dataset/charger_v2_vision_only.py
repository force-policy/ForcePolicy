from data_infra.configs.dataset import *


dataset_config = DatasetConfig(
    type = "vision",
    data_path = "data/charger_v2",
    robot_poses = {
        "main": np.eye(4, dtype = np.float32),
    },
    obs = ObservationConfig(
        vision = VisionConfig(
            main_cameras = ["035622060973"],
            aux_cameras = {},
            colors = ["main"],
            depths = ["main"],
            extrinsics = ["main"],
            intrinsics = ["main"]
        ),
        lowdim = {}
    ),
    action = ActionConfig(
        lowdim = {
            "tcp_pose": LowdimConfig(
                lowdim_name = "lowdim",
                field = "tcp_pose_063047",
                length = 50,
                freq = 10,
                data_freq = 1000,
                direction = 1,
                remove_first = False
            )
        }
    ),
    repeat_dataset = 10
)
