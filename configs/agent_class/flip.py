import time
import torch
import numpy as np

from runner.agent.real_agent import RealAgent
from runner.configs.agent.real_agent import RealAgentConfig


class Agent(RealAgent):
    def __init__(self, config: RealAgentConfig) -> None:
        super(Agent, self).__init__(config)
    
    def ready(self):    
        """Initialize robot to ready state."""

        # Close the gripper.
        self.config.grippers["right"].gripper.Move(0.0, 0.1, 50)

        # Ready joint pose. The values below are in degrees, while send_joint_pos expects radians.
        tar_pose_q_deg = [46.51, -60.72, -111.44, 129.64, 97.36, 24.09, 69.10]  # flip ready pose
        tar_pose_q_rad = np.deg2rad(tar_pose_q_deg).tolist()

        self.config.robots["right"].switch_mode("IDLE")
        self.config.robots["right"].cali_sensor()
        self.config.robots["right"].send_joint_pos(tar_pose_q_rad, max_vel = [0.3]*7, max_acc = [0.5]*7, impedance = True, blocking = True)
        time.sleep(3.0)
        self.config.robots["right"].cali_sensor()

        self.config.robots["right"].switch_mode("NRT_SUPER_PRIMITIVE")
        self.config.robots["right"].robot.SetForceControlAxis([0, 0, 0, 0, 0, 0])
        self.config.robots["right"].robot.SetCartesianImpedance([5000, 5000, 5000, 300, 300, 300], [0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
        self.config.robots["right"].robot.SetMaxContactWrench([50, 50, 50, 30, 30, 30])
