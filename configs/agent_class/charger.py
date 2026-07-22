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
        self.config.robots["main"].switch_mode("IDLE") 
        self.config.robots["main"].robot.SetPassiveForceControl(False)
        self.config.robots["main"].cali_sensor()
        init_q_rad = [1.376084804534912, -1.5889960527420044, -2.302342653274536, 1.6863548755645752, 1.494018793106079, -0.7301536202430725, -0.33375540375709534]
        self.config.robots["main"].send_joint_pos(init_q_rad, max_vel = [0.3]*7, max_acc = [0.5]*7, impedance = True,blocking = True)
        time.sleep(5)
        self.config.robots["main"].cali_sensor()
        self.config.robots["main"].switch_mode("NRT_SUPER_PRIMITIVE") 
        self.config.robots["main"].reset_cartesian_impedance()
        self.config.robots["main"].set_cartesian_impedance([10000, 10000, 10000, 20, 20, 20], [0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
        self.config.robots["main"].robot.SetMaxContactWrench([50, 10, 10, 1.0, 1.0, 1.0])
        # self.config.robots["main"].robot.SetNullSpacePosture( [1.0057107210159302, -1.4939353466033936, -2.243243455886841, 1.1804132461547852, 1.795068383216858, -0.6451272964477539, -0.12059247493743896])
        

