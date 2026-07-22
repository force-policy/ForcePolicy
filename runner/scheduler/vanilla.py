"""
Vanilla scheduler.
"""
import os
import time
import torch
import threading

from logger import logger

from runner.configs.scheduler import VanillaSchedulerConfig

from runner.scheduler.base import SchedulerBase
from runner.agent.real_agent import RealAgent
from runner.configs.agent import AgentObsKeysConfig
from runner.utils.ensemble_buffer import EnsembleBuffer


class VanillaScheduler(SchedulerBase):
    """
    Vanilla scheduler implemented for synchronous inference and control.
    """
    def __init__(
        self,
        config: VanillaSchedulerConfig,
        agent: RealAgent,
        agent_obs_keys: AgentObsKeysConfig,
        device: torch.device
    ) -> None:
        """ Initialization. """
        super(VanillaScheduler, self).__init__(
            config = config,
            agent = agent,
            agent_obs_keys = agent_obs_keys,
            device = device
        )
        self._stop_event = threading.Event()
        self.dt = 1.0 / config.frequency
        self.ensemble_buffer = EnsembleBuffer(self.config.ensemble_buffer_config)
    

    def eval_thread(self) -> None:
        """ Evaluation thread. """
        step = 0
        while not self._stop_event.is_set():
            if step >= self.config.max_steps:
                logger.info("Reach maximum steps, stopped.")
                break

            tic = time.time()

            if step % self.config.num_inference_step == 0:
                action_dict, _ = self.model_inference()
                self.visualize()
                self.ensemble_buffer.add_action(action_dict, step)
            
            step_action = self.ensemble_buffer.get_action()
            self.agent.action(step_action, self.config.action_params)

            step += 1
            elapsed_time = time.time() - tic
            if elapsed_time < self.dt:
                time.sleep(self.dt - elapsed_time)

    
    def start(self) -> None:
        """ Start. """
        self._stop_event.clear()
        self._eval_thread = threading.Thread(
            target = self.eval_thread,
            name = "eval_thread",
            daemon = True
        )
        self._eval_thread.start()


    def stop(self) -> None:
        """ Stop. """
        self._stop_event.set()
