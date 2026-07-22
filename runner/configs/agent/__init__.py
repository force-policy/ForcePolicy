from runner.configs.agent.base import BaseAgentConfig
from runner.configs.agent.base import PlatformConfig
from runner.configs.agent.base import AgentObsKeysConfig
from runner.configs.agent.local_agent import LocalAgentConfig

try:
    from runner.configs.agent.real_agent import RealAgentConfig
except ImportError:
    print("Library 'easyrobot' not found, RealAgentConfig is not available.")
    RealAgentConfig = None
