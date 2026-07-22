from runner.agent.base import BaseAgent
from runner.agent.local_agent import LocalAgent

try:
    from runner.agent.real_agent import RealAgent
except ImportError:
    print("Library 'easyrobot' not found, RealAgent is not available.")
    RealAgent = None

