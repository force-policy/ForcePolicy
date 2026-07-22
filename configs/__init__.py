import sys
import importlib


def parse_config_name(base_module, config_name):
    return importlib.import_module(f"{base_module.__name__}.{config_name}")


def get_config(config_name, config_type):
    """ Get config from config name. """
    return getattr(
        parse_config_name(sys.modules[__name__], config_name = f"{config_type}.{config_name}"),
        f"{config_type}_config"
    )

def get_agent(agent_config_name, agent_class_name):
    """ Get agent from config name. """
    agent_config = getattr(parse_config_name(sys.modules[__name__], config_name = f"agent.{agent_config_name}.agent"), "agent_config")
    agent_class = getattr(parse_config_name(sys.modules[__name__], config_name = f"agent_class.{agent_class_name}"), "Agent")
    return agent_class(agent_config)

def get_agent_obs_keys(agent_config_name, obs_key_config_name):
    """ Get agent obs key config from config name. """
    return getattr(parse_config_name(sys.modules[__name__], config_name = f"agent.{agent_config_name}.obs_key"), "obs_key_configs")[obs_key_config_name]
