"""
Logger configuration module.
"""
from logger.configs.logger import (
    LoggerConfig,
    setup_logger,
    get_default_config,
    get_training_config,
    get_deployment_config,
    get_debug_config,
    DEFAULT_FORMAT,
    SIMPLE_FORMAT,
    DETAILED_FORMAT
)

__all__ = [
    'LoggerConfig',
    'setup_logger',
    'get_default_config',
    'get_training_config',
    'get_deployment_config',
    'get_debug_config',
    'DEFAULT_FORMAT',
    'SIMPLE_FORMAT',
    'DETAILED_FORMAT'
]

