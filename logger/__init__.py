"""
Global Logger Module for ForcePolicy.

This module provides a centralized logging interface using loguru.

Usage:
    # Basic usage with default config
    from logger import logger
    logger.info("This is an info message")
    
    # Setup with custom config
    from logger import setup_logger, LoggerConfig
    config = LoggerConfig(level="DEBUG", save_to_file=True)
    setup_logger(config)
    
    # Use predefined configs
    from logger import setup_logger, get_training_config
    setup_logger(get_training_config(experiment_name="cable_v5"))
    
    # Then use logger anywhere in the project
    from logger import logger
    logger.info("Training started")
"""
from loguru import logger

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

# Setup default logger (console only, INFO level)
# Users can reconfigure by calling setup_logger() with custom config
_default_config = LoggerConfig(
    level="INFO",
    format=DEFAULT_FORMAT,
    colorize=True,
    save_to_file=False
)
setup_logger(_default_config)

__all__ = [
    'logger',
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

