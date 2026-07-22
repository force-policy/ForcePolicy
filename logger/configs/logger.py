"""
Logger Configuration.

This module provides a centralized logging configuration using loguru.
It supports different log levels, formatters, and file output options.
"""
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass, field
from pathlib import Path
import sys


@dataclass
class LoggerConfig:
    """
    Configuration for loguru logger.
    
    Attributes:
        level: Minimum log level to display (DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
        format: Log message format string (None to use default)
        colorize: Whether to colorize console output
        save_to_file: Whether to save logs to file
        log_dir: Directory to save log files (relative to project root)
        log_filename: Log file name template (supports time formatting)
        rotation: When to rotate log files (e.g., "500 MB", "1 day", "00:00")
        retention: How long to keep old log files (e.g., "10 days", "1 week")
        compression: Compression format for rotated logs (e.g., "zip", "gz")
        enqueue: Whether to make logging thread-safe
        backtrace: Whether to enable exception backtrace
        diagnose: Whether to enable diagnostic information in exceptions
    """
    level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Optional[str] = None
    colorize: bool = True
    save_to_file: bool = False
    log_dir: str = "logger/logs"
    log_filename: str = "{time:YYYY-MM-DD_HH-mm-ss}.log"
    rotation: Optional[str] = "500 MB"
    retention: Optional[str] = "10 days"
    compression: Optional[str] = "zip"
    enqueue: bool = True
    backtrace: bool = True
    diagnose: bool = True
    
    # Additional handlers
    extra_handlers: Dict[str, Any] = field(default_factory=dict)


# Default format string (used when format is None)
DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# Simplified format for production
SIMPLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

# Detailed format with process/thread info
DETAILED_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{process}</cyan>:<cyan>{thread}</cyan> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


def setup_logger(config: LoggerConfig) -> None:
    """
    Setup global logger with the given configuration.
    
    Args:
        config: LoggerConfig instance with logging settings
    """
    from loguru import logger
    
    # Remove default handler
    logger.remove()
    
    # Determine format
    log_format = config.format if config.format is not None else DEFAULT_FORMAT
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=config.level,
        colorize=config.colorize,
        backtrace=config.backtrace,
        diagnose=config.diagnose,
        enqueue=config.enqueue
    )
    
    # Add file handler if enabled
    if config.save_to_file:
        # Get project root directory
        project_root = Path(__file__).parent.parent.parent
        log_path = project_root / config.log_dir
        log_path.mkdir(parents=True, exist_ok=True)
        
        log_file = log_path / config.log_filename
        
        logger.add(
            str(log_file),
            format=log_format,
            level=config.level,
            rotation=config.rotation,
            retention=config.retention,
            compression=config.compression,
            backtrace=config.backtrace,
            diagnose=config.diagnose,
            enqueue=config.enqueue
        )
        
        logger.info(f"Log file created at: {log_file}")
    
    # Add extra handlers
    for handler_name, handler_config in config.extra_handlers.items():
        logger.add(**handler_config)
        logger.debug(f"Extra handler '{handler_name}' added")
    
    logger.info(f"Logger initialized with level: {config.level}")


def get_default_config(
    level: str = "INFO",
    save_to_file: bool = False,
    format_type: Literal["default", "simple", "detailed"] = "default"
) -> LoggerConfig:
    """
    Get a default logger configuration.
    
    Args:
        level: Log level
        save_to_file: Whether to save logs to file
        format_type: Format type to use
    
    Returns:
        LoggerConfig instance
    """
    format_map = {
        "default": DEFAULT_FORMAT,
        "simple": SIMPLE_FORMAT,
        "detailed": DETAILED_FORMAT
    }
    
    return LoggerConfig(
        level=level,
        format=format_map.get(format_type, DEFAULT_FORMAT),
        save_to_file=save_to_file
    )


def get_training_config(experiment_name: Optional[str] = None) -> LoggerConfig:
    """
    Get a logger configuration suitable for training.
    
    Args:
        experiment_name: Optional experiment name to include in log filename
    
    Returns:
        LoggerConfig instance for training
    """
    log_filename = f"train_{experiment_name}_{{time:YYYY-MM-DD_HH-mm-ss}}.log" if experiment_name else "train_{time:YYYY-MM-DD_HH-mm-ss}.log"
    
    return LoggerConfig(
        level="INFO",
        format=DEFAULT_FORMAT,
        save_to_file=True,
        log_filename=log_filename,
        rotation="1 GB",
        retention="30 days",
        compression="zip"
    )


def get_deployment_config(robot_name: Optional[str] = None) -> LoggerConfig:
    """
    Get a logger configuration suitable for deployment/inference.
    
    Args:
        robot_name: Optional robot name to include in log filename
    
    Returns:
        LoggerConfig instance for deployment
    """
    log_filename = f"deploy_{robot_name}_{{time:YYYY-MM-DD_HH-mm-ss}}.log" if robot_name else "deploy_{time:YYYY-MM-DD_HH-mm-ss}.log"
    
    return LoggerConfig(
        level="INFO",
        format=SIMPLE_FORMAT,
        save_to_file=True,
        log_filename=log_filename,
        rotation="100 MB",
        retention="7 days",
        compression="gz"
    )


def get_debug_config() -> LoggerConfig:
    """
    Get a logger configuration suitable for debugging.
    
    Returns:
        LoggerConfig instance for debugging
    """
    return LoggerConfig(
        level="DEBUG",
        format=DETAILED_FORMAT,
        save_to_file=True,
        log_filename="debug_{time:YYYY-MM-DD_HH-mm-ss}.log",
        rotation="200 MB",
        retention="3 days"
    )

