"""
Example usage of the logger module.

This file demonstrates various ways to use the logger.
"""
import time


def example_basic_usage() -> None:
    """Example 1: Basic logger usage with default configuration."""
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    from logger import logger
    
    logger.trace("This is a TRACE message (very detailed)")
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.success("This is a SUCCESS message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")


def example_custom_config() -> None:
    """Example 2: Using custom configuration."""
    print("\n" + "="*60)
    print("Example 2: Custom Configuration")
    print("="*60)
    
    from logger import setup_logger, LoggerConfig, logger
    
    # Create custom config
    config = LoggerConfig(
        level="DEBUG",
        save_to_file=True,
        log_filename="example_custom_{time:YYYY-MM-DD_HH-mm-ss}.log",
        rotation="10 MB",
        retention="1 day"
    )
    
    # Apply configuration
    setup_logger(config)
    
    logger.debug("This DEBUG message will now be visible")
    logger.info("Log file is being saved to logger/logs/")


def example_training_config() -> None:
    """Example 3: Training configuration."""
    print("\n" + "="*60)
    print("Example 3: Training Configuration")
    print("="*60)
    
    from logger import setup_logger, get_training_config, logger
    
    # Setup for training
    setup_logger(get_training_config(experiment_name="cable_v5_test"))
    
    logger.info("Training started")
    logger.info("Epoch 1/10")
    
    for step in range(3):
        logger.debug(f"Step {step}: Processing batch")
        time.sleep(0.1)
    
    logger.success("Epoch 1 completed, loss: 0.1234")


def example_deployment_config() -> None:
    """Example 4: Deployment configuration."""
    print("\n" + "="*60)
    print("Example 4: Deployment Configuration")
    print("="*60)
    
    from logger import setup_logger, get_deployment_config, logger
    
    # Setup for deployment
    setup_logger(get_deployment_config(robot_name="Rizon4"))
    
    logger.info("Robot initializing...")
    time.sleep(0.2)
    logger.success("Robot initialized successfully")
    logger.info("Running inference...")


def example_exception_handling() -> None:
    """Example 5: Exception handling with logger."""
    print("\n" + "="*60)
    print("Example 5: Exception Handling")
    print("="*60)
    
    from logger import logger
    
    try:
        # Simulate an error
        result = 1 / 0
    except ZeroDivisionError as e:
        logger.error(f"An error occurred: {e}")
        logger.exception("Full traceback:")


def example_format_types() -> None:
    """Example 6: Different format types."""
    print("\n" + "="*60)
    print("Example 6: Format Types")
    print("="*60)
    
    from logger import setup_logger, get_default_config, logger
    
    # Default format
    print("\n--- Default Format ---")
    setup_logger(get_default_config(format_type="default"))
    logger.info("This uses the default format")
    
    # Simple format
    print("\n--- Simple Format ---")
    setup_logger(get_default_config(format_type="simple"))
    logger.info("This uses the simple format")
    
    # Detailed format
    print("\n--- Detailed Format ---")
    setup_logger(get_default_config(format_type="detailed"))
    logger.info("This uses the detailed format")


def example_debug_config() -> None:
    """Example 7: Debug configuration."""
    print("\n" + "="*60)
    print("Example 7: Debug Configuration")
    print("="*60)
    
    from logger import setup_logger, get_debug_config, logger
    
    # Setup for debugging
    setup_logger(get_debug_config())
    
    logger.trace("Very detailed trace information")
    logger.debug("Debug information")
    logger.info("Normal information")


def example_contextual_logging() -> None:
    """Example 8: Contextual logging with extra information."""
    print("\n" + "="*60)
    print("Example 8: Contextual Logging")
    print("="*60)
    
    from logger import logger
    
    # Bind context to logger
    context_logger = logger.bind(user="shirun", task="training")
    
    context_logger.info("Starting task")
    context_logger.success("Task completed")


def main() -> None:
    """Run all examples."""
    print("\n" + "="*60)
    print("ForcePolicy Logger Examples")
    print("="*60)
    
    # Run examples
    example_basic_usage()
    example_custom_config()
    example_training_config()
    example_deployment_config()
    example_exception_handling()
    example_format_types()
    example_debug_config()
    example_contextual_logging()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("Check logger/logs/ for saved log files")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

