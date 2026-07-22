# Logger Module

Global logging system for ForcePolicy using [loguru](https://github.com/Delgan/loguru).

## Directory Structure

```
logger/
├── __init__.py              # Main logger interface
├── configs/
│   ├── __init__.py
│   └── logger.py            # Logger configuration classes
├── logs/                    # Log files directory (git-ignored)
└── README.md
```

## Quick Start

### Basic Usage

```python
# Import logger directly
from logger import logger

logger.debug("Debug message")
logger.info("Info message")
logger.success("Success message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
```

### Custom Configuration

```python
from logger import setup_logger, LoggerConfig

# Create custom config
config = LoggerConfig(
    level="DEBUG",
    save_to_file=True,
    log_filename="my_experiment_{time:YYYY-MM-DD}.log"
)

# Apply configuration
setup_logger(config)

# Now use logger as usual
from logger import logger
logger.info("This will be logged with custom config")
```

### Predefined Configurations

#### Training Configuration

```python
from logger import setup_logger, get_training_config

# Setup for training with experiment name
setup_logger(get_training_config(experiment_name="cable_v5"))

from logger import logger
logger.info("Training started")
```

#### Deployment Configuration

```python
from logger import setup_logger, get_deployment_config

# Setup for deployment with robot name
setup_logger(get_deployment_config(robot_name="Rizon4-062703"))

from logger import logger
logger.info("Robot initialized")
```

#### Debug Configuration

```python
from logger import setup_logger, get_debug_config

# Setup for debugging
setup_logger(get_debug_config())

from logger import logger
logger.debug("Detailed debug information")
```

## Configuration Options

### LoggerConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | str | `"INFO"` | Minimum log level (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL) |
| `format` | str | `None` | Custom format string (None = use DEFAULT_FORMAT) |
| `colorize` | bool | `True` | Enable colored output in console |
| `save_to_file` | bool | `False` | Save logs to file |
| `log_dir` | str | `"logger/logs"` | Directory for log files |
| `log_filename` | str | `"{time:YYYY-MM-DD_HH-mm-ss}.log"` | Log filename template |
| `rotation` | str | `"500 MB"` | When to rotate files (size/time) |
| `retention` | str | `"10 days"` | How long to keep old logs |
| `compression` | str | `"zip"` | Compression format for rotated logs |
| `enqueue` | bool | `True` | Thread-safe logging |
| `backtrace` | bool | `True` | Enable exception backtrace |
| `diagnose` | bool | `True` | Enable diagnostic info |

### Format Strings

Three predefined format strings are available:

1. **DEFAULT_FORMAT**: Standard format with time, level, location, and message
2. **SIMPLE_FORMAT**: Minimal format for production
3. **DETAILED_FORMAT**: Verbose format with process/thread info

## Examples

### Example 1: Training Script

```python
from logger import setup_logger, get_training_config, logger

# Setup logger at the start of training
setup_logger(get_training_config(experiment_name="cable_v5_rise2"))

logger.info("Starting training...")
logger.info(f"Config: {config}")

for epoch in range(num_epochs):
    logger.info(f"Epoch {epoch}/{num_epochs}")
    # Training code...
    logger.success(f"Epoch {epoch} completed, loss: {loss:.4f}")
```

### Example 2: Deployment Script

```python
from logger import setup_logger, get_deployment_config, logger

# Setup logger for robot deployment
setup_logger(get_deployment_config(robot_name="Rizon4"))

logger.info("Initializing robot...")
logger.success("Robot initialized successfully")

try:
    # Deployment code...
    logger.info("Running inference...")
except Exception as e:
    logger.error(f"Deployment failed: {e}")
    logger.exception("Full traceback:")
```

### Example 3: Custom Multi-Handler Setup

```python
from logger import setup_logger, LoggerConfig, logger

config = LoggerConfig(
    level="DEBUG",
    save_to_file=True,
    extra_handlers={
        "error_file": {
            "sink": "logger/logs/errors.log",
            "level": "ERROR",
            "rotation": "100 MB"
        }
    }
)

setup_logger(config)
logger.info("This goes to console and main log")
logger.error("This goes to console, main log, AND errors.log")
```

## Log Levels

- **TRACE**: Very detailed information, typically for diagnosing problems
- **DEBUG**: Detailed information for debugging
- **INFO**: General informational messages
- **SUCCESS**: Success messages (loguru-specific)
- **WARNING**: Warning messages for potentially harmful situations
- **ERROR**: Error messages
- **CRITICAL**: Very severe error messages

## File Rotation

Logs can be automatically rotated based on:
- **Size**: `"500 MB"`, `"1 GB"`
- **Time**: `"1 day"`, `"1 week"`, `"00:00"` (daily at midnight)
- **Both**: `"500 MB"` or `"1 week"`

Old logs can be automatically compressed and cleaned up based on the `compression` and `retention` settings.

## Integration with Existing Code

To integrate with existing code using `print()`:

```python
# Before
print("Training started")

# After
from logger import logger
logger.info("Training started")
```

## Best Practices

1. **Setup Once**: Call `setup_logger()` once at the start of your script
2. **Import Logger**: Import `logger` in each module that needs logging
3. **Use Appropriate Levels**: Use DEBUG for development, INFO for production
4. **Enable File Logging**: Always enable `save_to_file=True` for training/deployment
5. **Name Your Experiments**: Use descriptive names in `get_training_config()`
6. **Handle Exceptions**: Use `logger.exception()` in exception handlers

## Notes

- Log files are stored in `logger/logs/` by default
- The `logs/` directory is git-ignored
- Logs are automatically rotated and compressed to save space
- Thread-safe logging is enabled by default (`enqueue=True`)
- All timestamps are in local time

