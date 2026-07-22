"""
Lowdim Recorder and Observation Configuration

Architecture:
- LowdimRecorder: Background thread that records data at max_record_frequency
  - One recorder per unique (device_type, device_name, device_func)
  - Shared buffer for all observations of the same data type

- LowdimObservation: Sampling configuration for a specific use case
  - References a recorder by recorder_key
  - Specifies freq, length, time_reversed for sampling
"""
from typing import Any, Dict, Callable
from dataclasses import dataclass, field


@dataclass(kw_only=True)
class LowdimRecorderConfig:
    """
    Configuration for a LowdimRecorder (background recording thread).
    
    One recorder is created per unique (device_ref, device_func).
    Multiple LowdimObservations can share the same recorder.
    
    Attributes:
        device_ref: Reference to device in agent config, format: "type/name"
                    e.g., "robot/main" → agent.config.robots["main"]
        device_func: Method name to call on device (e.g., "get_tcp_pose", "get_force_torque_tcp")
        device_func_kwargs: Additional kwargs for the device function
        max_record_frequency: Recording frequency in Hz (default 200.0)
        max_window_time: Maximum buffer time in seconds (default 10.0)
    """
    device_ref: str  # "robot/main", "gripper/main", etc.
    device_func: str
    device_func_kwargs: Dict[str, Any] = field(default_factory=dict)
    max_record_frequency: float = 200.0
    max_window_time: float = 10.0


@dataclass(kw_only=True)
class LowdimObservationConfig:
    """
    Configuration for a LowdimObservation (sampling from a shared recorder).
    
    Attributes:
        recorder_key: Key to identify which recorder to use (must match a key in lowdim_recorder_configs)
        freq: Output sampling frequency (Hz)
        length: Output window length (number of frames)
        time_reversed: If True, reverse time order (newest first);
                       If False, normal order (newest last, corresponds to direction=-1 in dataset)
        remove_first: If True, drop the oldest sample in the window (default False)
    """
    recorder_key: str
    freq: int
    length: int
    time_reversed: bool = False
    remove_first: bool = False