"""
LowdimRecorder and make_lowdim_observation
------------------------------------------
Unified architecture for robot low-dimensional history data.

LowdimRecorder:
  - Background thread that records data at max_record_frequency
  - One recorder per unique (device_ref, device_func)
  - Maintains a shared rolling buffer

make_lowdim_observation:
  - Factory function that returns a callable
  - Captures freq, length, time_reversed in closure
  - Multiple observations can share one recorder

Typical usage:
    - tcp_pose: device_func="get_tcp_pose", returns (L, 7) [x,y,z,qw,qx,qy,qz]
    - force_torque: device_func="get_force_torque_tcp", returns (L, 6) [fx,fy,fz,mx,my,mz]
    - tcp_vel: device_func="get_tcp_vel", returns (L, 6) [vx,vy,vz,wx,wy,wz]
"""

import time
import threading
from collections import deque
from typing import Optional, Dict, Any

import numpy as np

from runner.configs.lowdim_provider import LowdimRecorderConfig, LowdimObservationConfig


# Mapping from device_func to expected data dimension
DEVICE_FUNC_DIM = {
    "get_tcp_pose": 7,
    "get_force_torque_tcp": 6,
    "get_tcp_vel": 6,
    "get_joint_pos": None,  # Variable, depends on robot
    "get_joint_vel": None,
}


class LowdimRecorder:
    """
    Background recorder for a single data type.
    
    Records data at max_record_frequency and maintains a rolling buffer.
    Multiple LowdimObservation instances can share one recorder.
    """

    def __init__(
        self,
        config: LowdimRecorderConfig,
        device: Any,
    ):
        """
        Initialize LowdimRecorder.

        Args:
            config: LowdimRecorderConfig with device info and recording parameters
            device: The actual device instance (robot/gripper/hand)
        """
        self.config = config
        self.device = device

        # Get the device function
        if not hasattr(self.device, config.device_func):
            raise ValueError(
                f"Device has no method '{config.device_func}'"
            )
        self.device_func = getattr(self.device, config.device_func)
        self.device_func_kwargs = config.device_func_kwargs

        # Determine data dimension
        self._data_dim = DEVICE_FUNC_DIM.get(config.device_func, None)
        if self._data_dim is None:
            # Try to infer from first call
            try:
                sample = self.device_func(**self.device_func_kwargs)
                self._data_dim = len(np.asarray(sample).flatten())
            except Exception:
                self._data_dim = 6  # Default fallback

        # Recording parameters
        self.max_record_frequency = config.max_record_frequency
        self.max_window_time = config.max_window_time
        self._dt = 1.0 / max(self.max_record_frequency, 1e-6)

        max_len = int(self.max_record_frequency * self.max_window_time)
        max_len = max(1, max_len)

        # Shared data buffer: (timestamp, data)
        self._buffer: deque = deque(maxlen=max_len)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Start background recording thread
        self._thread = threading.Thread(
            target=self._loop,
            name=f"LowdimRecorder-{config.device_ref}-{config.device_func}",
            daemon=True,
        )
        self._thread.start()

    @property
    def data_dim(self) -> int:
        """Return the data dimension."""
        return self._data_dim

    def _loop(self):
        """Background sampling loop"""
        while not self._stop.is_set():
            tic = time.time()
            try:
                data = self.device_func(**self.device_func_kwargs)
                data = np.asarray(data, dtype=np.float32).flatten()
                with self._lock:
                    self._buffer.append((tic, data))

            except Exception as e:
                print(f"[LowdimRecorder] Error in sampling loop: {e}")

            elapsed = time.time() - tic
            if elapsed < self._dt:
                time.sleep(self._dt - elapsed)

    def stop(self):
        """Stop background sampling thread."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_latest(self) -> np.ndarray:
        """Get the most recent data sample."""
        with self._lock:
            if len(self._buffer) == 0:
                return np.zeros(self._data_dim, dtype=np.float32)
            return self._buffer[-1][1].copy()

    def get_window(
        self,
        freq: float,
        length: int,
        is_reverse: bool = False,
        remove_first: bool = False,
    ) -> np.ndarray:
        """
        Return a data window of shape (length, dim), resampled to the requested
        temporal grid using nearest-neighbor sampling.

        Args:
            freq: Desired window frequency (Hz).
            length: Desired number of time steps.
            is_reverse: If True, reverse time order (index 0 = most recent).
            remove_first: If True, drop the oldest sample.

        Returns:
            np.ndarray of shape (length, data_dim)
        """
        L = int(length)
        if L <= 0:
            return np.zeros((0, self._data_dim), dtype=np.float32)

        need = L + (1 if remove_first else 0)
        dt = 1.0 / max(float(freq), 1e-6)

        with self._lock:
            if len(self._buffer) == 0:
                window = np.zeros((L, self._data_dim), dtype=np.float32)
                if is_reverse:
                    window = window[::-1]
                return window

            times = np.array([t for (t, _) in self._buffer], dtype=np.float64)
            data = np.stack([d for (_, d) in self._buffer], axis=0)

        t_end = times[-1]
        t_start = t_end - dt * (need - 1)
        target_ts = np.linspace(t_start, t_end, need, dtype=np.float64)

        # Nearest-neighbor in time
        idx = np.searchsorted(times, target_ts, side="left")
        idx = np.clip(idx, 0, len(times) - 1)

        sel_data = data[idx]

        if remove_first and sel_data.shape[0] > 0:
            sel_data = sel_data[1:]

        # If not enough samples (very early phase), left-pad with earliest data
        if sel_data.shape[0] < L:
            pad_num = L - sel_data.shape[0]
            if sel_data.shape[0] > 0:
                pad_block = np.repeat(sel_data[0:1, :], pad_num, axis=0)
            else:
                pad_block = np.zeros((pad_num, self._data_dim), dtype=np.float32)
            sel_data = np.concatenate([pad_block, sel_data], axis=0)

        if is_reverse:
            sel_data = sel_data[::-1]

        return sel_data.astype(np.float32)

    # ========== Compatibility methods for AdaptiveScheduler ==========

    def get_tcp_pose_window(
        self,
        freq: float,
        length: int,
        coordinate: str = "base",
        projector=None,
        remove_first: bool = False,
        is_reverse: bool = False,
    ) -> np.ndarray:
        """
        Return a TCP pose window of shape (length, 7).
        Compatibility method for AdaptiveScheduler.
        """
        if coordinate != "base" and projector is not None:
            print(f"[Warning] Coordinate projection to '{coordinate}' not yet implemented")

        return self.get_window(
            freq=freq,
            length=length,
            is_reverse=is_reverse,
            remove_first=remove_first,
        )

    def get_latest_tcp_pose(self) -> np.ndarray:
        """
        Get the most recent TCP pose.
        Compatibility method for AdaptiveScheduler.
        """
        return self.get_latest()


def get_lowdim_observation(
    config: LowdimObservationConfig,
    recorder: LowdimRecorder,
):
    """
    Factory function that returns a callable for getting windowed data.
    
    The returned function samples from the shared recorder using config parameters.
    
    Args:
        config: LowdimObservationConfig with sampling parameters
        recorder: Shared LowdimRecorder instance
        
    Returns:
        Callable that returns np.ndarray of shape (config.length, data_dim)
    """
    freq = config.freq
    length = config.length
    time_reversed = config.time_reversed
    remove_first = config.remove_first
    
    def get_data() -> np.ndarray:
        return recorder.get_window(
            freq=freq,
            length=length,
            is_reverse=time_reversed,
            remove_first=remove_first,
        )
    
    return get_data
