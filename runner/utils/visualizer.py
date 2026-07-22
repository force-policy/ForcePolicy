"""
2D Visualizer for robot control and waypoint visualization.
"""
import os
import cv2
import time
import threading
import numpy as np
from typing import Any, Dict, Optional, Tuple

from logger import logger

from easyrobot.utils.shared_memory import SharedMemoryManager
from utils.transforms.pose import xyz_rot_to_mat
from utils.transforms.projection import apply_mat_to_pose, apply_mat_to_point
from utils.transforms.rotation import RotationType


class Visualizer:
    """
    2D visualizer for robot TCP pose, force/torque, and predicted waypoints.
    
    Features:
    - Display current TCP pose with axis
    - Display force/torque vectors
    - Display predicted waypoints with different styles (raw/interp/head)
    - Display predicted force/torque from model
    """
    
    def __init__(
        self,
        image_shm: SharedMemoryManager,
        intrinsic: np.ndarray,
        T_world_camera: np.ndarray,
        freq: int = 20,
        axis_length: float = 0.05,
        force_scale: float = 0.004,
        record_path: Optional[str] = None,
        record_freq: Optional[int] = None,
    ):
        """
        Args:
            image_shm: Shared memory manager for camera images
            intrinsic: Camera intrinsic matrix (3x3)
            T_world_camera: 4x4 transformation matrix from camera to world
            freq: Visualization frequency in Hz
            axis_length: Length of TCP axis in meters
            force_scale: Scale factor for force vector visualization
            record_path: Optional path to save visualizer frames (e.g., scene_0001/visualizer)
            record_freq: Optional recording frequency in Hz (defaults to freq if not specified)
        """
        self.image_shm = image_shm
        self.intrinsic = intrinsic
        self.T_camera_world = np.linalg.inv(T_world_camera)  # World to camera
        self.freq = freq
        self.dt = 1.0 / freq
        
        self.axis_length = axis_length
        self.force_scale = force_scale
        
        # Recording settings
        self.record_path = record_path
        self.record_freq = record_freq if record_freq is not None else freq
        self.record_dt = 1.0 / self.record_freq
        self._recorder_thread = None
        
        # Create recording directory if needed
        if self.record_path is not None:
            os.makedirs(self.record_path, exist_ok=True)
            logger.info("[Visualizer] Recording enabled at {} with freq={} Hz", 
                       self.record_path, self.record_freq)
        
        # Registered agents and schedulers
        # Registered agents and schedulers
        # name -> agent (we extract poses and robots from agent itself)
        self.agents: Dict[str, Any] = {}
        self.schedulers: Dict[str, Any] = {}
        
        # Thread control
        self._stop_event = threading.Event()
        self._stop_event.clear()
        self._viewer_thread: Optional[threading.Thread] = None
        
        # Display state
        self.displayed_current_colors = None
        
        # Color palette
        self.axis_colors = [(30, 33, 217), (70, 161, 34), (0, 125, 206)]  # BGR: Red, Green, Blue
        self.interaction_frame_colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255)]  # BGR: Yellow, Cyan, Magenta
        self.force_color = (0, 165, 255)  # BGR: Orange
        self.force_predicted_color = (255, 200, 100)  # BGR: Light blue
        
        # Whether to consider translation when drawing interaction frame
        # If False, interaction frame is drawn at TCP position (only rotation is considered)
        self.if_translation = False
    
    def register_agent(
        self, 
        name: str, 
        agent
    ):
        """
        Register an agent for visualization.
        
        Args:
            name: Unique name for this agent
            agent: Agent instance with get_tcp_pose() and specific robot config
        """
        if name not in self.agents:
            self.agents[name] = agent

    def _get_tcp_pose(self, agent, robot_name: str) -> np.ndarray:
        """Get TCP pose in base frame, preferring RealAgent style APIs."""
        # Preferred: RealAgent.robot(robot_name).get_tcp_pose()
        if hasattr(agent, "robot"):
            r = agent.robot(robot_name)
            return np.asarray(r.get_tcp_pose(), dtype=np.float32)
        # Fallback: older Agent interface
        if hasattr(agent, "get_tcp_pose"):
            return np.asarray(agent.get_tcp_pose(robot_name), dtype=np.float32)

        raise AttributeError("agent has no supported tcp pose getter")
    
    def unregister_agent(self, name: str):
        """Unregister an agent."""
        if name in self.agents:
            del self.agents[name]
    
    def register_scheduler(
        self, 
        name: str, 
        scheduler
    ):
        """
        Register a scheduler for visualization.
        
        Args:
            name: Unique name for this scheduler
            scheduler: Scheduler instance with get_queue_tcps() and get_queue_wrenches() methods
        """
        if name not in self.schedulers:
            self.schedulers[name] = scheduler
    
    def unregister_scheduler(self, name: str):
        """Unregister a scheduler."""
        if name in self.schedulers:
            del self.schedulers[name]

    @staticmethod
    def _describe_obj(x: Any, max_repr: int = 240) -> str:
        """Return a short, safe-to-log description (used for error logs only)."""
        try:
            if x is None:
                return "None"

            if isinstance(x, np.ndarray):
                return f"ndarray(shape={x.shape}, dtype={x.dtype}, ndim={x.ndim})"

            if isinstance(x, dict):
                keys = list(x.keys())
                return f"dict(len={len(x)}, keys={keys[:20]})"
            if isinstance(x, (list, tuple)):
                return f"{type(x).__name__}(len={len(x)})"

            r = repr(x)
            if len(r) > max_repr:
                r = r[: max_repr - 3] + "..."
            return f"{type(x).__name__}({r})"
        except Exception as e:
            return f"<unprintable {type(x).__name__}: {e}>"

    @staticmethod
    def _extract_color_image(payload: Any) -> Optional[np.ndarray]:
        """
        Shared memory may return either:
        - an RGB/RGBA numpy array
        - a dict like {"rgb": np.ndarray, "depth": ...}
        This function extracts the color image part robustly.
        """
        if payload is None:
            return None

        if isinstance(payload, np.ndarray):
            return payload

        if isinstance(payload, dict):
            # Common keys used by camera pipelines
            for k in ("rgb", "color", "image", "frame"):
                if k in payload:
                    v = payload[k]
                    # ensure ndarray if possible
                    if isinstance(v, np.ndarray):
                        return v
                    try:
                        return np.asarray(v)
                    except Exception:
                        return None
            return None

        # Fallback: try numpy conversion
        try:
            return np.asarray(payload)
        except Exception:
            return None
    
    def run(self):
        """Start the visualization and recording threads."""
        if self._viewer_thread is None or not self._viewer_thread.is_alive():
            self._stop_event.clear()
            self._viewer_thread = threading.Thread(
                target=self._viewer_loop,
                daemon=True,
                name="Visualizer"
            )
            self._viewer_thread.start()
        
        # Start recording thread if enabled
        if self.record_path is not None:
            self._recorder_thread = threading.Thread(
                target=self._recorder_loop,
                daemon=True,
                name="VisualizerRecorder"
            )
            self._recorder_thread.start()
    
    def stop(self):
        """Stop the visualization and recording threads."""
        self._stop_event.set()
        if self._viewer_thread is not None:
            self._viewer_thread.join(timeout=1.0)
        if hasattr(self, '_recorder_thread') and self._recorder_thread is not None:
            self._recorder_thread.join(timeout=1.0)
    
    def _recorder_loop(self):
        """
        Separate recording loop that runs at record_freq.
        Decoupled from the viewer loop to ensure consistent recording frequency.
        """
        logger.info("[Visualizer] Recorder thread started at {} Hz", self.record_freq)
        
        while not self._stop_event.is_set():
            tic = time.time()
            
            try:
                # Get current displayed frame
                frame = self.displayed_current_colors
                
                if frame is not None:
                    ts_int = int(tic * 1000)  # timestamp in ms
                    frame_path = os.path.join(self.record_path, f"{ts_int}.jpg")
                    # Use JPEG for faster encoding
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            except Exception as e:
                logger.error("[Visualizer] Recorder error: {}", e)
            
            # Sleep to maintain frequency
            elapsed = time.time() - tic
            sleep_time = self.record_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.info("[Visualizer] Recorder thread stopped")
    
    def _project(self, pt_cam: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Project a 3D point in camera frame to 2D image coordinates.
        
        Args:
            pt_cam: 3D point in camera frame (x, y, z)
        
        Returns:
            (u, v) pixel coordinates, or None if behind camera
        """
        xc, yc, zc = pt_cam
        if zc <= 0:
            return None
        
        fx = self.intrinsic[0, 0]
        fy = self.intrinsic[1, 1]
        cx = self.intrinsic[0, 2]
        cy = self.intrinsic[1, 2]
        
        u = fx * (xc / zc) + cx
        v = fy * (yc / zc) + cy
        
        return int(u), int(v)
    
    def _draw_axis(
        self, 
        img: np.ndarray, 
        p_cam: np.ndarray, 
        R_cam: np.ndarray,
        colors: Optional[list] = None,
    ):
        """
        Draw TCP coordinate axes.
        
        Args:
            img: Image to draw on
            p_cam: TCP position in camera frame
            R_cam: TCP rotation matrix in camera frame
            colors: Optional list of BGR colors for each axis. Defaults to self.axis_colors.
        """
        if colors is None:
            colors = self.axis_colors
        axes = np.eye(3) * self.axis_length
        
        for i in range(3):
            end = p_cam + R_cam @ axes[i]
            p0 = self._project(p_cam)
            p1 = self._project(end[:3])
            
            if p0 and p1:
                cv2.arrowedLine(
                    img, p0, p1, 
                    colors[i], 
                    2, 
                    tipLength=0.15, 
                    line_type=cv2.LINE_AA
                )
    
    def _draw_force(
        self, 
        img: np.ndarray, 
        p_cam: np.ndarray, 
        R_cam: np.ndarray, 
        f_tcp: np.ndarray,
        color: Tuple[int, int, int],
        label_offset: Tuple[int, int] = (0, 0)
    ):
        """
        Draw force vector.
        
        Args:
            img: Image to draw on
            p_cam: TCP position in camera frame
            R_cam: TCP rotation matrix in camera frame
            f_tcp: Force vector in TCP frame
            color: BGR color
            label_offset: Offset for force magnitude label
        """
        f_cam = R_cam @ f_tcp
        end = p_cam + f_cam * self.force_scale
        
        p0 = self._project(p_cam)
        p1 = self._project(end)
        
        if p0 and p1:
            cv2.arrowedLine(
                img, p0, p1, 
                color, 
                2, 
                tipLength=0.2, 
                line_type=cv2.LINE_AA
            )
            
            # Draw force magnitude
            text_pos = (p0[0] + 10 + label_offset[0], p0[1] - 10 + label_offset[1])
            cv2.putText(
                img,
                f"{np.linalg.norm(f_tcp):.1f}N",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                lineType=cv2.LINE_AA
            )
    
    def _draw_mode(self, img: np.ndarray, mode: str):
        """Draw current scheduler mode."""
        if mode == "fast":
            color = (0, 0, 255) # Red for Fast
            mode_label = 'Local'
        else:
            color = (0, 255, 0) # Green for Slow
            mode_label = 'Global'
        
        cv2.putText(
            img,
            f"MODE: {mode_label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            lineType=cv2.LINE_AA
        )
    
    def _draw_legend(self, img: np.ndarray):
        """
        Draw legend in the top-right corner of the image.
        Shows TCP Frame, Interaction Frame, Ext. Wrench Measured, and Ext. Wrench Predicted.
        """
        h, w = img.shape[:2]
        
        # Legend parameters
        legend_items = [
            ("TCP Frame", self.axis_colors, "axis"),
            ("Interaction Frame", self.interaction_frame_colors, "axis"),
            ("Ext. Wrench Measured", self.force_color, "arrow"),
            ("Ext. Wrench Predicted", self.force_predicted_color, "arrow"),
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thickness = 1
        line_height = 22
        margin = 10
        axis_arrow_len = 15
        single_arrow_len = 25
        
        # Calculate legend box size
        max_text_width = 0
        for label, _, _ in legend_items:
            (text_w, _), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            max_text_width = max(max_text_width, text_w)
        
        box_width = margin + single_arrow_len + 10 + max_text_width + margin
        box_height = margin + len(legend_items) * line_height + margin
        
        # Position in top-right corner
        x0 = w - box_width - margin
        y0 = margin
        
        # Draw semi-transparent background
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_width, y0 + box_height), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, dst=img)
        
        # Draw border
        cv2.rectangle(img, (x0, y0), (x0 + box_width, y0 + box_height), (100, 100, 100), 1)
        
        # Draw legend items
        for i, (label, colors, style) in enumerate(legend_items):
            y_center = y0 + margin + i * line_height + line_height // 2
            x_icon_start = x0 + margin
            
            if style == "axis":
                # Draw 3 small arrows for axis (X, Y, Z)
                arrow_len = axis_arrow_len
                # X arrow (horizontal right)
                cv2.arrowedLine(img, (x_icon_start, y_center), 
                               (x_icon_start + arrow_len, y_center), 
                               colors[0], 2, tipLength=0.3, line_type=cv2.LINE_AA)
                # Y arrow (diagonal up-right)
                cv2.arrowedLine(img, (x_icon_start, y_center), 
                               (x_icon_start + arrow_len // 2, y_center - arrow_len // 2), 
                               colors[1], 2, tipLength=0.3, line_type=cv2.LINE_AA)
                # Z arrow (vertical up)
                cv2.arrowedLine(img, (x_icon_start, y_center), 
                               (x_icon_start, y_center - arrow_len), 
                               colors[2], 2, tipLength=0.3, line_type=cv2.LINE_AA)
            else:
                # Draw single arrow for force
                cv2.arrowedLine(img, (x_icon_start, y_center), 
                               (x_icon_start + single_arrow_len, y_center), 
                               colors, 2, tipLength=0.2, line_type=cv2.LINE_AA)
            
            # Draw label text
            text_x = x_icon_start + single_arrow_len + 10
            text_y = y_center + 4
            cv2.putText(img, label, (text_x, text_y), font, font_scale, 
                       (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

    def _draw_force_control(
        self,
        img: np.ndarray,
        T_base_camera: np.ndarray,
        pose_base: np.ndarray,
        force_info: Dict[str, Any]
    ):
        """
        Draw force control frame and wrench.
        Projects from Base/World frame to Camera frame.
        
        Args:
            img: Canvas
            T_base_camera: 4x4 Transform from Base to Camera
            pose_base: 4x4 or 7-element TCP pose in Base frame
            force_info: dict with 'tcp_wrench', 'force_frame'
        """
        force_frame = force_info.get("force_frame")
        tcp_wrench = force_info.get("tcp_wrench")
        
        if pose_base.shape == (4, 4):
            T_base_tcp = pose_base
        elif pose_base.size == 7:
            T_base_tcp = xyz_rot_to_mat(pose_base, rotation_rep=RotationType.QUATERNION)
        else:
            logger.error("[Visualizer] Invalid pose_base shape: {}", pose_base.shape)
            return

        if force_frame is not None:
            T_tcp_compliance = xyz_rot_to_mat(force_frame, rotation_rep=RotationType.QUATERNION)
            T_base_compliance = T_base_tcp @ T_tcp_compliance
            T_cam_compliance = T_base_camera @ T_base_compliance
            R_cam_compliance = T_cam_compliance[:3, :3]
            
            # Use TCP position or compliance frame position based on if_translation
            if self.if_translation:
                p_cam = T_cam_compliance[:3, 3]
            else:
                # Draw interaction frame at TCP position (ignore translation)
                T_cam_tcp = T_base_camera @ T_base_tcp
                p_cam = T_cam_tcp[:3, 3]
            
            self._draw_axis(img, p_cam, R_cam_compliance, colors=self.interaction_frame_colors)
        
        if tcp_wrench is not None:
            # force_compliance = tcp_wrench[:3]
            force_compliance = -tcp_wrench[:3] # negative for display
            # Use same colors as interaction frame: Yellow, Cyan, Magenta (BGR)
            colors = self.interaction_frame_colors
            
            for i in range(3):
                f_mag = force_compliance[i]
                if abs(f_mag) > 1e-6:

                    v_comp = np.zeros(3)
                    v_comp[i] = f_mag
                    
                    v_cam = R_cam_compliance[:, i] * f_mag
                    
                    start_pt = p_cam
                    end_pt = start_pt + v_cam * self.force_scale
                    
                    p0 = self._project(start_pt)
                    p1 = self._project(end_pt)
                    
                    if p0 and p1:
                        cv2.arrowedLine(img, p0, p1, colors[i], 2, tipLength=0.2, line_type=cv2.LINE_AA)
                        # Display value at arrow end (p1)
                        label_pos = (p1[0] + 5, p1[1] + 5)
                        cv2.putText(
                            img, 
                            f"{f_mag:.1f}", 
                            label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, 
                            colors[i], 
                            1, 
                            lineType=cv2.LINE_AA
                        )
    
    def _draw_waypoints(
        self, 
        img: np.ndarray, 
        scheduler,
        T_base_camera: np.ndarray,
        robot_name: str
    ):
        """
        Draw predicted waypoints with different styles.
        
        Waypoint types:
        - raw: Large circles with decreasing opacity
        - head_interp: Medium circles for head segment interpolation
        - interp: Small circles for trajectory interpolation
        - head_interp_runtime: Green circles for runtime head interpolation
        - interp_profiled: Green circles for time-profiled interpolation
        
        Args:
            img: Image to draw on
            scheduler: Scheduler with waypoint queue
            T_base_camera: Transformation matrix from base to camera
        """
        try:
            waypoints = scheduler.get_queue_tcps(robot_name)
        except Exception as e:
            logger.exception("[Visualizer] Failed to get waypoints: {}", e)
            return
        
        num_pts = len(waypoints)
        if num_pts == 0:
            return
        
        RAW_RADIUS = 5
        pt_prev = None
        
        for i, wp in enumerate(waypoints):
            # Expect (pose7, ee_cmd, waypoint_type)
            pose7, wp_type = wp
            ee_cmd = None
            
            # Project point from base to camera frame
            pt_cam = apply_mat_to_point(pose7[:3], T_base_camera)
            pt = self._project(pt_cam)
            if pt is None:
                continue
            
            # Determine color based on gripper width
            color = (255, 200, 200)  # Default: light red
            width = None
            
            if ee_cmd is not None:
                if isinstance(ee_cmd, (list, tuple, np.ndarray)) and len(ee_cmd) > 0:
                    width = float(ee_cmd[0])
                elif isinstance(ee_cmd, (int, float)):
                    width = float(ee_cmd)
            
            if width is not None:
                if width < 0.05:
                    color = (255, 100, 100)  # Dark red: gripper closed
                elif width > 0.05:
                    color = (100, 100, 255)  # Dark blue: gripper open
            
            # Override color for runtime waypoints
            if wp_type in ("head_interp_runtime", "interp_profiled"):
                color = (0, 255, 0)  # Green
            
            # Determine style based on waypoint type
            if wp_type == "raw":
                radius = RAW_RADIUS
                alpha_traj = i / max(num_pts - 1, 1)
                opacity = 1.0 - 0.9 * alpha_traj
                opacity = float(np.clip(opacity, 0.1, 1.0))
                thickness_line = 2
            elif wp_type == "head_interp":
                radius = max(1, int(round(RAW_RADIUS * 0.5)))
                opacity = 0.8
                thickness_line = 2
            elif wp_type == "interp":
                radius = max(1, int(round(RAW_RADIUS * 0.2)))
                opacity = 0.8
                thickness_line = 1
            elif wp_type in ("head_interp_runtime", "interp_profiled"):
                radius = max(1, int(round(RAW_RADIUS * 0.5)))
                opacity = 0.8
                thickness_line = 2
            else:
                # Fallback: treat as raw
                radius = RAW_RADIUS
                alpha_traj = i / max(num_pts - 1, 1)
                opacity = 1.0 - 0.9 * alpha_traj
                opacity = float(np.clip(opacity, 0.1, 1.0))
                thickness_line = 2
            
            # Draw point with transparency
            overlay_dot = img.copy()
            cv2.circle(overlay_dot, pt, radius, color, -1, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay_dot, opacity, img, 1.0 - opacity, 0, dst=img)
            
            # Draw line connecting waypoints
            if pt_prev is not None:
                overlay_line = img.copy()
                cv2.line(overlay_line, pt, pt_prev, color, thickness_line, lineType=cv2.LINE_AA)
                cv2.addWeighted(overlay_line, opacity, img, 1.0 - opacity, 0, dst=img)
            
            pt_prev = pt
    
    def _viewer_loop(self):
        """Main visualization loop."""
        while not self._stop_event.is_set():
            tic = time.time()
            
            try:
                # Get camera image
                shm_payload = self.image_shm.execute()
                color_image = self._extract_color_image(shm_payload)

                # Explicit pre-checks around the known crash site (OpenCV cvtColor)
                if not isinstance(color_image, np.ndarray):
                    if isinstance(shm_payload, dict):
                        logger.error(
                            "[Visualizer] Failed to extract color ndarray from shm payload dict. keys={}",
                            list(shm_payload.keys()),
                        )
                    else:
                        logger.error(
                            "[Visualizer] Failed to extract color ndarray from shm payload: {}",
                            self._describe_obj(shm_payload),
                        )

                    # Skip this frame (avoid cvtColor crash)
                    elapsed = time.time() - tic
                    sleep_time = self.dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue
                else:
                    if color_image.ndim != 3:
                        logger.warning(
                            "[Visualizer] Unexpected image ndim (expected HxWxC): {}",
                            self._describe_obj(color_image),
                        )
                    elif color_image.shape[-1] not in (3, 4):
                        logger.warning(
                            "[Visualizer] Unexpected channel count (expected 3/4): {}",
                            self._describe_obj(color_image),
                        )

                # Support RGB or RGBA
                if color_image.ndim == 3 and color_image.shape[-1] == 4:
                    bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGBA2BGR)
                else:
                    bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                overlay = bgr_image.copy()
                
                # Check if any scheduler is in "fast" mode
                is_fast_mode = False
                for scheduler_key in list(self.schedulers.keys()):
                    if scheduler_key in self.schedulers:
                        scheduler = self.schedulers[scheduler_key]
                        if hasattr(scheduler, "current_mode") and scheduler.current_mode == "fast":
                            is_fast_mode = True
                            break
                
                # Draw agents (current TCP pose and force)
                # In fast mode, skip TCP frame (only show interaction frame)
                agent_keys = list(self.agents.keys())
                for agent_key in agent_keys:
                    if agent_key not in self.agents:
                        continue
                    
                    agent = self.agents[agent_key]
                    
                    # Iterate all robots in the agent
                    # Assuming agent has platform_config.robot_names or config.robots.keys()
                    # RealAgent has platform_config.robot_names
                    robot_names = []
                    if hasattr(agent, "platform_config"):
                        robot_names = agent.platform_config.robot_names
                    elif hasattr(agent, "config") and hasattr(agent.config, "robots"):
                        robot_names = list(agent.config.robots.keys())
                    
                    for robot_name in robot_names:
                        try:
                            # Get robot base pose
                            T_world_base = np.eye(4)
                            if hasattr(agent, "config") and hasattr(agent.config, "robot_poses"):
                                T_world_base = agent.config.robot_poses.get(robot_name, np.eye(4))
                            
                            # Get current TCP pose in base frame
                            pose_base = self._get_tcp_pose(agent, robot_name)
                            
                            # Compute transformation: base -> world -> camera
                            T_base_camera = self.T_camera_world @ T_world_base
                            
                            # Transform pose to camera frame
                            pose_cam = apply_mat_to_pose(
                                pose_base,
                                T_base_camera,
                                rotation_rep=RotationType.QUATERNION,
                            )
                            
                            # Convert to 4x4 matrix for drawing
                            pose_in_cam = xyz_rot_to_mat(
                                pose_cam,
                                rotation_rep=RotationType.QUATERNION
                            )
                            
                            # Draw TCP axes only in slow/global mode
                            if not is_fast_mode:
                                self._draw_axis(overlay, pose_in_cam[:3, 3], pose_in_cam[:3, :3])
                            
                            # Draw force/torque if available
                            if hasattr(agent, 'get_force_torque_tcp'):
                                try:
                                    ft_tcp = agent.get_force_torque_tcp(robot_name)
                                    f_tcp_to_draw = -ft_tcp[:3]  # Negative for display
                                    self._draw_force(
                                        overlay, 
                                        pose_in_cam[:3, 3], 
                                        pose_in_cam[:3, :3], 
                                        f_tcp_to_draw, 
                                        self.force_color
                                    )
                                except Exception as e:
                                    logger.error(
                                        "[Visualizer] Failed to draw agent force tcp: robot={}, err={}",
                                        robot_name,
                                        e,
                                    )
                            
                            pose_base_mat = xyz_rot_to_mat(pose_base, rotation_rep=RotationType.QUATERNION)

                        except Exception as e:
                            logger.exception(
                                "[Visualizer] Failed to draw agent: agent={}, robot={}, err={}",
                                agent_key,
                                robot_name,
                                e,
                            )
                
                # Draw schedulers (predicted waypoints and forces)
                scheduler_keys = list(self.schedulers.keys())
                for scheduler_key in scheduler_keys:
                    if scheduler_key not in self.schedulers:
                        continue
                    
                    scheduler = self.schedulers[scheduler_key]
                    
                    # Draw Mode if available (once per scheduler)
                    if hasattr(scheduler, "current_mode"):
                        self._draw_mode(overlay, scheduler.current_mode)
                    
                    # Get all predicted wrenches
                    predicted_wrenches_map = {}
                    if hasattr(scheduler, 'get_queue_wrenches'):
                        predicted_wrenches_map = scheduler.get_queue_wrenches()
                    
                    # Iterate all robots known to scheduler
                    robot_names = []
                    if hasattr(scheduler, "platform_config"):
                        robot_names = scheduler.platform_config.robot_names
                    elif hasattr(scheduler, "agent") and hasattr(scheduler.agent, "platform_config"):
                        robot_names = scheduler.agent.platform_config.robot_names
                        
                    for robot_name in robot_names:
                        try:
                            # Resolve T_world_base from scheduler's agent
                            T_world_base = np.eye(4)
                            if hasattr(scheduler, "agent") and hasattr(scheduler.agent.config, "robot_poses"):
                                T_world_base = scheduler.agent.config.robot_poses.get(robot_name, np.eye(4))
                            
                            # Compute transformation: base -> world -> camera
                            T_base_camera = self.T_camera_world @ T_world_base
                            
                            # Draw predicted force/torque if available for this robot
                            if robot_name in predicted_wrenches_map:
                                # We need current TCP pose to place the vector
                                # Use scheduler.agent which should track current state
                                if hasattr(scheduler, 'agent'):
                                    pose_base = self._get_tcp_pose(scheduler.agent, robot_name)
                                    pose_base_mat = xyz_rot_to_mat(pose_base, rotation_rep=RotationType.QUATERNION)
                                                                        
                                    self._draw_force_control(
                                        overlay,
                                        T_base_camera,
                                        pose_base_mat,
                                        predicted_wrenches_map[robot_name]
                                    )
                            
                            # Draw waypoints
                            self._draw_waypoints(overlay, scheduler, T_base_camera, robot_name)
                        
                        except Exception as e:
                            logger.exception(
                                "[Visualizer] Failed to draw scheduler: scheduler={}, robot={}, err={}",
                                scheduler_key,
                                robot_name,
                                e,
                            )
                
                # Draw legend
                self._draw_legend(overlay)
                
                # Display image
                self.displayed_current_colors = overlay
                cv2.imshow("viewer", overlay)
                
                # Note: Recording is now handled by _recorder_loop in a separate thread
                
                # Check for ESC key
                if cv2.waitKey(1) == 27:
                    break
            
            except Exception as e:
                logger.exception(
                    "[Visualizer] Viewer loop error: err={}, last_color_image={}",
                    e,
                    self._describe_obj(locals().get("color_image", None)),
                )
            
            # Sleep to maintain frequency
            elapsed = time.time() - tic
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        cv2.destroyWindow("viewer")

