"""
Module: camera_streamer
Project: Touchless Docker Dashboard
Author: Python Software Architect & DevOps Engineer
Description: Thread-safe, asynchronous camera streaming module using OpenCV and Threading.
             This implementation adheres to SOLID principles (Single Responsibility) and OOP best practices.
"""

import threading
import time
from typing import Tuple, Union, Optional
import cv2
import numpy as np


class CameraError(Exception):
    """Base exception class for Camera related issues."""
    pass


class CameraConnectionError(CameraError):
    """Exception raised when camera connection fails or cannot be established."""
    pass


class CameraStreamer:
    """
    A class that manages a camera stream using a dedicated background thread 
    to fetch frames asynchronously, avoiding I/O bottlenecks in the main application loop.

    Attributes:
        src (Union[int, str]): The camera source index (int) or stream URL (str).
        width (int): The target width of the frame resolution.
        height (int): The target height of the frame resolution.
        cap (cv2.VideoCapture): The OpenCV VideoCapture object.
        frame (Optional[np.ndarray]): The latest frame fetched from the camera.
        grabbed (bool): True if the latest frame was successfully retrieved, False otherwise.
        started (bool): Flag indicating if the background thread is currently running.
        read_lock (threading.Lock): A lock for synchronizing thread-safe reads and writes of frames.
        thread (Optional[threading.Thread]): The background worker thread object.
    """

    def __init__(self, src: Union[int, str] = 0, width: int = 640, height: int = 480) -> None:
        """
        Initializes the CameraStreamer with configuration.

        Args:
            src (Union[int, str]): Camera index (typically 0 for built-in camera) or stream path.
            width (int): Target frame width. Defaults to 640.
            height (int): Target frame height. Defaults to 480.
        """
        self.src: Union[int, str] = src
        self.width: int = width
        self.height: int = height
        
        # Thread safety and control
        self.read_lock: threading.Lock = threading.Lock()
        self.started: bool = False
        self.thread: Optional[threading.Thread] = None
        
        # State
        self.frame: Optional[np.ndarray] = None
        self.grabbed: bool = False
        
        # OpenCV Capture Object
        self.cap: Optional[cv2.VideoCapture] = None

    def _initialize_camera(self) -> None:
        """
        Establishes connection to the video capture device and configures resolution parameters.

        Raises:
            CameraConnectionError: If the camera cannot be opened or initialized.
        """
        try:
            self.cap = cv2.VideoCapture(self.src)
            if not self.cap.isOpened():
                raise CameraConnectionError(
                    f"Failed to open video source: {self.src}. "
                    "Please check if the device is plugged in or already in use."
                )
            
            # Configure resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Read an initial frame to populate buffer and verify it works
            grabbed, frame = self.cap.read()
            if not grabbed:
                raise CameraConnectionError(
                    f"Camera device {self.src} was opened successfully, but failed to retrieve the first frame."
                )
            
            with self.read_lock:
                self.frame = frame
                self.grabbed = grabbed
                
        except Exception as e:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            raise CameraConnectionError(f"Error during camera initialization: {str(e)}") from e

    def start(self) -> 'CameraStreamer':
        """
        Starts the background thread to continuously fetch frames.

        Returns:
            CameraStreamer: Returns self for chaining support (fluent interface).

        Raises:
            CameraConnectionError: If camera connection fails.
        """
        if self.started:
            return self

        # Initialize connection before launching the thread
        self._initialize_camera()
        
        self.started = True
        self.thread = threading.Thread(target=self._update, name="CameraStreamerWorker", daemon=True)
        self.thread.start()
        return self

    def _update(self) -> None:
        """
        The background thread worker loop. Continuously pulls frames from cv2.VideoCapture.
        If reading fails repeatedly or the stream is disconnected, raises a warning and stops.
        """
        consecutive_failures = 0
        max_failures = 5

        while self.started:
            grabbed, frame = self.cap.read()
            
            if not grabbed:
                consecutive_failures += 1
                time.sleep(0.01) # Yield CPU
                if consecutive_failures >= max_failures:
                    print(f"[ERROR] CameraStreamer: Connection lost after {consecutive_failures} failed read attempts.")
                    with self.read_lock:
                        self.grabbed = False
                    self.started = False
                    break
                continue
            
            consecutive_failures = 0
            
            with self.read_lock:
                self.frame = frame
                self.grabbed = grabbed

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Thread-safe method to read the most recently captured frame.

        Returns:
            Tuple[bool, Optional[np.ndarray]]:
                - bool: True if the frame was grabbed successfully, False otherwise.
                - np.ndarray: The actual frame image (BGR format), or None if unavailable.
        """
        # The worker replaces the frame reference instead of mutating an existing
        # array, so take a consistent snapshot under the lock and copy it afterward.
        # This keeps the camera worker from waiting while a full HD frame is copied.
        with self.read_lock:
            frame = self.frame
            grabbed = self.grabbed

        frame_copy = frame.copy() if frame is not None else None
        return grabbed, frame_copy

    def stop(self) -> None:
        """
        Stops the background worker thread and safely releases camera hardware resources.

        Cleanup is intentionally performed even when the worker has already marked
        itself stopped after repeated frame-read failures.
        """
        self.started = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)  # Gracefully wait for the worker to exit

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        print("[INFO] CameraStreamer stopped and hardware resources released safely.")

    def __enter__(self) -> 'CameraStreamer':
        """Supports the 'with' statement context manager protocol."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Supports the 'with' statement context manager protocol to ensure safe release of resources."""
        self.stop()


# Quick verification script that runs only if this file is executed directly
if __name__ == "__main__":
    print("[TEST] Running CameraStreamer verification...")
    # Using index 0 (default camera). We'll catch error if no camera is connected.
    try:
        with CameraStreamer(src=0, width=640, height=480) as streamer:
            print("[TEST] Camera initialized and background thread started successfully!")
            
            # Read a few frames to verify asynchronous fetch
            for i in range(5):
                success, frame = streamer.read()
                if success and frame is not None:
                    print(f"[TEST] Frame {i+1} fetched successfully! Shape: {frame.shape}")
                else:
                    print(f"[TEST] Warning: Frame {i+1} could not be read yet.")
                time.sleep(0.5)
                
    except CameraConnectionError as err:
        print(f"\n[TEST ERROR] Camera Connection Failed: {err}")
    except Exception as err:
        print(f"\n[TEST ERROR] Unexpected error: {err}")


class SimulatedCameraStreamer:
    """
    A simulated camera streamer that generates synthetic video frames 
    representing a futuristic holographic interface. Used when a physical 
    webcam is not available.
    """
    def __init__(self, src: Union[int, str] = 0, width: int = 1280, height: int = 720) -> None:
        self.src = src
        self.width = width
        self.height = height
        self.read_lock = threading.Lock()
        self.started = False
        self.thread = None
        self.frame = None
        self.grabbed = False
        self.start_time = time.time()

    def start(self) -> 'SimulatedCameraStreamer':
        if self.started:
            return self
        self.started = True
        self.grabbed = True
        # Generate initial frame
        self._generate_frame()
        self.thread = threading.Thread(target=self._update, name="SimulatedCameraStreamerWorker", daemon=True)
        self.thread.start()
        print("[INFO] SimulatedCameraStreamer started successfully (Simulation Mode).")
        return self

    def _generate_frame(self) -> None:
        import math
        # Create dark background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Deep space gradient background
        frame[:] = (12, 10, 8) # Very dark slate blue/black
        
        t = time.time() - self.start_time
        
        # Draw a beautiful digital sci-fi background
        # 1. Subtle horizontal scanning lines
        for y in range(0, self.height, 4):
            cv2.line(frame, (0, y), (self.width, y), (16, 14, 12), 1)
            
        # 2. Glowing circular grid in center-right (Radar Area)
        radar_center = (910, 360)
        # Draw concentric expanding circles
        for r_offset in [0, 40, 80, 120, 160]:
            r = int((r_offset + t * 20) % 200)
            alpha = max(0.0, min(1.0, (200 - r) / 200.0))
            color = (int(0 * alpha), int(100 * alpha), int(100 * alpha)) # Cyan glow
            cv2.circle(frame, radar_center, r, color, 1)

        # 3. Rotating radar sweep line
        angle = t * 1.5
        end_x = int(radar_center[0] + 180 * math.cos(angle))
        end_y = int(radar_center[1] + 180 * math.sin(angle))
        cv2.line(frame, radar_center, (end_x, end_y), (0, 120, 120), 2)
        
        # 4. Animated corner crosshairs
        c_len = 30
        margin = 15
        color_cyan = (255, 255, 0)
        # Top-Left
        cv2.line(frame, (margin, margin), (margin + c_len, margin), color_cyan, 1)
        cv2.line(frame, (margin, margin), (margin, margin + c_len), color_cyan, 1)
        # Top-Right
        cv2.line(frame, (self.width - margin, margin), (self.width - margin - c_len, margin), color_cyan, 1)
        cv2.line(frame, (self.width - margin, margin), (self.width - margin, margin + c_len), color_cyan, 1)
        # Bottom-Left
        cv2.line(frame, (margin, self.height - margin), (margin + c_len, self.height - margin), color_cyan, 1)
        cv2.line(frame, (margin, self.height - margin), (margin, self.height - margin - c_len), color_cyan, 1)
        # Bottom-Right
        cv2.line(frame, (self.width - margin, self.height - margin), (self.width - margin - c_len, self.height - margin), color_cyan, 1)
        cv2.line(frame, (self.width - margin, self.height - margin), (self.width - margin, self.height - margin - c_len), color_cyan, 1)

        # 5. Adding status overlays
        cv2.putText(frame, "SIMULATOR CAM", (30, self.height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1, cv2.LINE_AA)
        
        # Add decorative data blocks
        for i in range(3):
            val = int(math.sin(t + i) * 50 + 50)
            cv2.rectangle(frame, (1150 + i * 20, 50), (1160 + i * 20, 50 + val), (150, 100, 0), -1)

        with self.read_lock:
            self.frame = frame
            self.grabbed = True

    def _update(self) -> None:
        while self.started:
            self._generate_frame()
            time.sleep(1 / 30.0) # 30 FPS

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.read_lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame_copy

    def stop(self) -> None:
        if not self.started:
            return
        self.started = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        print("[INFO] SimulatedCameraStreamer stopped successfully.")

    def __enter__(self) -> 'SimulatedCameraStreamer':
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def get_camera_streamer(src: Union[int, str] = 0, width: int = 1280, height: int = 720, force_simulate: bool = False) -> Union[CameraStreamer, SimulatedCameraStreamer]:
    """
    Factory function following the Dependency Inversion Principle.
    Attempts to initialize the physical CameraStreamer; if it fails or if force_simulate is True,
    it gracefully falls back to SimulatedCameraStreamer.
    """
    if force_simulate:
        print("[INFO] Forcing Simulation Mode for Camera Streamer.")
        return SimulatedCameraStreamer(src, width, height)
    try:
        # Try real camera
        streamer = CameraStreamer(src, width, height)
        # Test initialize to verify
        streamer._initialize_camera()
        # If it succeeds, release and return it (since the context manager in main will start it)
        if streamer.cap and streamer.cap.isOpened():
            streamer.cap.release()
        return streamer
    except Exception as e:
        print(f"[INFO] Running Camera Streamer in simulation mode. Reason: {e}")
        return SimulatedCameraStreamer(src, width, height)
