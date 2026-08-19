"""
Module: hand_tracker
Project: Touchless Docker Dashboard
Author: Python Software Architect & DevOps Engineer
Description: SOLID-compliant, OOP-designed wrapper for MediaPipe Hand Landmarker.
             It translates raw MediaPipe results into decoupled, type-safe Python dataclasses.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import mediapipe as mp
import numpy as np


class TrackerError(Exception):
    """Base exception for hand tracker related issues."""
    pass


class ModelNotFoundError(TrackerError):
    """Exception raised when the MediaPipe model asset file cannot be located."""
    pass


@dataclass(frozen=True)
class HandLandmark:
    """
    Represents a single hand landmark with both normalized and pixel coordinates.
    Using frozen=True makes instances immutable and hashable (thread-safe).
    """
    id: int
    x: float  # Normalized X coordinate [0.0, 1.0]
    y: float  # Normalized Y coordinate [0.0, 1.0]
    z: float  # Normalized Z coordinate (depth representation)
    px: int   # Computed Pixel X coordinate on the frame
    py: int   # Computed Pixel Y coordinate on the frame


@dataclass(frozen=True)
class HandDetection:
    """
    Represents a fully processed hand detection containing its handedness (label) 
    and list of landmarks.
    """
    label: str  # "Left" or "Right"
    confidence: float
    landmarks: List[HandLandmark]


class HandTracker:
    """
    Manages the lifecycle and execution of Google MediaPipe Hand Landmarker.
    Processes BGR frames (OpenCV format) and converts results into decoupled, 
    strongly-typed Python dataclasses.
    """

    def __init__(
        self,
        model_path: str = "hand_landmarker.task",
        num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_presence_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7
    ) -> None:
        """
        Initializes the HandTracker.

        Args:
            model_path (str): File path to the hand_landmarker.task model.
            num_hands (int): Maximum number of hands to detect. Defaults to 2.
            min_detection_confidence (float): Minimum confidence for detection. Defaults to 0.7.
            min_presence_confidence (float): Minimum confidence for presence. Defaults to 0.7.
            min_tracking_confidence (float): Minimum confidence for tracking. Defaults to 0.7.
        """
        self.model_path: str = model_path
        self.num_hands: int = num_hands
        self.min_detection_confidence: float = min_detection_confidence
        self.min_presence_confidence: float = min_presence_confidence
        self.min_tracking_confidence: float = min_tracking_confidence
        
        self.landmarker: Optional[mp.tasks.vision.HandLandmarker] = None
        self._is_initialized: bool = False

    def initialize(self) -> 'HandTracker':
        """
        Initializes the underlying MediaPipe HandLandmarker instance.

        Raises:
            ModelNotFoundError: If the model file is missing.
            TrackerError: If MediaPipe fails to initialize.
        
        Returns:
            HandTracker: Returns self for chaining support.
        """
        if self._is_initialized:
            return self

        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise ModelNotFoundError(
                f"MediaPipe model file not found at: '{self.model_path}'. "
                "Please download 'hand_landmarker.task' and place it in the specified path."
            )

        try:
            # Set up MediaPipe Hand Landmarker options in VIDEO mode (ideal for sequential frames)
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=self.num_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            self.landmarker = HandLandmarker.create_from_options(options)
            self._is_initialized = True
            print(f"[INFO] HandTracker initialized successfully with model: {self.model_path}")
            return self
            
        except Exception as e:
            raise TrackerError(f"Failed to initialize MediaPipe HandLandmarker: {str(e)}") from e

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> List[HandDetection]:
        """
        Processes a single BGR OpenCV frame and returns a list of custom HandDetection.

        Args:
            frame (np.ndarray): OpenCV frame image (BGR format).
            timestamp_ms (int): Monotonically increasing timestamp in milliseconds.

        Raises:
            TrackerError: If tracker is not initialized or processing fails.

        Returns:
            List[HandDetection]: List of successfully detected hands with landmarks.
        """
        if not self._is_initialized or self.landmarker is None:
            raise TrackerError("HandTracker is not initialized. Please call .initialize() first.")

        if frame is None:
            return []

        h, w, _ = frame.shape

        try:
            # Convert OpenCV BGR image to RGB (MediaPipe requirement)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Wrap in MediaPipe Image format
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Perform synchronous detection for video
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            detections: List[HandDetection] = []
            
            if results.hand_landmarks and results.handedness:
                # Zip hand landmarks and handedness together
                for landmarks, handedness_list in zip(results.hand_landmarks, results.handedness):
                    if not handedness_list:
                        continue
                        
                    # Get primary handedness category
                    primary_hand = handedness_list[0]
                    label = primary_hand.category_name  # "Left" or "Right"
                    confidence = primary_hand.score
                    
                    processed_landmarks: List[HandLandmark] = []
                    for idx, lm in enumerate(landmarks):
                        # Calculate physical pixel coordinates
                        px = int(lm.x * w)
                        py = int(lm.y * h)
                        
                        processed_landmarks.append(
                            HandLandmark(
                                id=idx,
                                x=lm.x,
                                y=lm.y,
                                z=lm.z,
                                px=px,
                                py=py
                            )
                        )
                    
                    detections.append(
                        HandDetection(
                            label=label,
                            confidence=confidence,
                            landmarks=processed_landmarks
                        )
                    )
            
            return detections

        except Exception as e:
            raise TrackerError(f"Error occurred during frame processing: {str(e)}") from e

    def close(self) -> None:
        """
        Releases the underlying MediaPipe Landmarker resources.
        """
        if self.landmarker is not None:
            try:
                self.landmarker.close()
                print("[INFO] HandTracker MediaPipe resources closed successfully.")
            except Exception as e:
                print(f"[WARNING] Error while closing HandTracker resources: {e}")
            finally:
                self.landmarker = None
                self._is_initialized = False

    def __enter__(self) -> 'HandTracker':
        """Supports the context manager protocol."""
        return self.initialize()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensures resources are safely closed upon exiting context."""
        self.close()


# Quick verification script that runs only if this file is executed directly
if __name__ == "__main__":
    print("[TEST] Running HandTracker verification...")
    # Mocking a black frame for testing
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # We expect a failure if the task file is not present in the workspace
    try:
        # Looking for model in current directory or model path
        tracker = HandTracker(model_path="hand_landmarker.task")
        with tracker as initialized_tracker:
            print("[TEST] Tracker initialized in with block successfully!")
            results = initialized_tracker.process_frame(mock_frame, timestamp_ms=int(time.time() * 1000))
            print(f"[TEST] Processed mock frame successfully. Detected hands: {len(results)}")
            
    except ModelNotFoundError as err:
        print(f"\n[TEST EXCEPTION CAUGHT (EXPECTED)] Model file missing: {err}")
    except Exception as err:
        print(f"\n[TEST ERROR] Unexpected error: {err}")
