"""
Main Orchestrator: main.py
Project: Touchless Docker Dashboard
Author: Python Software Architect & DevOps Engineer
Description: Integrates CameraStreamer, HandTracker, DockerService, and UIManager.
             Implements asynchronous background Docker stats fetching to prevent lag,
             gesture cursor mapping, pinch-to-click detection, and debounce control.
"""

import os
import sys
import threading
import time
import math
import statistics
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np

# Import custom OOP components
from camera_streamer import get_camera_streamer, CameraConnectionError
from hand_tracker import HandTracker, HandDetection, TrackerError
from docker_service import get_docker_service, IDockerService, DockerContainer
from ui_manager import UIManager


# Shared thread-safe resources
shared_containers: List[DockerContainer] = []
containers_lock = threading.Lock()
running = True


def docker_stats_fetcher(service: IDockerService) -> None:
    """
    Background worker function that polls the Docker SDK/Simulator for real-time stats.
    Offloading this process ensures the main UI thread runs at a silky smooth 30+ FPS.
    """
    global shared_containers, running
    print("[INFO] Background Docker statistics worker started.")
    
    while running:
        try:
            # list_containers() might take up to 200ms depending on Docker daemon workload.
            # Running this in a background thread prevents UI freezing.
            fresh_containers = service.list_containers()
            with containers_lock:
                shared_containers = fresh_containers
        except Exception as e:
            print(f"[ERROR] Docker Stats Fetcher: {e}")
        
        # Poll and update stats every 1 second
        time.sleep(1.0)
        
    print("[INFO] Background Docker statistics worker stopped.")


def calculate_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Helper method to calculate Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class FrameProfiler:
    """Opt-in rolling profiler for the main camera/UI pipeline.

    Enable with GHOST_PROFILE=1. When disabled, stage timers return immediately
    and no profiling output is printed.
    """

    STAGES = ("camera_read", "preprocess", "hand_detection", "ui_render", "display")

    def __init__(self, enabled: bool = False, report_interval: float = 5.0,
                 window_size: int = 120) -> None:
        self.enabled = enabled
        self.report_interval = report_interval
        self.samples: Dict[str, Deque[float]] = {
            stage: deque(maxlen=window_size) for stage in self.STAGES
        }
        self._report_started = time.monotonic()
        self._frames = 0

    def start_stage(self) -> int:
        return time.perf_counter_ns() if self.enabled else 0

    @staticmethod
    def percentile(values: List[float], percentile: float) -> float:
        """Return a nearest-rank percentile from a non-empty sample."""
        if not values:
            raise ValueError("percentile requires at least one sample")
        if not 0 < percentile <= 100:
            raise ValueError("percentile must be in the range (0, 100]")
        ordered = sorted(values)
        rank = max(1, math.ceil((percentile / 100) * len(ordered)))
        return ordered[rank - 1]

    def record(self, stage: str, started_ns: int) -> None:
        if self.enabled and started_ns:
            self.samples[stage].append((time.perf_counter_ns() - started_ns) / 1_000_000)

    def complete_frame(self) -> None:
        if not self.enabled:
            return
        self._frames += 1
        now = time.monotonic()
        elapsed = now - self._report_started
        if elapsed < self.report_interval:
            return

        fps = self._frames / elapsed if elapsed > 0 else 0.0
        details = []
        for stage, values in self.samples.items():
            if not values:
                continue
            p95 = self.percentile(list(values), 95)
            details.append(f"{stage} avg={statistics.fmean(values):.1f}ms p95={p95:.1f}ms")
        print(f"[PROFILE] FPS={fps:.1f} | " + " | ".join(details))
        self._report_started = now
        self._frames = 0


def main() -> None:
    global shared_containers, running

    # Reset the stop flag so main() can be safely invoked again in the same process.
    running = True

    profiler = FrameProfiler(enabled=os.getenv("GHOST_PROFILE", "0").strip() == "1")
    if profiler.enabled:
        print("[INFO] Performance profiling enabled (GHOST_PROFILE=1).")

    # 1. Initialize UIManager (1280x720 HD display target)
    width, height = 1280, 720
    ui = UIManager(width=width, height=height)

    # 2. Instantiate and establish Docker Service
    print("[INFO] Setting up Docker Controller Service...")
    docker_service = get_docker_service()

    # Fetch initial container state before starting background polling.
    with containers_lock:
        shared_containers = docker_service.list_containers()

    # 3. Instantiate CameraStreamer and HandTracker before starting the worker.
    # This avoids leaving a background stats thread running if tracker setup fails.
    # Camera index 1 is the working webcam on this machine (index 0 is unavailable).
    camera = get_camera_streamer(src=1, width=width, height=height)
    tracker = HandTracker(model_path="hand_landmarker.task", num_hands=2)

    # Launch async stats updater thread after the main components are constructed.
    stats_thread = threading.Thread(
        target=docker_stats_fetcher,
        args=(docker_service,),
        name="DockerStatsUpdater",
        daemon=True
    )
    stats_thread.start()

    # State variables for Gesture Control and UX Debouncing
    last_click_time = 0.0
    click_debounce_interval = 1.5  # 1.5 seconds delay between clicks to prevent multiple registers
    pinch_released = True         # Must release pinch before clicking again
    last_timestamp_ms = 0

    print("\n" + "="*50)
    print("      TOUCHLESS DOCKER CONTROLLER STARTED")
    print("      Press 'q' inside video window to EXIT")
    print("="*50 + "\n")

    try:
        # Open camera and hand tracker using Context Managers to guarantee resource release
        with camera as streamer, tracker as hand_detector:
            while running:
                # A. Read asynchronous webcam frame
                stage_started = profiler.start_stage()
                success, frame = streamer.read()
                profiler.record("camera_read", stage_started)
                if not success or frame is None:
                    time.sleep(0.01)
                    continue

                # Keep the camera's native frame when it already matches the target size;
                # avoid a redundant full-HD resize on every frame.
                stage_started = profiler.start_stage()
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                # Mirror for natural selfie-style interaction.
                frame = cv2.flip(frame, 1)
                profiler.record("preprocess", stage_started)

                # B. Perform hand landmark extraction. MediaPipe VIDEO mode requires
                # strictly increasing timestamps; monotonic time is immune to wall-clock changes.
                timestamp_ms = max(last_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
                last_timestamp_ms = timestamp_ms
                stage_started = profiler.start_stage()
                try:
                    detection_frame = cv2.resize(
                        frame, (640, 360), interpolation=cv2.INTER_AREA
                    )
                    detected_hands = hand_detector.process_frame(
                        detection_frame, timestamp_ms
                    )
                    # MediaPipe landmarks are in the smaller inference frame;
                    # scale pixel positions back for the 1280x720 UI hit-testing.
                    hands = [
                        HandDetection(
                            label=hand.label,
                            confidence=hand.confidence,
                            landmarks=[
                                type(lm)(
                                    id=lm.id, x=lm.x, y=lm.y, z=lm.z,
                                    px=lm.px * 2, py=lm.py * 2
                                )
                                for lm in hand.landmarks
                            ],
                        )
                        for hand in detected_hands
                    ]
                except TrackerError as tracker_err:
                    print(f"[WARNING] Hand tracker error: {tracker_err}")
                    hands = []
                profiler.record("hand_detection", stage_started)

                # C. Extract virtual cursor and pinch gesture status
                cursor_pos: Optional[Tuple[int, int]] = None
                is_pinched = False
                
                if hands:
                    # Choose first hand on screen to act as the primary mouse pointer
                    primary_hand = hands[0]
                    
                    # Cursor maps to Index Finger Tip (ID 8)
                    index_tip = next((lm for lm in primary_hand.landmarks if lm.id == 8), None)
                    thumb_tip = next((lm for lm in primary_hand.landmarks if lm.id == 4), None)
                    
                    if index_tip and thumb_tip:
                        cursor_pos = (index_tip.px, index_tip.py)
                        
                        # Calculate distance between Index Tip and Thumb Tip to detect Pinch (Click)
                        dist = math.hypot(index_tip.px - thumb_tip.px, index_tip.py - thumb_tip.py)
                        
                        # Threshold < 35 pixels represents a Pinch (Click gesture)
                        if dist < 32:
                            is_pinched = True
                        else:
                            pinch_released = True  # User opened their hand, ready to click again

                # D. Retrieve thread-safe copy of Docker container data
                with containers_lock:
                    local_containers = list(shared_containers)

                # E. Draw background radar, hand diagnostics, and HUD layouts
                stage_started = profiler.start_stage()
                ui.draw_radar_hud(frame)
                ui.draw_hud_panel(frame)
                ui.draw_hand_tracking(frame, hands, is_pinched)
                ui.draw_container_list(frame, local_containers, cursor_pos)

                # G. Draw and process interactive Cursor Actions
                if cursor_pos:
                    cx, cy = cursor_pos
                    ui.draw_cursor(frame, cx, cy, is_pinched)
                    
                    # Collision and Trigger click check
                    if is_pinched and pinch_released:
                        pinch_released = False  # Lock until they open their pinch
                        current_time = time.monotonic()
                        if current_time - last_click_time >= click_debounce_interval:
                            hit = ui.detect_button_click(cx, cy)
                            if hit:
                                action, container_id = hit
                                print(f"[GESTURE] Triggered '{action}' on container ID: {container_id}")
                                
                                # Visual indicator of triggering action
                                cv2.putText(frame, "TRIGGERING ACTION...", (cx - 50, cy - 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2)
                                
                                # Execute Docker action in a non-blocking way
                                if action == "START":
                                    threading.Thread(target=docker_service.start_container, args=(container_id,), daemon=True).start()
                                elif action == "STOP":
                                    threading.Thread(target=docker_service.stop_container, args=(container_id,), daemon=True).start()
                                elif action == "RESTART":
                                    threading.Thread(target=docker_service.restart_container, args=(container_id,), daemon=True).start()
                                
                                # Update timestamp
                                last_click_time = current_time
                profiler.record("ui_render", stage_started)

                # H. Show final BGR rendered frame
                stage_started = profiler.start_stage()
                cv2.imshow("Touchless Docker Dashboard", frame)
                
                # Check for keyboard escape (Exit on pressing 'q')
                pressed_key = cv2.waitKey(1) & 0xFF
                profiler.record("display", stage_started)
                profiler.complete_frame()
                if pressed_key == ord('q'):
                    running = False
                    break

    except CameraConnectionError as cam_err:
        print(f"\n[CRITICAL ERROR] Camera Initialization Failed: {cam_err}")
    except Exception as general_err:
        print(f"\n[CRITICAL ERROR] Application crashed unexpectedly: {general_err}")
    finally:
        # Gracefully release all locks and close threads
        running = False
        stats_thread.join(timeout=1.5)
        cv2.destroyAllWindows()
        print("\n[INFO] Touchless Docker Dashboard terminated gracefully. All resources released.")


if __name__ == "__main__":
    main()
