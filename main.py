"""
Main Orchestrator: main.py
Project: Touchless Docker Dashboard
Author: Python Software Architect & DevOps Engineer
Description: Integrates CameraStreamer, HandTracker, DockerService, and UIManager.
             Implements asynchronous background Docker stats fetching to prevent lag,
             gesture cursor mapping, pinch-to-click detection, and debounce control.
"""

import sys
import threading
import time
import math
from typing import List, Optional, Tuple
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


def main() -> None:
    global shared_containers, running
    
    # 1. Initialize UIManager (1280x720 HD display target)
    width, height = 1280, 720
    ui = UIManager(width=width, height=height)

    # 2. Instantiate and establish Docker Service
    print("[INFO] Setting up Docker Controller Service...")
    docker_service = get_docker_service()
    
    # Initialize shared list with an initial fetch before launching background thread
    with containers_lock:
        shared_containers = docker_service.list_containers()

    # Launch async stats updater thread
    stats_thread = threading.Thread(
        target=docker_stats_fetcher, 
        args=(docker_service,), 
        name="DockerStatsUpdater", 
        daemon=True
    )
    stats_thread.start()

    # 3. Instantiate CameraStreamer and HandTracker
    # Index 0 is standard webcam.
    camera = get_camera_streamer(src=0, width=width, height=height)
    tracker = HandTracker(model_path="hand_landmarker.task", num_hands=2)

    # State variables for Gesture Control and UX Debouncing
    last_click_time = 0.0
    click_debounce_interval = 1.5  # 1.5 seconds delay between clicks to prevent multiple registers
    pinch_released = True         # Must release pinch before clicking again

    print("\n" + "="*50)
    print("      TOUCHLESS DOCKER CONTROLLER STARTED")
    print("      Press 'q' inside video window to EXIT")
    print("="*50 + "\n")

    try:
        # Open camera and hand tracker using Context Managers to guarantee resource release
        with camera as streamer, tracker as hand_detector:
            while running:
                # A. Read asynchronous webcam frame
                success, frame = streamer.read()
                if not success or frame is None:
                    time.sleep(0.01)
                    continue

                # OpenCV default is BGR. Let's resize it to exactly fit our HUD dimensions
                frame = cv2.resize(frame, (width, height))
                
                # B. Perform hand landmark extraction
                timestamp_ms = int(time.time() * 1000)
                try:
                    hands = hand_detector.process_frame(frame, timestamp_ms)
                except TrackerError as tracker_err:
                    print(f"[WARNING] Hand tracker error: {tracker_err}")
                    hands = []

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

                # E. Draw background radar and HUD layouts
                ui.draw_radar_hud(frame)
                ui.draw_hud_panel(frame)
                ui.draw_container_list(frame, local_containers, cursor_pos)

                # F. Handle Portal (If 2 hands are visible, open the magical neon portal)
                if len(hands) >= 2:
                    ui.draw_magic_portal(frame, hands)

                # G. Draw and process interactive Cursor Actions
                if cursor_pos:
                    cx, cy = cursor_pos
                    ui.draw_cursor(frame, cx, cy, is_pinched)
                    
                    # Collision and Trigger click check
                    if is_pinched and pinch_released:
                        pinch_released = False  # Lock until they open their pinch
                        current_time = time.time()
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

                # H. Show final BGR rendered frame
                cv2.imshow("Touchless Docker Dashboard", frame)
                
                # Check for keyboard escape (Exit on pressing 'q')
                if cv2.waitKey(1) & 0xFF == ord('q'):
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
