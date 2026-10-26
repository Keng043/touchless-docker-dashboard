"""
Module: ui_manager
Project: Touchless Docker Dashboard
Author: Python Software Architect & DevOps Engineer
Description: OpenCV HUD rendering and low-overhead gesture interaction UI.
"""

from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from docker_service import DockerContainer
from hand_tracker import HandDetection, HandLandmark


class UIManager:
    """
    Manages the UI layer of the Touchless Docker Dashboard.
    Responsible for rendering the cyberpunk HUD, drawing the virtual hand cursor,
    and handling button hover/click detection.
    """

    def __init__(self, width: int = 1280, height: int = 720) -> None:
        """
        Initializes the UIManager with screen dimensions and color palette.
        """
        self.width: int = width
        self.height: int = height

        # Color Palette (BGR format)
        self.COLOR_CYAN: Tuple[int, int, int] = (255, 243, 0)       # Cyan
        self.COLOR_NEON_GREEN: Tuple[int, int, int] = (0, 255, 128) # Neon Green
        self.COLOR_RED: Tuple[int, int, int] = (0, 0, 255)          # Bright Red
        self.COLOR_WHITE: Tuple[int, int, int] = (255, 255, 255)    # White
        self.COLOR_GRAY: Tuple[int, int, int] = (100, 100, 100)      # Gray
        self.COLOR_BG_DARK: Tuple[int, int, int] = (20, 15, 15)      # Dark Tech BG
        
        # Define button bounds in memory for collision checking
        # Format: { button_name: (x_min, y_min, x_max, y_max, container_id) }
        self.interactive_buttons: Dict[str, Tuple[int, int, int, int, str]] = {}


    def draw_hud_panel(self, frame: np.ndarray) -> None:
        """
        Draws the sci-fi translucent background panel and bounding frames.
        """
        # Blend only the sidebar region. Copying and blending the full 1280x720
        # frame here adds avoidable memory bandwidth cost on every rendered frame.
        x1, y1, x2, y2 = 20, 20, min(540, self.width - 1), min(700, self.height - 1)
        panel = frame[y1:y2 + 1, x1:x2 + 1]
        overlay = panel.copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), (30, 20, 20), -1)
        # Apply transparency to the panel only; pixels outside remain untouched.
        cv2.addWeighted(overlay, 0.75, panel, 0.25, 0, panel)
        
        # Draw glowing neon borders
        cv2.rectangle(frame, (20, 20), (540, 700), self.COLOR_CYAN, 2)
        cv2.rectangle(frame, (18, 18), (542, 702), self.COLOR_CYAN, 1) # Double line effect
        
        # Draw tech corner decorations
        corners = [
            ((15, 15), (40, 15), (15, 40)),      # Top-Left
            ((545, 15), (520, 15), (545, 40)),    # Top-Right
            ((15, 705), (40, 705), (15, 680)),    # Bottom-Left
            ((545, 705), (520, 705), (545, 680))   # Bottom-Right
        ]
        for c_pts in corners:
            cv2.line(frame, c_pts[0], c_pts[1], self.COLOR_NEON_GREEN, 4)
            cv2.line(frame, c_pts[0], c_pts[2], self.COLOR_NEON_GREEN, 4)

        # Header Title
        cv2.putText(frame, "TOUCHLESS DOCKER CONTROLLER v1.0", (40, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_NEON_GREEN, 2, cv2.LINE_AA)
        cv2.line(frame, (30, 80), (530, 80), self.COLOR_CYAN, 1)

    def draw_container_list(
        self, 
        frame: np.ndarray, 
        containers: List[DockerContainer], 
        cursor_pos: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Renders the Docker containers with their status, metrics progress bars, and action buttons.
        Registers button bounding coordinates for gesture clicks.
        """
        self.interactive_buttons.clear() # Clear bounds from last frame
        
        if not containers:
            cv2.putText(frame, "NO ACTIVE CONTAINERS FOUND", (60, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_RED, 2)
            return

        cx, cy = cursor_pos if cursor_pos else (-1, -1)
        start_y = 100
        row_height = 110

        for idx, container in enumerate(containers[:5]): # Display top 5 containers
            y = start_y + (idx * row_height)
            
            # 1. Indicator Circle (Running = Green, Stopped = Red)
            status_color = self.COLOR_NEON_GREEN if container.status == "running" else self.COLOR_RED
            cv2.circle(frame, (50, y + 25), 8, status_color, -1)
            # Outer pulsing ring
            cv2.circle(frame, (50, y + 25), 12, status_color, 1)

            # 2. Container Name & Image
            name_text = container.name[:22]
            cv2.putText(frame, name_text, (80, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_WHITE, 2, cv2.LINE_AA)
            image_text = f"Image: {container.image[:26]}"
            cv2.putText(frame, image_text, (80, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_GRAY, 1, cv2.LINE_AA)

            # 3. CPU Progress Bar
            cv2.putText(frame, f"CPU: {container.cpu_percentage:5}%", (80, y + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_WHITE, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (180, y + 55), (320, y + 65), (50, 50, 50), -1) # BG bar
            cpu_fill_width = int((container.cpu_percentage / 100.0) * 140)
            cpu_fill_width = max(0, min(140, cpu_fill_width))
            cv2.rectangle(frame, (180, y + 55), (180 + cpu_fill_width, y + 65), self.COLOR_NEON_GREEN, -1)

            # 4. Memory Progress Bar
            mem_pct = (container.memory_usage_mb / container.memory_limit_mb) if container.memory_limit_mb > 0 else 0
            cv2.putText(frame, f"MEM: {container.memory_usage_mb:.1f}M / {container.memory_limit_mb:.0f}M", (80, y + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WHITE, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (180, y + 80), (320, y + 90), (50, 50, 50), -1) # BG bar
            mem_fill_width = int(mem_pct * 140)
            mem_fill_width = max(0, min(140, mem_fill_width))
            cv2.rectangle(frame, (180, y + 80), (180 + mem_fill_width, y + 90), self.COLOR_CYAN, -1)

            # 5. Interactive Action Buttons
            # Action Button (START / STOP)
            action_btn_x_min, action_btn_x_max = 350, 430
            action_btn_y_min, action_btn_y_max = y + 15, y + 45
            is_hover_action = (action_btn_x_min <= cx <= action_btn_x_max) and (action_btn_y_min <= cy <= action_btn_y_max)
            
            action_label = "STOP" if container.status == "running" else "START"
            action_color = self.COLOR_RED if container.status == "running" else self.COLOR_NEON_GREEN
            
            # Hover highlight
            btn_color = self.COLOR_WHITE if is_hover_action else action_color
            cv2.rectangle(frame, (action_btn_x_min, action_btn_y_min), (action_btn_x_max, action_btn_y_max), btn_color, 2)
            cv2.putText(frame, action_label, (action_btn_x_min + 12, action_btn_y_min + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, btn_color, 2)
            
            # Register button
            self.interactive_buttons[f"{action_label}_{container.id}"] = (
                action_btn_x_min, action_btn_y_min, action_btn_x_max, action_btn_y_max, container.id
            )

            # Restart Button (RESTART)
            re_btn_x_min, re_btn_x_max = 445, 525
            re_btn_y_min, re_btn_y_max = y + 15, y + 45
            is_hover_re = (re_btn_x_min <= cx <= re_btn_x_max) and (re_btn_y_min <= cy <= re_btn_y_max)
            
            re_color = self.COLOR_CYAN
            re_btn_color = self.COLOR_WHITE if is_hover_re else re_color
            cv2.rectangle(frame, (re_btn_x_min, re_btn_y_min), (re_btn_x_max, re_btn_y_max), re_btn_color, 2)
            cv2.putText(frame, "RESTART", (re_btn_x_min + 5, re_btn_y_min + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, re_btn_color, 2)
            
            # Register button
            self.interactive_buttons[f"RESTART_{container.id}"] = (
                re_btn_x_min, re_btn_y_min, re_btn_x_max, re_btn_y_max, container.id
            )

            # Draw divider
            cv2.line(frame, (30, y + 105), (530, y + 105), (40, 40, 40), 1)

    def draw_hand_tracking(self, frame: np.ndarray, hands: List[HandDetection], is_pinched: bool) -> None:
        """Draw visible hand landmarks and a live detection status for gesture debugging."""
        status = f"HANDS DETECTED: {len(hands)}"
        status_color = self.COLOR_NEON_GREEN if hands else self.COLOR_RED
        cv2.putText(frame, status, (580, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    status_color, 2, cv2.LINE_AA)
        pinch_text = "PINCH: ACTIVE" if is_pinched else "PINCH: READY"
        pinch_color = (0, 215, 255) if is_pinched else self.COLOR_WHITE
        cv2.putText(frame, pinch_text, (580, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    pinch_color, 2, cv2.LINE_AA)

        connections = (
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),(0,17)
        )
        for hand in hands:
            points = {lm.id: (lm.px, lm.py) for lm in hand.landmarks}
            for a, b in connections:
                if a in points and b in points:
                    cv2.line(frame, points[a], points[b], self.COLOR_CYAN, 2, cv2.LINE_AA)
            for lm in hand.landmarks:
                cv2.circle(frame, (lm.px, lm.py), 3, self.COLOR_NEON_GREEN, -1)
            if 8 in points:
                cv2.circle(frame, points[8], 10, self.COLOR_WHITE, 2)

    def draw_cursor(self, frame: np.ndarray, cx: int, cy: int, is_pinched: bool) -> None:
        """
        Draws an interactive sci-fi tracking cursor at the hand pointer coordinates.
        The cursor changes style and color when the pinch (click) is triggered.
        """
        cursor_color = self.COLOR_NEON_GREEN if not is_pinched else (0, 215, 255) # Green vs Gold
        cursor_radius = 12 if not is_pinched else 6
        
        # Center point
        cv2.circle(frame, (cx, cy), 4, cursor_color, -1)
        # Outer ring
        cv2.circle(frame, (cx, cy), cursor_radius, cursor_color, 2)
        
        # Crosshair lines
        offset = 18
        cv2.line(frame, (cx - offset, cy), (cx - 8, cy), cursor_color, 1)
        cv2.line(frame, (cx + 8, cy), (cx + offset, cy), cursor_color, 1)
        cv2.line(frame, (cx, cy - offset), (cx, cy - 8), cursor_color, 1)
        cv2.line(frame, (cx, cy + 8), (cx, cy + offset), cursor_color, 1)


    def detect_button_click(self, cx: int, cy: int) -> Optional[Tuple[str, str]]:
        """
        Checks if the virtual cursor coordinates lie inside any registered button.

        Args:
            cx (int): Cursor X coordinate.
            cy (int): Cursor Y coordinate.

        Returns:
            Optional[Tuple[str, str]]: Tuple of (action_type, container_id) if hit, None otherwise.
                                      e.g., ("START", "a1b2c3d4")
        """
        for btn_name, (x_min, y_min, x_max, y_max, container_id) in self.interactive_buttons.items():
            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                action = btn_name.split("_")[0] # Extract action from registered key
                return action, container_id
        return None

    def draw_radar_hud(self, frame: np.ndarray) -> None:
        """
        Draws decorative sci-fi background details in the right section.
        """
        # Decorative Grid Lines
        grid_color = (30, 25, 25)
        for i in range(560, 1260, 40):
            cv2.line(frame, (i, 20), (i, 700), grid_color, 1)
        for j in range(20, 700, 40):
            cv2.line(frame, (560, j), (1260, j), grid_color, 1)

        # Radar circle
        radar_center = (910, 360)
        cv2.circle(frame, radar_center, 180, (40, 30, 30), 1)
        cv2.circle(frame, radar_center, 100, (40, 30, 30), 1)
        cv2.line(frame, (910 - 200, 360), (910 + 200, 360), (40, 30, 30), 1)
        cv2.line(frame, (910, 360 - 200), (910, 360 + 200), (40, 30, 30), 1)

        cv2.putText(frame, "RADAR GESTURE SENSOR: ONLINE", (580, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, "SYSTEM MONITORS: LIVE", (580, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_CYAN, 1, cv2.LINE_AA)
