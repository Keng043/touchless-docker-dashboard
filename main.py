"""PROJECT GHOST — live hand-vision and gesture diagnostics."""

import math
import os
import time
import webbrowser
from collections import defaultdict, deque
from pathlib import Path

import cv2

from gesture_classifier import classify_detection
from gesture_wheel import COMMANDS, GestureWheel
from hand_tracker import HandTracker, TrackerError

WINDOW_NAME = "PROJECT GHOST | FLOATING GESTURE UI"
MUSIC_URL = "https://youtu.be/hqCMb8Vr1Ss?si=f9VvH1pJmNsY7qQf"
MUSIC_TITLE = "รอ SIS (TER)"
ACCENT = (105, 226, 157)       # BGR
ACCENT_DIM = (61, 137, 103)
TEXT = (232, 239, 235)
MUTED = (143, 160, 151)
PANEL = (13, 22, 19)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def draw_panel(frame, rect, fill=PANEL, alpha=0.78, border=ACCENT_DIM):
    """Draw a subtle translucent control-room panel."""
    x1, y1, x2, y2 = rect
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (min(x1 + 34, x2), y1), ACCENT, 2, cv2.LINE_AA)


def put_text(frame, text, position, scale=0.55, color=TEXT, thickness=1):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_frame_guides(frame):
    """Minimal corner brackets and a central alignment tick."""
    height, width = frame.shape[:2]
    margin, length = 18, 28
    corners = [
        (margin, margin, 1, 1), (width - margin, margin, -1, 1),
        (margin, height - margin, 1, -1), (width - margin, height - margin, -1, -1),
    ]
    for x, y, sx, sy in corners:
        cv2.line(frame, (x, y), (x + sx * length, y), ACCENT_DIM, 1, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + sy * length), ACCENT_DIM, 1, cv2.LINE_AA)
    cv2.line(frame, (width // 2 - 9, height // 2), (width // 2 + 9, height // 2), (90, 112, 100), 1, cv2.LINE_AA)
    cv2.line(frame, (width // 2, height // 2 - 9), (width // 2, height // 2 + 9), (90, 112, 100), 1, cv2.LINE_AA)


def draw_hand(frame, hand, pose, music_active=False):
    """Render one detected hand, its skeleton, and its current pose label."""
    points = {landmark.id: (landmark.px, landmark.py) for landmark in hand.landmarks}
    for start, end in HAND_CONNECTIONS:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], ACCENT, 2, cv2.LINE_AA)
    for landmark_id, (x, y) in points.items():
        radius = 5 if landmark_id in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, (x, y), radius + 2, (22, 48, 36), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, TEXT if landmark_id in (4, 8) else ACCENT, -1, cv2.LINE_AA)

    if music_active:
        for tip_id in (8, 12, 16, 20):
            if tip_id in points:
                tx, ty = points[tip_id]
                put_text(frame, "♪", (max(8, tx - 5), max(20, ty - 10)), 0.72, ACCENT, 2)
    wrist = points.get(0, (24, 80))
    label = f"{hand.label.upper()}  /  {pose.upper()}  /  {hand.confidence:.0%}"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    x = max(10, min(wrist[0] - 18, frame.shape[1] - text_w - 16))
    y = max(text_h + 12, wrist[1] - 20)
    draw_panel(frame, (x - 8, y - text_h - 7, x + text_w + 8, y + 7), alpha=0.86)
    put_text(frame, label, (x, y), 0.52, ACCENT, 1)


def hand_pointing_direction(hand):
    """Infer one of four directions from the index-finger vector."""
    points = {point.id: point for point in hand.landmarks}
    if 0 not in points or 8 not in points:
        return None
    dx = points[8].x - points[0].x
    dy = points[8].y - points[0].y
    if math.hypot(dx, dy) < 0.045:
        return None
    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


def palm_anchor(hand):
    """Return a stable palm-center point in camera pixels."""
    points = {point.id: point for point in hand.landmarks}
    ids = (0, 5, 9, 13, 17)
    available = [points[i] for i in ids if i in points]
    if not available:
        return None
    return (int(sum(p.px for p in available) / len(available)),
            int(sum(p.py for p in available) / len(available)))


def floating_menu_items(frame_shape, anchor):
    """Place two numbered launcher rows above the palm, clamped to the frame."""
    height, width = frame_shape[:2]
    ax, ay = anchor
    item_w, item_h, gap = 224, 46, 8
    menu_w, menu_h = item_w, item_h * len(COMMANDS) + gap * (len(COMMANDS) - 1)
    left = max(8, min(width - menu_w - 8, ax - menu_w // 2))
    top = ay - menu_h - 30
    if top < 8:
        top = min(height - menu_h - 8, ay + 38)
    top = max(8, min(height - menu_h - 8, top))
    return [
        (command, (left, top + i * (item_h + gap),
                   left + item_w, top + i * (item_h + gap) + item_h))
        for i, command in enumerate(COMMANDS)
    ]


def draw_command_wheel(frame, wheel_update, anchor, controller_point=None, music_active=False):
    """Draw a palm-anchored launcher and safe music status."""
    height, width = frame.shape[:2]
    if anchor is None:
        return
    ax, ay = anchor
    items = floating_menu_items(frame.shape, anchor)
    menu_top = min(rect[1] for _, rect in items)
    menu_bottom = max(rect[3] for _, rect in items)
    cv2.line(frame, (ax, menu_bottom if menu_bottom < ay else menu_top),
             (ax, ay - 8 if menu_bottom < ay else ay + 8), ACCENT_DIM, 1, cv2.LINE_AA)
    cv2.circle(frame, (ax, ay), 7, ACCENT, 2, cv2.LINE_AA)

    for index, (command, (x1, y1, x2, y2)) in enumerate(items):
        active = command == wheel_update.selected
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 25, 19), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT if active else ACCENT_DIM,
                      2 if active else 1, cv2.LINE_AA)
        label = "1  MUSIC" if index == 0 else "2  SYSTEM 2"
        put_text(frame, label, (x1 + 16, y1 + 29), 0.52, ACCENT if active else TEXT, 1)

    if music_active:
        # Show track metadata near the fingertips, without reproducing copyrighted lyrics.
        finger_tip = (ax, max(16, ay - 20))
        put_text(frame, "♪  รอ SIS (TER)", finger_tip, 0.50, ACCENT, 1)
        put_text(frame, "MUSIC REQUESTED  /  LYRICS NOT LOADED", (8, height - 18), 0.40, TEXT, 1)
    else:
        hint = "1 FINGER: MUSIC   /   2 FINGERS: SYSTEM 2   /   FIST: CONFIRM"
        (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        hx = max(8, min(width - tw - 8, ax - tw // 2))
        hy = min(height - 12, max(18, menu_bottom + 22))
        put_text(frame, hint, (hx, hy), 0.38, TEXT, 1)
    put_text(frame, wheel_update.status[:64], (8, height - 38), 0.40, ACCENT, 1)


def draw_hud(frame, poses, fps, elapsed, frame_number, wheel_update=None, menu_anchor=None, music_active=False):
    """Draw a responsive GHOST telemetry overlay without obscuring the center."""
    height, width = frame.shape[:2]
    margin = 30 if width >= 900 else 12
    header_h = 68 if width >= 900 else 54
    draw_panel(frame, (margin, margin, width - margin, margin + header_h), alpha=0.84)
    put_text(frame, "G H O S T", (margin + 18, margin + 27), 0.78, ACCENT, 2)
    put_text(frame, "GESTURE LAB  /  COMPUTER VISION", (margin + 20, margin + header_h - 12), 0.40, MUTED, 1)
    right_text = f"VISION ACTIVE   |   {fps:04.1f} FPS"
    (rw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    put_text(frame, right_text, (width - margin - rw - 18, margin + 29), 0.48, TEXT, 1)
    put_text(frame, f"SESSION {elapsed:05.1f}s  /  FRAME {frame_number:06d}",
             (width - margin - rw - 18, margin + header_h - 12), 0.38, MUTED, 1)

    card_w, card_h = min(330, width - margin * 2), 40 + 28 * max(1, len(poses))
    bottom = height - margin
    left = margin
    draw_panel(frame, (left, bottom - card_h, left + card_w, bottom), alpha=0.84)
    put_text(frame, "DETECTED HANDS", (left + 14, bottom - card_h + 24), 0.45, MUTED, 1)
    if not poses:
        put_text(frame, "NO HAND SIGNAL", (left + 14, bottom - 16), 0.57, TEXT, 1)
    else:
        for index, (label, pose, confidence) in enumerate(poses):
            y = bottom - card_h + 51 + index * 28
            put_text(frame, f"{label.upper():<5}  {pose.upper()}", (left + 14, y), 0.53, ACCENT, 1)
            confidence_text = f"{confidence:.0%}"
            (tw, _), _ = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            put_text(frame, confidence_text, (left + card_w - tw - 14, y), 0.48, TEXT, 1)

    if wheel_update and wheel_update.is_open:
        draw_command_wheel(frame, wheel_update, menu_anchor, music_active=music_active)
    else:
        hint = "OPEN PALM  /  ACTIVATE COMMAND WHEEL"
        put_text(frame, hint, (margin + 8, height - margin - 15), 0.42, MUTED, 1)

    legend = "Q / ESC  EXIT"
    (tw, th), _ = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)
    lx, ly = width - margin - tw - 22, height - margin - th - 15
    draw_panel(frame, (lx, ly - 10, width - margin, height - margin), alpha=0.76)
    put_text(frame, legend, (lx + 11, height - margin - 12), 0.43, MUTED, 1)


def main() -> None:
    model_path = Path(__file__).resolve().parent / "hand_landmarker.task"
    camera_index = int(os.getenv("GHOST_CAMERA_INDEX", "1"))
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened() and camera_index != 0:
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Could not open a webcam. Check the camera connection or set GHOST_CAMERA_INDEX."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("PROJECT GHOST | Gesture Lab started. Press 'q' or ESC in the window to quit.")

    frame_times = deque(maxlen=30)
    pose_history = defaultdict(lambda: deque(maxlen=5))
    wheel = GestureWheel()
    started_at = time.monotonic()
    frame_number = 0
    last_timestamp_ms = 0
    last_wheel_update = wheel.update({})
    smoothed_menu_anchor = None
    music_active = False
    music_opened = False

    try:
        with HandTracker(model_path=str(model_path), num_hands=2) as tracker:
            while True:
                success, frame = camera.read()
                if not success or frame is None:
                    print("Camera frame could not be read; stopping.")
                    break

                frame = cv2.flip(frame, 1)
                frame_number += 1
                timestamp_ms = max(last_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
                last_timestamp_ms = timestamp_ms
                try:
                    hands = tracker.process_frame(frame, timestamp_ms)
                except TrackerError as error:
                    print(f"Hand detection error: {error}")
                    hands = []

                active_keys = set()
                poses = []
                stable_hands = {}
                for hand in hands:
                    key = hand.label
                    active_keys.add(key)
                    current_pose = classify_detection(hand)
                    history = pose_history[key]
                    history.append(current_pose)
                    stable_pose = max(set(history), key=history.count)
                    draw_hand(frame, hand, stable_pose, music_active)
                    poses.append((hand.label, stable_pose, hand.confidence))
                    stable_hands[hand.label] = (hand, stable_pose)
                for key in list(pose_history):
                    if key not in active_keys:
                        pose_history[key].clear()

                # Anchor the menu to the activator palm, not the screen center.
                current_anchor = None
                if wheel.is_open and wheel.activator_hand in stable_hands:
                    current_anchor = palm_anchor(stable_hands[wheel.activator_hand][0])
                if current_anchor is not None:
                    if smoothed_menu_anchor is None:
                        smoothed_menu_anchor = current_anchor
                    else:
                        smoothed_menu_anchor = (
                            int(smoothed_menu_anchor[0] * 0.65 + current_anchor[0] * 0.35),
                            int(smoothed_menu_anchor[1] * 0.65 + current_anchor[1] * 0.35),
                        )
                elif not wheel.is_open:
                    smoothed_menu_anchor = None

                wheel_inputs = {
                    label: (stable_pose, None)
                    for label, (_hand, stable_pose) in stable_hands.items()
                }
                last_wheel_update = wheel.update(wheel_inputs)
                # If the open-palm hand activated this frame, attach the menu immediately.
                if last_wheel_update.is_open:
                    active_hand = stable_hands.get(last_wheel_update.activator_hand)
                    if active_hand:
                        current_anchor = palm_anchor(active_hand[0])
                        if smoothed_menu_anchor is None:
                            smoothed_menu_anchor = current_anchor
                        elif current_anchor is not None:
                            smoothed_menu_anchor = (
                                int(smoothed_menu_anchor[0] * 0.65 + current_anchor[0] * 0.35),
                                int(smoothed_menu_anchor[1] * 0.65 + current_anchor[1] * 0.35),
                            )
                else:
                    smoothed_menu_anchor = None
                if last_wheel_update.confirmed == "MUSIC" and not music_opened:
                    music_active = True
                    music_opened = webbrowser.open(MUSIC_URL)
                    print(f"[MUSIC] Opening {MUSIC_TITLE}: {MUSIC_URL}")
                elif last_wheel_update.confirmed == "SYSTEM 2":
                    print("[SYSTEM 2] Placeholder selected; no action is configured.")
                if last_wheel_update.shutdown:
                    music_active = False
                    music_opened = False
                    print("[GHOST] Gesture UI deactivated by two fists.")

                now = time.monotonic()
                frame_times.append(now)
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 and frame_times[-1] > frame_times[0] else 0.0
                draw_frame_guides(frame)
                draw_hud(frame, poses, fps, now - started_at, frame_number, last_wheel_update,
                         smoothed_menu_anchor, music_active)
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("PROJECT GHOST | Camera and hand tracker stopped.")


if __name__ == "__main__":
    main()
