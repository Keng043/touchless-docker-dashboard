"""PROJECT GHOST — live hand-vision and gesture diagnostics."""

import math
import os
import time
import threading
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from custom_gestures import CustomGestureModel, landmark_features
from gesture_classifier import classify_detection
from gesture_wheel import COMMANDS, GestureWheel
from hand_tracker import HandTracker, TrackerError
from local_audio import LocalAudioPlayer
from subtitle_reader import load_srt, subtitle_at

WINDOW_NAME = "PROJECT GHOST"
MUSIC_TITLE = "รอ SIS (TER)"
ACCENT = (245, 218, 145)       # BGR: soft pastel cyan
ACCENT_DIM = (176, 148, 105)
TEXT = (248, 250, 255)
MUTED = (190, 196, 216)
PANEL = (20, 23, 35)
BLUE = (255, 215, 145)          # BGR: gentle cyan glow for fingertips
LAVENDER = (232, 194, 255)
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
    # Blend only the panel region instead of copying the entire HD frame.
    roi = frame[y1:y2 + 1, x1:x2 + 1]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), fill, -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (min(x1 + 34, x2), y1), ACCENT, 2, cv2.LINE_AA)


def put_text(frame, text, position, scale=0.55, color=TEXT, thickness=1):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_unicode_subtitle(frame, text, anchor):
    """Render Thai/Unicode captions near the tracked fingertips using a system font."""
    if not text:
        return
    height, width = frame.shape[:2]
    font_path = Path(os.getenv("GHOST_FONT_PATH", r"C:\Windows\Fonts\LeelawUI.ttf"))
    if not font_path.exists():
        font_path = Path(r"C:\Windows\Fonts\segoeui.ttf")
    try:
        font = ImageFont.truetype(str(font_path), 25 if width >= 900 else 20)
    except OSError:
        print(f"[SUBTITLE] Unicode font unavailable: {font_path}")
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    painter = ImageDraw.Draw(image)
    max_text_width = min(660, width - 36)
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and painter.textlength(candidate, font=font) > max_text_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    # Thai does not always contain spaces, so split oversized lines by characters.
    wrapped = []
    for line in lines or [text]:
        part = ""
        for char in line:
            if part and painter.textlength(part + char, font=font) > max_text_width:
                wrapped.append(part)
                part = char
            else:
                part += char
        if part:
            wrapped.append(part)
    lines = wrapped or [text]

    line_height = 34 if width >= 900 else 29
    box_w = min(max_text_width + 24, int(max(painter.textlength(line, font=font) for line in lines) + 24))
    box_h = line_height * len(lines) + 16
    ax, ay = anchor if anchor is not None else (width // 2, height - 32)
    left = max(8, min(width - box_w - 8, int(ax - box_w // 2)))
    top = int(ay - box_h - 34)
    if top < 8:
        top = min(height - box_h - 8, int(ay + 30))
    top = max(8, min(height - box_h - 8, top))
    painter.rounded_rectangle((left, top, left + box_w, top + box_h), radius=10,
                              fill=(13, 22, 19), outline=(105, 226, 157), width=2)
    for index, line in enumerate(lines):
        painter.text((left + 12, top + 8 + index * line_height), line,
                     font=font, fill=(232, 239, 235), stroke_width=0)
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


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
            cv2.line(frame, points[start], points[end], ACCENT_DIM, 1, cv2.LINE_AA)
    fingertip_ids = (4, 8, 12, 16, 20)
    for landmark_id, (x, y) in points.items():
        if landmark_id in fingertip_ids:
            cv2.circle(frame, (x, y), 6, BLUE, 1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 2, (255, 250, 230), -1, cv2.LINE_AA)
        elif landmark_id in (0, 5, 9, 13, 17):
            cv2.circle(frame, (x, y), 2, LAVENDER, -1, cv2.LINE_AA)

    # Small unobtrusive hand label instead of a heavy panel.
    wrist = points.get(0, (24, 80))
    label = f"{hand.label.upper()}  {pose.upper()}"
    x = max(8, min(wrist[0] - 10, frame.shape[1] - 150))
    y = max(18, wrist[1] - 14)
    put_text(frame, label, (x, y), 0.38, LAVENDER, 1)


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


def fingertip_anchor(hand):
    """Return the average visible fingertip position for the floating caption."""
    tips = {point.id: point for point in hand.landmarks if point.id in (8, 12, 16, 20)}
    if not tips:
        return palm_anchor(hand)
    return (int(sum(point.px for point in tips.values()) / len(tips)),
            int(sum(point.py for point in tips.values()) / len(tips)))


def detect_hand_frame_rect(stable_hands, frame_shape, min_corner_spread=0.0):
    """Build a frame from the thumb/index corner landmarks, not every fingertip."""
    polygon = detect_hand_frame_polygon(stable_hands)
    if polygon is None:
        return None
    height, width = frame_shape[:2]
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    left, right = max(0, min(xs)), min(width - 1, max(xs))
    top, bottom = max(0, min(ys)), min(height - 1, max(ys))
    box_width, box_height = right - left, bottom - top
    if box_width < width * 0.12 or box_height < height * 0.08:
        return None
    if box_width / max(box_height, 1) > 5.0:
        return None
    return (int(left), int(top), int(right), int(bottom))


def detect_hand_frame_polygon(stable_hands):
    """Use thumb and index tips on each hand as the four corners of an air-drawn frame."""
    if len(stable_hands) < 2:
        return None
    hand_corners = []
    for hand, _pose in stable_hands.values():
        points = {point.id: point for point in hand.landmarks}
        if 4 not in points or 8 not in points:
            return None
        thumb, index = points[4], points[8]
        center_x = (thumb.px + index.px) / 2.0
        hand_corners.append((center_x, (int(thumb.px), int(thumb.py)),
                             (int(index.px), int(index.py))))
    hand_corners.sort(key=lambda item: item[0])
    left_hand, right_hand = hand_corners
    return (left_hand[1], right_hand[1], right_hand[2], left_hand[2])


def has_any_finger_pinch(stable_hands, threshold=0.045):
    """True when any two fingertips on either hand pinch together."""
    tip_ids = (4, 8, 12, 16, 20)
    for hand, _pose in stable_hands.values():
        points = {point.id: point for point in hand.landmarks}
        tips = [points[i] for i in tip_ids if i in points]
        for index, first in enumerate(tips):
            for second in tips[index + 1:]:
                if math.hypot(first.x - second.x, first.y - second.y) < threshold:
                    return True
    return False


def smooth_rect(previous, current, weight=0.35):
    """Smooth a valid hand-frame rectangle to keep text stable as hands move."""
    if current is None:
        return None
    if previous is None:
        return tuple(int(value) for value in current)
    return tuple(int(old * (1.0 - weight) + new * weight)
                 for old, new in zip(previous, current))


def draw_index_trail(frame, points):
    """Draw a short, delicate cyan-lavender trail with a soft fingertip glow."""
    if not points:
        return
    total = len(points)
    overlay = frame.copy()
    for index in range(1, total):
        fade = index / max(1, total - 1)
        thickness = 1 if fade < 0.65 else 2
        color = (
            int(210 + 45 * fade),
            int(185 + 30 * fade),
            int(150 + 90 * fade),
        )
        cv2.line(overlay, points[index - 1], points[index], color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
    x, y = points[-1]
    glow = frame.copy()
    cv2.circle(glow, (x, y), 11, BLUE, -1, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.10, frame, 0.90, 0, frame)
    cv2.circle(frame, (x, y), 3, (255, 250, 235), -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 6, BLUE, 1, cv2.LINE_AA)


PORTAL_FILTERS = (
    "Mono", "Dual-Tone", "Pixelate", "Invert", "Sepia", "Blur",
    "Thermal", "Sketch", "Glitch", "Neon", "Galaxy",
)


def apply_portal_filter(frame, polygon, filter_name="Mono"):
    """Apply one OpenCV visual filter strictly inside the detected hand quadrilateral."""
    if polygon is None or len(polygon) != 4:
        return frame
    height, width = frame.shape[:2]
    points = np.asarray(polygon, dtype=np.int32)
    x1 = max(0, int(points[:, 0].min()))
    y1 = max(0, int(points[:, 1].min()))
    x2 = min(width, int(points[:, 0].max()) + 1)
    y2 = min(height, int(points[:, 1].max()) + 1)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return frame

    roi = frame[y1:y2, x1:x2]
    local_points = points - np.array([x1, y1], dtype=np.int32)
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local_points], 255, lineType=cv2.LINE_AA)
    if not np.any(mask):
        return frame

    name = filter_name if filter_name in PORTAL_FILTERS else PORTAL_FILTERS[0]
    if name == "Mono":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        filtered = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif name == "Dual-Tone":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dark = np.array([132, 48, 38], dtype=np.uint8)
        light = np.array([245, 218, 174], dtype=np.uint8)
        amount = gray.astype(np.float32)[:, :, None] / 255.0
        filtered = np.clip(dark * (1.0 - amount) + light * amount, 0, 255).astype(np.uint8)
    elif name == "Pixelate":
        small = cv2.resize(roi, (max(1, roi.shape[1] // 14), max(1, roi.shape[0] // 14)),
                           interpolation=cv2.INTER_LINEAR)
        filtered = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    elif name == "Invert":
        filtered = cv2.bitwise_not(roi)
    elif name == "Sepia":
        matrix = np.array([[0.131, 0.534, 0.272],
                           [0.168, 0.686, 0.349],
                           [0.189, 0.769, 0.393]], dtype=np.float32)
        filtered = cv2.transform(roi, matrix)
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    elif name == "Blur":
        filtered = cv2.GaussianBlur(roi, (0, 0), sigmaX=8, sigmaY=8)
    elif name == "Thermal":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        filtered = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    elif name == "Sketch":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        filtered_gray = cv2.divide(gray, 255 - cv2.GaussianBlur(inverted, (0, 0), 9), scale=256)
        filtered = cv2.cvtColor(filtered_gray, cv2.COLOR_GRAY2BGR)
    elif name == "Glitch":
        filtered = roi.copy()
        shift = max(2, roi.shape[1] // 35)
        filtered[:, :, 2] = np.roll(roi[:, :, 2], shift, axis=1)
        filtered[:, :, 0] = np.roll(roi[:, :, 0], -shift, axis=1)
        stripe_step = max(8, roi.shape[0] // 12)
        filtered[::stripe_step, :, :] = np.clip(
            filtered[::stripe_step, :, :].astype(np.int16) + 24, 0, 255).astype(np.uint8)
    elif name == "Neon":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 55, 130)
        glow = cv2.GaussianBlur(edges, (0, 0), 3)
        filtered = np.zeros_like(roi)
        filtered[:, :, 0] = glow
        filtered[:, :, 1] = np.clip(glow.astype(np.int16) // 2 + edges, 0, 255)
        filtered[:, :, 2] = np.clip(glow.astype(np.int16) + edges, 0, 255)
        filtered = cv2.addWeighted(filtered, 0.9, roi, 0.18, 0)
    else:  # Galaxy: indigo/violet color grade with a sparse star field.
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.uint16) + 105) % 180).astype(np.uint8)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) + 28, 0, 255).astype(np.uint8)
        galaxy = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        galaxy_tint = np.empty_like(galaxy)
        galaxy_tint[:] = (92, 34, 75)
        galaxy = cv2.addWeighted(galaxy, 0.72, galaxy_tint, 0.28, 0)
        # Fixed, sparse star points keep the effect stable instead of flickering each frame.
        step_x, step_y = max(13, roi.shape[1] // 12), max(13, roi.shape[0] // 8)
        for sy in range(step_y // 2, roi.shape[0], step_y):
            for sx in range(step_x // 3, roi.shape[1], step_x):
                cv2.circle(galaxy, (sx, sy), 1, (235, 230, 255), -1, cv2.LINE_AA)
        filtered = galaxy

    alpha = mask.astype(np.float32)[:, :, None] / 255.0
    roi[:] = np.clip(roi.astype(np.float32) * (1.0 - alpha) +
                     filtered.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return frame


def draw_hand_frame_energy(frame, rect, tick, polygon=None):
    """Draw the translucent quadrilateral hand frame and subtle cyan moving accents."""
    if rect is None:
        return
    left, top, right, bottom = rect
    if right - left < 30 or bottom - top < 25:
        return
    if polygon and len(polygon) == 4:
        pts = np.array(polygon, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (210, 198, 175), lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.13, frame, 0.87, 0, frame)
        cv2.polylines(frame, [pts], True, (225, 218, 195), 1, cv2.LINE_AA)
        corners = polygon
    else:
        corners = ((left, top), (right, top), (left, bottom), (right, bottom))
    for x, y in corners:
        cv2.circle(frame, (x, y), 8, BLUE, 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 2, (255, 250, 235), -1, cv2.LINE_AA)
    # No moving beams inside the frame; only outline and corner glows.


def draw_subtitle_in_hand_frame(frame, text, rect):
    """Draw subtitle characters centered inside the hand-defined area, with no box."""
    if not text or rect is None:
        return
    left, top, right, bottom = rect
    width, height = right - left, bottom - top
    if width < 20 or height < 20:
        return
    # Leelawadee UI includes Thai glyphs; Segoe UI/Arial may render Thai as tofu squares.
    font_candidates = [
        Path(os.getenv("GHOST_FONT_PATH", r"C:\Windows\Fonts\LeelawUI.ttf")),
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    try:
        font_path = next(path for path in font_candidates if path.exists())
        # Keep the subtitle small and clean while preserving Thai character support.
        font = ImageFont.truetype(str(font_path), max(12, min(18, int(height * 0.15))))
    except OSError:
        return
    crop = frame[top:bottom, left:right]
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    painter = ImageDraw.Draw(image)
    max_width = max(10, width - 20)
    lines, current = [], ""
    for char in text:
        candidate = current + char
        if current and painter.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = char
        else:
            current = candidate
    if current:
        lines.append(current)
    line_height = max(15, int(font.size * 1.18))
    max_lines = max(1, (height - 12) // line_height)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and painter.textlength(lines[-1] + "…", font=font) > max_width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "…"
    total_height = len(lines) * line_height
    y = max(0, (height - total_height) // 2)
    for line in lines:
        line_width = painter.textlength(line, font=font)
        x = max(0, int((width - line_width) / 2))
        # A small dark stroke keeps letters legible without drawing a panel or border.
        painter.text((x, y), line, font=font, fill=(242, 248, 255),
                     stroke_width=1, stroke_fill=(18, 28, 40))
        y += line_height
    frame[top:bottom, left:right] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


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


def draw_command_wheel(frame, wheel_update, anchor, controller_point=None, music_active=False, subtitle_text=None, subtitle_anchor=None):
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
        roi = frame[y1:y2, x1:x2]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1 - 1, y2 - y1 - 1), (12, 25, 19), -1)
        cv2.addWeighted(overlay, 0.88, roi, 0.12, 0, roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT if active else ACCENT_DIM,
                      2 if active else 1, cv2.LINE_AA)
        label = "1  MUSIC" if index == 0 else "2  SYSTEM 2"
        put_text(frame, label, (x1 + 16, y1 + 29), 0.52, ACCENT if active else TEXT, 1)

    # Keep the launcher self-contained; no status text is painted along the bottom.


def draw_hud(frame, poses, fps, elapsed, frame_number, wheel_update=None, menu_anchor=None,
             music_active=False, subtitle_text=None, subtitle_anchor=None,
             subtitle_frame_rect=None, subtitle_frame_polygon=None, program_status="READY",
             portal_filter_name="Mono"):
    """Minimal HUD: only program status and FPS in the top strip."""
    height, width = frame.shape[:2]
    margin = 18 if width >= 900 else 10
    header_h = 48 if width >= 900 else 42
    draw_panel(frame, (margin, margin, width - margin, margin + header_h),
               fill=PANEL, alpha=0.88, border=ACCENT_DIM)

    status = program_status.upper()
    if status == "READY":
        if wheel_update and wheel_update.is_open:
            status = ("MENU ACTIVE / 1 MUSIC" if wheel_update.selected == "MUSIC"
                      else "SYSTEM 2 SELECTED / FIST")
        else:
            status = "OPEN PALM TO ACTIVATE"
    status = status[:32]
    status_color = (80, 165, 255) if any(
        marker in status for marker in ("FAILED", "MISSING", "ERROR")
    ) else ACCENT
    cv2.circle(frame, (margin + 18, margin + header_h // 2), 4, status_color, -1, cv2.LINE_AA)
    put_text(frame, status, (margin + 32, margin + header_h // 2 + 6),
             0.53 if width >= 900 else 0.45, TEXT, 1)
    fps_text = f"FPS  {fps:04.1f}"
    (fps_width, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.53, 1)
    put_text(frame, fps_text, (width - margin - fps_width - 16, margin + header_h // 2 + 6),
             0.53, ACCENT, 1)

    if wheel_update and wheel_update.is_open:
        draw_command_wheel(frame, wheel_update, menu_anchor, music_active=music_active,
                           subtitle_text=subtitle_text, subtitle_anchor=subtitle_anchor)
    if music_active and subtitle_frame_rect is not None:
        draw_hand_frame_energy(frame, subtitle_frame_rect, time.monotonic(), subtitle_frame_polygon)
        left, top, right, bottom = subtitle_frame_rect
        if right - left > 70 and bottom - top > 25:
            put_text(frame, f"PINCH TO CHANGE  /  {portal_filter_name.upper()}",
                     (left + 8, min(bottom - 8, top + 20)), 0.42, ACCENT, 1)
        if subtitle_text:
            draw_subtitle_in_hand_frame(frame, subtitle_text, subtitle_frame_rect)


class LatestFrameReader:
    """Read camera frames in the background and expose only the newest frame."""

    def __init__(self, camera):
        self.camera = camera
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame = None
        self._ok = False
        self._thread = threading.Thread(target=self._capture_loop, name="ghost-camera", daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while not self._stop.is_set():
            ok, frame = self.camera.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
                    self._ok = True
            else:
                # A transient failed read should not discard the most recent good frame.
                # Keep retrying; the consumer decides when repeated failures are fatal.
                time.sleep(0.005)

    def read(self):
        with self._lock:
            if not self._ok or self._frame is None:
                return False, None
            # cv2.flip creates a new array in the consumer, so this reference is safe.
            return True, self._frame

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)


def configure_camera(camera, width=None, height=None):
    """Request a sharp camera mode and return the resolution the driver reports."""
    width = int(width or os.getenv("GHOST_CAMERA_WIDTH", "1280"))
    height = int(height or os.getenv("GHOST_CAMERA_HEIGHT", "720"))

    # MJPG can reduce USB bandwidth compared with raw YUYV at HD resolutions.
    # Drivers are allowed to ignore this request; report the negotiated mode below.
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_width = int(round(camera.get(cv2.CAP_PROP_FRAME_WIDTH)))
    actual_height = int(round(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return actual_width, actual_height


def main() -> None:
    model_path = Path(__file__).resolve().parent / "hand_landmarker.task"
    camera_index = int(os.getenv("GHOST_CAMERA_INDEX", "1"))
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened() and camera_index != 0:
        camera.release()
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            "Could not open a webcam. Check the camera connection or set GHOST_CAMERA_INDEX."
        )

    actual_width, actual_height = configure_camera(camera)
    camera_reader = LatestFrameReader(camera)
    print(f"[CAMERA] Requested {os.getenv('GHOST_CAMERA_WIDTH', '1280')}x{os.getenv('GHOST_CAMERA_HEIGHT', '720')}; driver reports {actual_width}x{actual_height}.")
    if actual_width < 1280 or actual_height < 720:
        print("[CAMERA] WARNING: camera/driver did not accept HD resolution. Try GHOST_CAMERA_INDEX=0 or another camera mode.")
    print("[CAMERA] For 1080p, set GHOST_CAMERA_WIDTH=1920 and GHOST_CAMERA_HEIGHT=1080 before launch.")
    print("PROJECT GHOST | Gesture Lab started. Press 'q' or ESC in the window to quit.")

    frame_times = deque(maxlen=30)
    pose_history = defaultdict(lambda: deque(maxlen=5))
    wheel = GestureWheel()
    started_at = time.monotonic()
    frame_number = 0
    last_timestamp_ms = 0
    last_wheel_update = wheel.update({})
    portal_filter_index = 0
    portal_pinch_was_active = False
    print("[PORTAL] Filter: Mono. Pinch any two fingertips together to change the filter.")
    custom_model_path = Path(__file__).resolve().parent / "data" / "custom_gestures.json"
    try:
        custom_model = CustomGestureModel.load(custom_model_path)
        print(f"[CUSTOM GESTURES] Loaded {len(custom_model.prototypes)} trained gesture classes.")
    except (OSError, ValueError, TypeError) as error:
        custom_model = CustomGestureModel()
        print(f"[CUSTOM GESTURES] Model unavailable; built-in gestures remain active: {error}")
    custom_gesture_latched = None
    smoothed_menu_anchor = None
    smoothed_subtitle_frame = None
    index_trails = {label: deque(maxlen=16) for label in ("Left", "Right")}
    music_active = False
    program_status = "READY"
    project_dir = Path(__file__).resolve().parent
    audio_path = Path(os.getenv("GHOST_AUDIO_PATH", str(project_dir / "assets" / "music_track.mp3")))
    audio_player = LocalAudioPlayer()
    subtitle_path = project_dir / "ghost_music.srt"
    try:
        music_cues = load_srt(subtitle_path)
        print(f"[SUBTITLE] Loaded {len(music_cues)} timed cues from {subtitle_path.name}")
    except (OSError, ValueError) as error:
        music_cues = []
        print(f"[SUBTITLE] Could not load {subtitle_path.name}: {error}")

    try:
        with HandTracker(model_path=str(model_path), num_hands=2) as tracker:
            consecutive_frame_misses = 0
            while True:
                success, frame = camera_reader.read()
                if not success or frame is None:
                    consecutive_frame_misses += 1
                    if consecutive_frame_misses == 1:
                        print("[CAMERA] Waiting for the first frame...")
                    if consecutive_frame_misses >= 100:
                        print("[CAMERA] No valid frame received for about 1 second; stopping. Check camera index/connection.")
                        break
                    time.sleep(0.01)
                    continue
                consecutive_frame_misses = 0

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

                # Show the cyan glow/trail at both index fingertips only.
                for label in ("Left", "Right"):
                    trail_hand = stable_hands.get(label)
                    if trail_hand and 8 in {point.id for point in trail_hand[0].landmarks}:
                        index_point = next(point for point in trail_hand[0].landmarks if point.id == 8)
                        index_trails[label].append((index_point.px, index_point.py))
                        draw_index_trail(frame, index_trails[label])
                    else:
                        index_trails[label].clear()

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
                custom_prediction = None
                if custom_model.prototypes and stable_hands:
                    # Use the clearest available hand; features are normalized for scale/translation.
                    custom_hand = max(stable_hands.values(), key=lambda item: item[0].confidence)[0]
                    custom_features = landmark_features(custom_hand.landmarks)
                    custom_prediction = custom_model.predict(custom_features)
                custom_label = custom_prediction[0] if custom_prediction else None
                custom_command_detected = False
                if custom_label is None:
                    custom_gesture_latched = None
                elif custom_label != custom_gesture_latched:
                    custom_gesture_latched = custom_label
                    custom_command = custom_prediction[1]
                    custom_command_detected = True
                    print(f"[CUSTOM GESTURES] Recognized {custom_label!r} -> {custom_command}")
                    if custom_command == "menu" and not music_active:
                        if wheel.is_open:
                            wheel._close("CUSTOM GESTURE / MENU CLOSED")
                        else:
                            wheel.is_open = True
                            wheel.status = "CUSTOM GESTURE / MENU ACTIVE"
                        program_status = wheel.status
                    elif custom_command == "filter":
                        portal_filter_index = (portal_filter_index + 1) % len(PORTAL_FILTERS)
                        print(f"[PORTAL] Filter: {PORTAL_FILTERS[portal_filter_index]}")
                    elif custom_command == "music":
                        if music_active:
                            audio_player.stop()
                            music_active = False
                            wheel.music_active = False
                            program_status = "MUSIC STOPPED / CUSTOM GESTURE"
                        else:
                            try:
                                if audio_player.play(audio_path):
                                    music_active = True
                                    wheel.music_active = True
                                    program_status = "MUSIC PLAYING / CUSTOM GESTURE"
                                else:
                                    program_status = "AUDIO FILE MISSING"
                            except Exception as error:
                                program_status = "MUSIC START FAILED"
                                print(f"[MUSIC] Custom gesture playback failed: {type(error).__name__}: {error}")
                    elif custom_command == "system":
                        # Deliberately safe placeholder: custom gestures do not run shell/Docker commands.
                        program_status = "CUSTOM SYSTEM GESTURE / SYSTEM PLACEHOLDER"
                if not music_active and not custom_command_detected:
                    # Reflect the gesture state machine's last command even if a hand disappears.
                    program_status = last_wheel_update.status
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
                if last_wheel_update.confirmed == "MUSIC" and not music_active:
                    try:
                        if audio_player.play(audio_path):
                            music_active = True
                            program_status = "MUSIC CONFIRMED / PLAYING"
                            print(f"[MUSIC] CONFIRMED. Playback started: {audio_path}")
                            print(f"[MUSIC] Output mixer: {__import__('pygame').mixer.get_init()}")
                        else:
                            wheel.music_active = False
                            program_status = "AUDIO FILE MISSING"
                            print(f"[MUSIC] CONFIRMATION RECEIVED, but file is missing: {audio_path}")
                            print("[MUSIC] Place your authorized audio file at assets/music_track.mp3 or set GHOST_AUDIO_PATH.")
                    except Exception as error:
                        music_active = False
                        wheel.music_active = False
                        program_status = "MUSIC START FAILED"
                        print(f"[MUSIC] CONFIRMATION RECEIVED, but playback failed: {type(error).__name__}: {error}")
                elif last_wheel_update.confirmed == "SYSTEM 2":
                    program_status = "SYSTEM 2 CONFIRMED"
                    print("[SYSTEM 2] CONFIRMED. Placeholder selected; no action is configured.")
                if last_wheel_update.shutdown:
                    audio_player.stop()
                    music_active = False
                    wheel.music_active = False
                    program_status = "MUSIC STOPPED"
                    print("[GHOST] BOTH INDEX FINGERS held steadily. Music stopped.")

                pinch_active = has_any_finger_pinch(stable_hands)
                if music_active and pinch_active and not portal_pinch_was_active:
                    portal_filter_index = (portal_filter_index + 1) % len(PORTAL_FILTERS)
                    print(f"[PORTAL] Filter: {PORTAL_FILTERS[portal_filter_index]}")
                portal_pinch_was_active = pinch_active

                if music_active and not audio_player.is_playing():
                    music_active = False
                    wheel.music_active = False
                    program_status = "MUSIC FINISHED"
                now = time.monotonic()
                frame_times.append(now)
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 and frame_times[-1] > frame_times[0] else 0.0
                subtitle_elapsed = audio_player.elapsed_seconds() if music_active else None
                active_subtitle = subtitle_at(music_cues, subtitle_elapsed) if subtitle_elapsed is not None else None
                subtitle_anchor = None
                if music_active and wheel.activator_hand in stable_hands:
                    subtitle_anchor = fingertip_anchor(stable_hands[wheel.activator_hand][0])
                elif music_active and stable_hands:
                    subtitle_anchor = fingertip_anchor(next(iter(stable_hands.values()))[0])
                current_subtitle_frame = detect_hand_frame_rect(stable_hands, frame.shape) if music_active else None
                current_subtitle_polygon = detect_hand_frame_polygon(stable_hands) if current_subtitle_frame else None
                if music_active and current_subtitle_polygon is not None:
                    apply_portal_filter(frame, current_subtitle_polygon, PORTAL_FILTERS[portal_filter_index])
                # Invalid/incomplete hand frames hide text immediately but never affect playback.
                smoothed_subtitle_frame = smooth_rect(smoothed_subtitle_frame, current_subtitle_frame) if current_subtitle_frame else None
                draw_hud(frame, poses, fps, now - started_at, frame_number, last_wheel_update,
                         smoothed_menu_anchor, music_active, active_subtitle, subtitle_anchor,
                         subtitle_frame_rect=smoothed_subtitle_frame,
                         subtitle_frame_polygon=current_subtitle_polygon,
                         program_status=program_status,
                         portal_filter_name=PORTAL_FILTERS[portal_filter_index])
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord("]"), ord("n"), ord("N")):
                    portal_filter_index = (portal_filter_index + 1) % len(PORTAL_FILTERS)
                    print(f"[PORTAL] Filter: {PORTAL_FILTERS[portal_filter_index]}")
                elif key in (ord("["), ord("p"), ord("P")):
                    portal_filter_index = (portal_filter_index - 1) % len(PORTAL_FILTERS)
                    print(f"[PORTAL] Filter: {PORTAL_FILTERS[portal_filter_index]}")
    finally:
        audio_player.close()
        camera_reader.close()
        camera.release()
        cv2.destroyAllWindows()
        print("PROJECT GHOST | Camera, audio, and hand tracker stopped.")


if __name__ == "__main__":
    main()
