"""Smoke tests for GHOST's floating launcher drawing helpers."""

import numpy as np

from gesture_wheel import GestureWheel
from main import draw_frame_guides, draw_hud, floating_menu_items, palm_anchor


def test_hud_draws_on_desktop_frame():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_frame_guides(frame)
    draw_hud(frame, [("Right", "Open palm", 0.97)], 29.8, 12.3, 321)
    assert frame.shape == (720, 1280, 3)
    assert int(frame.sum()) > 0


def test_hud_draws_without_hands_on_small_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_frame_guides(frame)
    draw_hud(frame, [], 24.1, 1.2, 28)
    assert frame.shape == (480, 640, 3)
    assert int(frame.sum()) > 0


def test_floating_launcher_is_anchored_and_clamped():
    frame_shape = (480, 640, 3)
    left_items = floating_menu_items(frame_shape, (100, 260))
    right_items = floating_menu_items(frame_shape, (540, 260))
    assert min(rect[0] for _, rect in left_items) >= 8
    assert max(rect[2] for _, rect in right_items) <= 632
    assert min(rect[0] for _, rect in left_items) != min(rect[0] for _, rect in right_items)


def test_floating_launcher_shows_two_systems():
    items = floating_menu_items((720, 1280, 3), (640, 520))
    assert [command for command, _rect in items] == ["MUSIC", "SYSTEM 2"]
    assert items[0][1][1] < items[1][1][1]


def test_menu_overlay_renders_with_selected_system():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    wheel = GestureWheel(open_frames=1)
    update = wheel.update({"Left": ("Open palm", None)})
    draw_hud(frame, [("Left", "Open palm", 0.96)], 24.0, 2.0, 60, update, (320, 320))
    assert update.is_open
    assert int(frame.sum()) > 0
