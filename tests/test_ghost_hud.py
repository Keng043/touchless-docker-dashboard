"""Smoke tests for GHOST's floating launcher drawing helpers."""

import numpy as np

from gesture_wheel import GestureWheel
from main import (LatestFrameReader, configure_camera, detect_hand_frame_rect,
                  draw_frame_guides, draw_hud, draw_hand_frame_energy, draw_index_trail,
                  detect_hand_frame_polygon, floating_menu_items, palm_anchor, smooth_rect,
                  apply_portal_filter, PORTAL_FILTERS)


class FakeCamera:
    def __init__(self, width=1280, height=720):
        self.values = {
            3: width,
            4: height,
        }
        self.set_calls = []

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        if prop in self.values:
            self.values[prop] = int(value)
        return True

    def get(self, prop):
        return self.values.get(prop, 0)


def test_latest_frame_reader_returns_frames_and_stops_cleanly():
    import time

    class StreamingCamera:
        def __init__(self):
            self.counter = 0

        def read(self):
            self.counter += 1
            time.sleep(0.001)
            if self.counter % 5 == 0:
                return False, None
            return True, np.full((8, 8, 3), self.counter % 255, dtype=np.uint8)

    reader = LatestFrameReader(StreamingCamera())
    try:
        time.sleep(0.02)
        ok, frame = reader.read()
        assert ok
        assert frame.shape == (8, 8, 3)
    finally:
        reader.close()
    assert not reader._thread.is_alive()


def test_camera_requests_hd_mjpg_and_reports_negotiated_resolution(monkeypatch):
    monkeypatch.setenv("GHOST_CAMERA_WIDTH", "1280")
    monkeypatch.setenv("GHOST_CAMERA_HEIGHT", "720")
    camera = FakeCamera()
    width, height = configure_camera(camera)
    assert (width, height) == (1280, 720)
    assert any(prop == 6 for prop, _value in camera.set_calls)  # CAP_PROP_FOURCC
    assert any(prop == 3 for prop, _value in camera.set_calls)  # CAP_PROP_FRAME_WIDTH
    assert any(prop == 4 for prop, _value in camera.set_calls)  # CAP_PROP_FRAME_HEIGHT


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


def test_timed_unicode_caption_renders_even_when_menu_is_closed():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hud(frame, [], 24.1, 10.0, 280, music_active=True,
             subtitle_text="แล้วพี่จะไปไหน", subtitle_anchor=(320, 220),
             subtitle_frame_rect=(100, 100, 540, 350))
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


def test_hud_exposes_gesture_confirmation_status(monkeypatch):
    import main

    rendered_text = []
    monkeypatch.setattr(main, "put_text", lambda frame, text, *args, **kwargs: rendered_text.append(text))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hud(frame, [], 24.0, 1.0, 30, program_status="MUSIC CONFIRMED / PLAYING")
    assert "MUSIC CONFIRMED / PLAYING" in rendered_text


def test_hud_explains_how_to_activate_and_select(monkeypatch):
    import main

    rendered_text = []
    monkeypatch.setattr(main, "put_text", lambda frame, text, *args, **kwargs: rendered_text.append(text))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    wheel = GestureWheel(open_frames=1)
    update = wheel.update({"Left": ("Open palm", None)})
    draw_hud(frame, [], 24.0, 1.0, 30, wheel_update=update, program_status="READY")
    assert "MENU ACTIVE / 1 MUSIC" in rendered_text


def test_menu_overlay_renders_with_selected_system():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    wheel = GestureWheel(open_frames=1)
    update = wheel.update({"Left": ("Open palm", None)})
    draw_hud(frame, [("Left", "Open palm", 0.96)], 24.0, 2.0, 60, update, (320, 320))
    assert update.is_open
    assert int(frame.sum()) > 0


def test_hand_frame_rect_uses_thumb_and_index_corners_only():
    from types import SimpleNamespace

    def hand(coords):
        landmarks = [SimpleNamespace(id=i, px=coords.get(i, (0, 0))[0],
                                    py=coords.get(i, (0, 0))[1]) for i in range(21)]
        return SimpleNamespace(landmarks=landmarks)

    left = hand({0: (200, 300), 4: (150, 150), 8: (155, 400), 9: (200, 200),
                 12: (180, 210), 16: (170, 250), 20: (160, 290)})
    right = hand({0: (450, 300), 4: (500, 150), 8: (495, 400), 9: (450, 200),
                  12: (470, 210), 16: (480, 250), 20: (490, 290)})
    rect = detect_hand_frame_rect({"Left": (left, "Relaxed"), "Right": (right, "Fist")}, (720, 1280, 3))
    assert rect is not None
    assert rect[0] < 160 and rect[2] > 490
    assert rect[1] == 150 and rect[3] == 400
    assert detect_hand_frame_rect({"Left": (left, "Relaxed")}, (720, 1280, 3)) is None


def test_any_pair_of_fingertips_can_trigger_filter_pinch():
    from types import SimpleNamespace
    from main import has_any_finger_pinch

    # Thumb and ring finger pinch; index finger remains far away.
    landmarks = [
        SimpleNamespace(id=i, x=0.8, y=0.8, px=0, py=0) for i in range(21)
    ]
    for point_id, x, y in ((4, 0.40, 0.40), (16, 0.42, 0.41), (8, 0.70, 0.20)):
        landmarks[point_id] = SimpleNamespace(id=point_id, x=x, y=y, px=0, py=0)
    hand = SimpleNamespace(landmarks=landmarks)

    assert has_any_finger_pinch({"Left": (hand, "Hand detected")})
    assert not has_any_finger_pinch({})


def test_hand_frame_smoothing_tracks_motion_and_hides_when_invalid():
    assert smooth_rect(None, (100, 100, 500, 400)) == (100, 100, 500, 400)
    assert smooth_rect((100, 100, 500, 400), (200, 150, 600, 450)) == (135, 117, 535, 417)
    assert smooth_rect((135, 117, 535, 417), None) is None


def test_single_index_finger_trail_draws_glow():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    draw_index_trail(frame, [(20, 40), (30, 45), (42, 50)])
    assert int(frame.sum()) > 0


def test_hand_frame_energy_draws_outline_without_moving_beam_or_panel():
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    draw_hand_frame_energy(frame, (30, 30, 210, 130), tick=0.4,
                           polygon=((30, 30), (210, 40), (200, 130), (40, 120)))
    assert int(frame.sum()) > 0
    # The translucent interior is evenly shaded; no animated horizontal beam crosses it.
    assert abs(int(frame[80, 120].sum()) - int(frame[80, 100].sum())) <= 3


def test_every_portal_filter_changes_only_pixels_inside_hand_polygon():
    polygon = ((20, 15), (80, 25), (75, 65), (25, 60))
    y, x = np.indices((80, 100))
    source = np.stack(((x * 2) % 256, (y * 3) % 256, (x + y * 2) % 256), axis=2).astype(np.uint8)
    for filter_name in PORTAL_FILTERS:
        frame = source.copy()
        outside_before = frame[0:10, 0:10].copy()
        inside_before = frame[25:55, 35:65].copy()

        result = apply_portal_filter(frame, polygon, filter_name)

        assert result is frame
        assert np.any(result[25:55, 35:65] != inside_before), filter_name
        assert np.array_equal(result[0:10, 0:10], outside_before), filter_name


def test_portal_filter_ignores_missing_or_invalid_polygon():
    frame = np.full((30, 30, 3), 90, dtype=np.uint8)
    original = frame.copy()

    apply_portal_filter(frame, None)
    apply_portal_filter(frame, ((1, 1), (2, 2)))

    assert np.array_equal(frame, original)
