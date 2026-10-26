"""Regression tests for touchless dashboard UI hit-testing."""

from main import FrameProfiler
from ui_manager import UIManager


def test_frame_profiler_disabled_is_silent_and_collects_no_samples(capsys):
    profiler = FrameProfiler(enabled=False, report_interval=0)
    started = profiler.start_stage()
    profiler.record("camera_read", started)
    profiler.complete_frame()

    assert started == 0
    assert not any(profiler.samples.values())
    assert capsys.readouterr().out == ""


def test_frame_profiler_percentile_uses_nearest_rank():
    assert FrameProfiler.percentile([1.0, 2.0, 3.0, 4.0], 95) == 4.0
    assert FrameProfiler.percentile([10.0, 20.0, 30.0, 40.0], 50) == 20.0


def test_frame_profiler_percentile_rejects_invalid_inputs():
    import pytest

    with pytest.raises(ValueError):
        FrameProfiler.percentile([], 95)
    with pytest.raises(ValueError):
        FrameProfiler.percentile([1.0], 0)


def test_frame_profiler_reports_stage_metrics_when_enabled(capsys):
    profiler = FrameProfiler(enabled=True, report_interval=0.0, window_size=4)
    started = profiler.start_stage()
    profiler.record("camera_read", started)
    profiler.complete_frame()

    output = capsys.readouterr().out
    assert "[PROFILE] FPS=" in output
    assert "camera_read avg=" in output
    assert "p95=" in output


def test_draw_hud_panel_blends_only_the_sidebar_region():
    import numpy as np

    ui = UIManager(width=1280, height=720)
    frame = np.full((720, 1280, 3), 100, dtype=np.uint8)

    ui.draw_hud_panel(frame)

    # A quiet pixel inside the filled panel receives the configured 75/25 blend.
    assert frame[100, 100].tolist() == [48, 40, 40]
    # A pixel in the radar area must not be changed by the sidebar blend.
    assert frame[300, 700].tolist() == [100, 100, 100]


def test_detect_button_click_returns_action_and_container_id():
    ui = UIManager()
    ui.interactive_buttons = {
        "START_container123": (10, 20, 50, 60, "container123"),
        "RESTART_container456": (70, 20, 120, 60, "container456"),
    }

    assert ui.detect_button_click(30, 40) == ("START", "container123")
    assert ui.detect_button_click(100, 40) == ("RESTART", "container456")


def test_detect_button_click_includes_button_edges():
    ui = UIManager()
    ui.interactive_buttons = {
        "STOP_container123": (10, 20, 50, 60, "container123"),
    }

    assert ui.detect_button_click(10, 20) == ("STOP", "container123")
    assert ui.detect_button_click(50, 60) == ("STOP", "container123")


def test_detect_button_click_returns_none_outside_buttons():
    ui = UIManager()
    ui.interactive_buttons = {
        "STOP_container123": (10, 20, 50, 60, "container123"),
    }

    assert ui.detect_button_click(9, 40) is None
    assert ui.detect_button_click(30, 61) is None
    assert ui.detect_button_click(500, 500) is None
