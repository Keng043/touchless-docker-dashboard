"""Regression tests for touchless dashboard UI hit-testing."""

from ui_manager import UIManager


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
