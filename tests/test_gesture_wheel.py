"""Tests for the single-hand floating launcher."""

import pytest

from gesture_wheel import GestureWheel


def test_open_palm_activates_menu_after_stable_frames():
    wheel = GestureWheel(open_frames=2)
    assert not wheel.update({"Left": ("Open palm", None)}).is_open
    update = wheel.update({"Left": ("Open palm", None)})
    assert update.is_open
    assert update.activator_hand == "Left"
    assert update.controller_hand is None


def test_one_finger_selects_music():
    wheel = GestureWheel(open_frames=1, select_frames=2)
    wheel.update({"Left": ("Open palm", None)})
    wheel.update({"Left": ("Pointing", None)})
    update = wheel.update({"Left": ("Pointing", None)})
    assert update.selected == "MUSIC"
    assert update.is_open


def test_two_fingers_select_system_two():
    wheel = GestureWheel(open_frames=1, select_frames=2)
    wheel.update({"Left": ("Open palm", None)})
    wheel.update({"Left": ("Peace", None)})
    update = wheel.update({"Left": ("Peace", None)})
    assert update.selected == "SYSTEM 2"


def test_fist_confirms_selected_system_once():
    wheel = GestureWheel(open_frames=1, confirm_frames=2)
    wheel.update({"Left": ("Open palm", None)})
    wheel.update({"Left": ("Peace", None)})
    wheel.update({"Left": ("Peace", None)})
    assert wheel.selected == "SYSTEM 2"
    assert wheel.update({"Left": ("Fist", None)}).confirmed is None
    assert wheel.update({"Left": ("Fist", None)}).confirmed == "SYSTEM 2"
    assert wheel.update({"Left": ("Fist", None)}).confirmed is None


def test_two_fists_deactivate_system():
    wheel = GestureWheel(open_frames=1, cancel_frames=2)
    wheel.update({"Left": ("Open palm", None)})
    assert wheel.update({"Left": ("Fist", None), "Right": ("Fist", None)}).is_open
    update = wheel.update({"Left": ("Fist", None), "Right": ("Fist", None)})
    assert not update.is_open
    assert update.shutdown
    assert "SYSTEM OFF" in update.status


def test_one_fist_does_not_shutdown_system():
    wheel = GestureWheel(open_frames=1, confirm_frames=1)
    wheel.update({"Left": ("Open palm", None)})
    update = wheel.update({"Left": ("Fist", None)})
    assert update.is_open
    assert update.confirmed == "MUSIC"
    assert not update.shutdown


def test_menu_closes_when_activator_lowers_hand_or_is_lost():
    wheel = GestureWheel(open_frames=1, release_frames=2)
    wheel.update({"Left": ("Open palm", None)})
    wheel.update({"Left": ("Thumbs up", None)})
    update = wheel.update({"Left": ("Thumbs up", None)})
    assert not update.is_open

    wheel = GestureWheel(open_frames=1, missing_frames=2)
    wheel.update({"Left": ("Open palm", None)})
    wheel.update({})
    assert wheel.update({}).is_open is False


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        GestureWheel(open_frames=0)
