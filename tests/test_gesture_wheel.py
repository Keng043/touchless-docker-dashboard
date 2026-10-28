"""Tests for PROJECT GHOST's two-hand launcher state machine."""

import pytest

from gesture_wheel import GestureWheel


def test_open_palm_activates_menu_after_stable_frames():
    wheel = GestureWheel(open_frames=2)
    assert not wheel.update({"Left": ("Open palm", None)}).is_open
    update = wheel.update({"Left": ("Open palm", None)})
    assert update.is_open
    assert update.activator_hand == "Left"
    assert "MENU ACTIVE" in update.status


def test_one_finger_selects_music_and_status_persists_on_tracking_loss():
    wheel = GestureWheel(open_frames=1, select_frames=2)
    wheel.update({"Left": ("Open palm", None)})
    wheel.update({"Left": ("Pointing", None)})
    update = wheel.update({"Left": ("Pointing", None)})
    assert update.selected == "MUSIC"
    assert update.is_open
    assert update.status == "MUSIC SELECTED / 2 HANDS CONFIRM"

    lost = wheel.update({})
    assert lost.is_open
    assert lost.selected == "MUSIC"
    assert lost.status == update.status


def test_two_fingers_select_system_two():
    wheel = GestureWheel(open_frames=1, select_frames=2)
    wheel.update({"Left": ("Open palm", None)})
    wheel.update({"Left": ("Peace", None)})
    update = wheel.update({"Left": ("Peace", None)})
    assert update.selected == "SYSTEM 2"
    assert update.status == "SYSTEM 2 SELECTED / 2 HANDS CONFIRM"


def test_both_hands_pointing_confirms_music_once():
    wheel = GestureWheel(open_frames=1, select_frames=1, confirm_frames=2)
    wheel.update({"Left": ("Open palm", None)})
    wheel.update({"Left": ("Pointing", None)})
    assert wheel.update({"Left": ("Pointing", None), "Right": ("Pointing", None)}).confirmed is None
    update = wheel.update({"Left": ("Pointing", None), "Right": ("Pointing", None)})
    assert update.confirmed == "MUSIC"
    assert not update.is_open
    assert update.status == "MUSIC CONFIRMED / PLAYING"
    assert wheel.music_active
    assert wheel.update({"Left": ("Pointing", None), "Right": ("Pointing", None)}).confirmed is None


def test_no_hand_gesture_stops_music_after_confirmation():
    wheel = GestureWheel(open_frames=1, confirm_frames=1, cancel_frames=2)
    wheel.update({"Left": ("Open palm", None)})
    update = wheel.update({"Left": ("Pointing", None), "Right": ("Pointing", None)})
    assert update.confirmed == "MUSIC"
    assert wheel.music_active

    for hands in (
        {"Left": ("Fist", None), "Right": ("Fist", None)},
        {"Left": ("Pointing", None), "Right": ("Pointing", None)},
        {"Left": ("Open palm", None), "Right": ("Open palm", None)},
        {},
    ):
        update = wheel.update(hands)
        assert not update.shutdown
        assert wheel.music_active
        assert not update.is_open
        assert update.status == "MUSIC CONFIRMED / PLAYING"


def test_one_hand_pointing_does_not_confirm():
    wheel = GestureWheel(open_frames=1, confirm_frames=1)
    wheel.update({"Left": ("Open palm", None)})
    update = wheel.update({"Left": ("Pointing", None)})
    assert update.confirmed is None
    assert update.is_open


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        GestureWheel(open_frames=0)
