"""Tests for the timed captions used by PROJECT GHOST."""

from pathlib import Path

import pytest

from subtitle_reader import load_srt, parse_srt, subtitle_at


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_project_music_srt_loads_and_is_ordered():
    cues = load_srt(PROJECT_DIR / "ghost_music.srt")
    assert len(cues) == 61
    assert cues[0].start == 9.0
    assert cues[0].end == 14.0
    assert cues[0].text == "แล้วพี่จะไปที่ไหน"
    assert cues[-1].end == 235.0
    assert all(cue.end > cue.start for cue in cues)
    assert all(cues[index].start >= cues[index - 1].start for index in range(1, len(cues)))


def test_subtitle_at_returns_active_caption_and_respects_gaps():
    cues = load_srt(PROJECT_DIR / "ghost_music.srt")
    assert subtitle_at(cues, 8.99) is None
    assert subtitle_at(cues, 9.0) == "แล้วพี่จะไปที่ไหน"
    assert subtitle_at(cues, 14.0) == "ไปกับใคร ผมก็จะรอพี่"
    assert subtitle_at(cues, 17.5) is None


def test_parser_rejects_invalid_timing():
    with pytest.raises(ValueError, match="Invalid SRT timing"):
        parse_srt("1\n00:00:02,000 - 00:00:01,000\nCaption")


def test_parser_rejects_end_before_start():
    with pytest.raises(ValueError, match="end must be after start"):
        parse_srt("1\n00:00:03,000 --> 00:00:02,000\nCaption")
