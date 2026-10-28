"""Tests for local audio playback without requiring a real sound device."""

from pathlib import Path

from local_audio import LocalAudioPlayer


class FakeMusic:
    def __init__(self):
        self.loaded = None
        self.playing = False
        self.position = 0
        self.stopped = False
        self.start_playback = True
        self.volume = None

    def load(self, path):
        self.loaded = path

    def play(self):
        self.playing = self.start_playback

    def set_volume(self, volume):
        self.volume = volume

    def get_busy(self):
        return self.playing

    def get_pos(self):
        return self.position

    def stop(self):
        self.playing = False
        self.stopped = True


class FakeMixer:
    def __init__(self):
        self.music = FakeMusic()
        self.initialized = False
        self.quit_called = False

    def init(self):
        self.initialized = True

    def quit(self):
        self.quit_called = True


def test_missing_local_file_does_not_start_playback(tmp_path):
    mixer = FakeMixer()
    player = LocalAudioPlayer(mixer=mixer)
    assert player.play(tmp_path / "missing.mp3") is False
    assert not mixer.initialized
    assert not player.is_playing()


def test_local_file_playback_exposes_elapsed_seconds_and_stops(tmp_path):
    track = tmp_path / "track.mp3"
    track.write_bytes(b"test fixture")
    mixer = FakeMixer()
    player = LocalAudioPlayer(mixer=mixer)

    assert player.play(track) is True
    assert mixer.music.loaded == str(track.resolve())
    assert mixer.music.volume == 1.0
    assert player.is_playing()
    mixer.music.position = 12345
    assert player.elapsed_seconds() == 12.345

    player.stop()
    assert not player.is_playing()
    player.close()
    assert mixer.quit_called


def test_player_reports_failure_if_backend_does_not_start(tmp_path):
    import pytest

    track = tmp_path / "track.mp3"
    track.write_bytes(b"test fixture")
    mixer = FakeMixer()
    mixer.music.start_playback = False
    player = LocalAudioPlayer(mixer=mixer)

    with pytest.raises(RuntimeError, match="playback did not start"):
        player.play(track)
