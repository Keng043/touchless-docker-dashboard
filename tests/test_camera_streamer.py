"""Regression tests for camera resource cleanup."""

from camera_streamer import CameraStreamer


class FakeCapture:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def test_stop_releases_capture_even_if_worker_already_stopped():
    streamer = CameraStreamer()
    capture = FakeCapture()
    streamer.cap = capture
    streamer.started = False

    streamer.stop()

    assert capture.released is True
    assert streamer.cap is None


def test_stop_is_safe_when_called_more_than_once():
    streamer = CameraStreamer()
    capture = FakeCapture()
    streamer.cap = capture

    streamer.stop()
    streamer.stop()

    assert capture.released is True
    assert streamer.cap is None
