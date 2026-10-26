"""Regression tests for camera frame handling and resource cleanup."""

import numpy as np

from camera_streamer import CameraStreamer


class FakeCapture:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


def test_read_returns_an_independent_frame_copy():
    streamer = CameraStreamer()
    source_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    streamer.frame = source_frame
    streamer.grabbed = True

    success, frame = streamer.read()

    assert success is True
    assert frame is not source_frame
    frame[0, 0] = (255, 255, 255)
    assert source_frame[0, 0].tolist() == [0, 0, 0]


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
