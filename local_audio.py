"""Local audio playback for PROJECT GHOST (pygame mixer backend)."""

from pathlib import Path
from typing import Any


class LocalAudioPlayer:
    """Play a local audio file and expose playback time for subtitle sync.

    The pygame dependency is imported lazily so subtitle/UI tests can run without
    initializing an audio device. A mixer-like object can be injected in tests.
    """

    def __init__(self, mixer: Any | None = None) -> None:
        self._mixer = mixer
        self._initialized = False
        self._path: Path | None = None

    @property
    def path(self) -> Path | None:
        return self._path

    def _get_mixer(self):
        if self._mixer is None:
            try:
                import pygame
            except ImportError as error:
                raise RuntimeError(
                    "Audio backend missing. Install project requirements with pip."
                ) from error
            self._mixer = pygame.mixer
        return self._mixer

    def play(self, path: str | Path) -> bool:
        """Start a local track; return False if the configured file is absent."""
        audio_path = Path(path).expanduser().resolve()
        if not audio_path.is_file():
            self._path = None
            return False
        mixer = self._get_mixer()
        if not self._initialized:
            mixer.init()
            self._initialized = True
        mixer.music.load(str(audio_path))
        mixer.music.set_volume(1.0)
        mixer.music.play()
        # Do not report success just because the file decoded: verify that the
        # backend actually entered its playing state.
        if not mixer.music.get_busy():
            self._path = None
            raise RuntimeError(
                "Audio backend accepted the file but playback did not start. "
                "Check Windows output device and app volume."
            )
        self._path = audio_path
        return True

    def elapsed_seconds(self) -> float | None:
        """Return current track position in seconds, or None if not playing."""
        if not self._initialized or self._mixer is None:
            return None
        if not self._mixer.music.get_busy():
            return None
        position_ms = self._mixer.music.get_pos()
        if position_ms is None or position_ms < 0:
            return None
        return position_ms / 1000.0

    def is_playing(self) -> bool:
        return bool(self._initialized and self._mixer is not None
                    and self._mixer.music.get_busy())

    def stop(self) -> None:
        if self._initialized and self._mixer is not None:
            self._mixer.music.stop()
        self._path = None

    def close(self) -> None:
        if self._mixer is not None:
            if self._initialized:
                self._mixer.music.stop()
                self._mixer.quit()
            self._initialized = False
        self._path = None
