"""Small SRT reader for PROJECT GHOST's timed music captions."""

from dataclasses import dataclass
from pathlib import Path
import re


_TIMESTAMP = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})$"
)


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str


def _seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def parse_srt(content: str) -> list[SubtitleCue]:
    """Parse SRT text, ignoring blank blocks and rejecting malformed timings."""
    cues: list[SubtitleCue] = []
    for block in re.split(r"\r?\n\s*\r?\n", content.strip()):
        lines = [line.rstrip("\r") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timing_index = 1 if lines[0].strip().isdigit() else 0
        if timing_index >= len(lines):
            raise ValueError(f"Missing SRT timing line in block: {block[:80]!r}")
        match = _TIMESTAMP.match(lines[timing_index].strip())
        if not match:
            raise ValueError(f"Invalid SRT timing line: {lines[timing_index]!r}")
        values = match.groups()
        start = _seconds(*values[:4])
        end = _seconds(*values[4:])
        text = "\n".join(lines[timing_index + 1:]).strip()
        if end <= start:
            raise ValueError(f"Subtitle end must be after start: {lines[timing_index]!r}")
        if not text:
            raise ValueError(f"Subtitle text is empty: {lines[timing_index]!r}")
        cues.append(SubtitleCue(start, end, text))
    if any(cues[i].start < cues[i - 1].start for i in range(1, len(cues))):
        raise ValueError("SRT cues must be sorted by start time")
    return cues


def load_srt(path: str | Path) -> list[SubtitleCue]:
    """Load an UTF-8 (with optional BOM) SRT file."""
    return parse_srt(Path(path).read_text(encoding="utf-8-sig"))


def subtitle_at(cues: list[SubtitleCue], elapsed_seconds: float) -> str | None:
    """Return the caption active at elapsed_seconds, or None between cues."""
    for cue in cues:
        if cue.start <= elapsed_seconds < cue.end:
            return cue.text
        if cue.start > elapsed_seconds:
            break
    return None
