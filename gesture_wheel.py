"""Gesture state machine for PROJECT GHOST's two-hand launcher."""

from dataclasses import dataclass

COMMANDS = ("MUSIC", "SYSTEM 2")


@dataclass(frozen=True)
class WheelUpdate:
    is_open: bool
    selected: str
    status: str
    confirmed: str | None = None
    activator_hand: str | None = None
    controller_hand: str | None = None
    shutdown: bool = False


class GestureWheel:
    """Open with one palm, select with one finger, confirm with two pointing hands.

    Tracking loss does not close the menu or erase the last selected command.
    Music has no hand-gesture stop command; playback ends when the app closes. Poses are handedness -> (pose, unused_direction).
    """

    def __init__(self, open_frames: int = 3, select_frames: int = 2,
                 confirm_frames: int = 3, cancel_frames: int = 10,
                 missing_frames: int = 12, release_frames: int = 3):
        thresholds = (open_frames, select_frames, confirm_frames,
                      cancel_frames, missing_frames, release_frames)
        if min(thresholds) < 1:
            raise ValueError("Gesture frame thresholds must be at least 1")
        self.open_frames = open_frames
        self.select_frames = select_frames
        self.confirm_frames = confirm_frames
        # About 0.6 s at GHOST's typical ~16 FPS; configurable for tests/tuning.
        self.cancel_frames = cancel_frames
        # Retained for API compatibility; temporary tracking loss no longer closes the menu.
        self.missing_frames = missing_frames
        self.release_frames = release_frames
        self.is_open = False
        self.selected_index = 0
        self.status = "OPEN PALM TO ACTIVATE"
        self.activator_hand = None
        self.controller_hand = None
        self._pose_counts = {}
        self._confirm_latched = False
        self.music_active = False

    @property
    def selected(self) -> str:
        return COMMANDS[self.selected_index]

    def update(self, hands: dict[str, tuple[str | None, str | None]]) -> WheelUpdate:
        normalized = {
            label: pose
            for label, (pose, _unused) in hands.items()
            if label in ("Left", "Right") and pose is not None
        }
        for label in ("Left", "Right"):
            pose = normalized.get(label)
            old_pose, count = self._pose_counts.get(label, (None, 0))
            self._pose_counts[label] = (pose, count + 1 if pose == old_pose else 1)

        confirmed = None
        shutdown = False
        both_pointing = all(normalized.get(label) == "Pointing" for label in ("Left", "Right"))

        # No gesture stops music. Open the launcher only while music is inactive.
        if not self.is_open and not self.music_active:
            for label in ("Left", "Right"):
                if (normalized.get(label) == "Open palm"
                        and self._pose_counts[label][1] >= self.open_frames):
                    self.is_open = True
                    self.activator_hand = label
                    self.status = "MENU ACTIVE / SELECT COMMAND"
                    self._confirm_latched = False
                    break

        if self.is_open:
                # Selection can be made with either visible hand. Keep the last
                # selection/status when tracking drops or an unrecognized pose appears.
                for label in ("Left", "Right"):
                    pose = normalized.get(label)
                    count = self._pose_counts[label][1]
                    if pose == "Pointing" and count >= self.select_frames:
                        self.selected_index = 0
                        self.activator_hand = label
                        self.status = "MUSIC SELECTED / 2 HANDS CONFIRM"
                    elif pose == "Peace" and count >= self.select_frames:
                        self.selected_index = 1
                        self.activator_hand = label
                        self.status = "SYSTEM 2 SELECTED / 2 HANDS CONFIRM"

                # Confirmation requires both hands to show one raised index finger.
                if (both_pointing
                        and min(self._pose_counts[label][1] for label in ("Left", "Right")) >= self.confirm_frames
                        and not self._confirm_latched):
                    confirmed = self.selected
                    self.status = ("MUSIC CONFIRMED / PLAYING" if confirmed == "MUSIC"
                                   else "SYSTEM 2 CONFIRMED")
                    self._confirm_latched = True
                    if confirmed == "MUSIC":
                        self.music_active = True
                    self.is_open = False
                    self.activator_hand = None
                    self.controller_hand = None

        return WheelUpdate(self.is_open, self.selected, self.status, confirmed,
                           self.activator_hand, self.controller_hand, shutdown)

    def _close(self, status: str) -> None:
        self.is_open = False
        self.status = status
        self.activator_hand = None
        self.controller_hand = None
        self._confirm_latched = False
