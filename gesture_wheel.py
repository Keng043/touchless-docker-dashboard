"""Single-hand floating launcher state machine. Gestures trigger only explicit local actions."""

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
    """Open with an open palm; one finger selects item 1, two select item 2.

    A fist confirms the current selection. Two simultaneous fists shut down the
    gesture UI. Input is handedness -> (pose, unused_direction).
    """

    def __init__(self, open_frames: int = 3, select_frames: int = 2,
                 confirm_frames: int = 2, cancel_frames: int = 3,
                 missing_frames: int = 12, release_frames: int = 3):
        thresholds = (open_frames, select_frames, confirm_frames,
                      cancel_frames, missing_frames, release_frames)
        if min(thresholds) < 1:
            raise ValueError("Gesture frame thresholds must be at least 1")
        self.open_frames = open_frames
        self.select_frames = select_frames
        self.confirm_frames = confirm_frames
        self.cancel_frames = cancel_frames
        self.missing_frames = missing_frames
        self.release_frames = release_frames
        self.is_open = False
        self.selected_index = 0
        self.status = "Open palm to activate"
        self.activator_hand = None
        self.controller_hand = None
        self._pose_counts = {}
        self._missing_count = 0
        self._release_count = 0
        self._confirm_latched = False
        self._shutdown_latched = False

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
        both_fists = all(normalized.get(label) == "Fist" for label in ("Left", "Right"))
        if both_fists:
            if min(self._pose_counts[label][1] for label in ("Left", "Right")) >= self.cancel_frames:
                shutdown = not self._shutdown_latched
                self._shutdown_latched = True
                self._close("SYSTEM OFF  /  Both fists detected")
        else:
            self._shutdown_latched = False
            if not self.is_open:
                for label in ("Left", "Right"):
                    if (normalized.get(label) == "Open palm"
                            and self._pose_counts[label][1] >= self.open_frames):
                        self.is_open = True
                        self.activator_hand = label
                        self.status = "MENU ACTIVE  /  1 finger: MUSIC  /  2 fingers: SYSTEM 2"
                        self._missing_count = 0
                        self._release_count = 0
                        self._confirm_latched = False
                        break
            else:
                pose = normalized.get(self.activator_hand)
                if pose is None:
                    self._missing_count += 1
                else:
                    self._missing_count = 0

                if pose == "Open palm":
                    self._release_count = 0
                    self._confirm_latched = False
                elif pose in ("Pointing", "Peace"):
                    self._release_count = 0
                    self._confirm_latched = False
                    count = self._pose_counts[self.activator_hand][1]
                    if count >= self.select_frames:
                        self.selected_index = 0 if pose == "Pointing" else 1
                        self.status = f"{self.selected} selected  /  Make a fist to confirm"
                elif pose == "Fist":
                    self._release_count = 0
                    count = self._pose_counts[self.activator_hand][1]
                    if count >= self.confirm_frames and not self._confirm_latched:
                        confirmed = self.selected
                        self.status = f"CONFIRMED: {confirmed}"
                        self._confirm_latched = True
                elif pose is None:
                    if self._missing_count >= self.missing_frames:
                        self._close("MENU CLOSED  /  Hand lost")
                else:
                    self._release_count += 1
                    if self._release_count >= self.release_frames:
                        self._close("MENU CLOSED  /  Open palm lowered")

        return WheelUpdate(self.is_open, self.selected, self.status, confirmed,
                           self.activator_hand, None, shutdown)

    def _close(self, status: str) -> None:
        self.is_open = False
        self.status = status
        self.activator_hand = None
        self.controller_hand = None
        self._missing_count = 0
        self._release_count = 0
        self._confirm_latched = False
