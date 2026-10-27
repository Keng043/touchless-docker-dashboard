"""Tests for basic hand-pose classification."""

from hand_tracker import HandLandmark
from gesture_classifier import classify_hand


def make_hand(extended):
    """Create normalized test landmarks with selected fingers extended."""
    points = {
        0: (0.50, 0.90),
        1: (0.44, 0.78), 2: (0.40, 0.70), 3: (0.36, 0.62), 4: (0.32, 0.54),
        5: (0.43, 0.62), 6: (0.42, 0.52), 7: (0.41, 0.42), 8: (0.40, 0.30),
        9: (0.50, 0.60), 10: (0.50, 0.50), 11: (0.50, 0.40), 12: (0.50, 0.28),
        13: (0.57, 0.62), 14: (0.58, 0.54), 15: (0.59, 0.48), 16: (0.60, 0.42),
        17: (0.63, 0.67), 18: (0.66, 0.61), 19: (0.68, 0.56), 20: (0.70, 0.50),
    }
    finger_points = {
        "INDEX": (8, 6, (0.40, 0.30)),
        "MIDDLE": (12, 10, (0.50, 0.28)),
        "RING": (16, 14, (0.60, 0.42)),
        "PINKY": (20, 18, (0.70, 0.50)),
    }
    for name, (tip, pip, up_pos) in finger_points.items():
        if name in extended:
            points[tip] = up_pos
            points[pip] = (points[pip][0], up_pos[1] + 0.12)
        else:
            points[tip] = (points[pip][0], points[pip][1] + 0.08)
    if not extended:
        points[4] = (0.43, 0.78)
    return [HandLandmark(i, x, y, 0.0, int(x * 640), int(y * 480)) for i, (x, y) in points.items()]


def test_classifies_open_palm():
    assert classify_hand(make_hand({"INDEX", "MIDDLE", "RING", "PINKY"})) == "Open palm"


def test_classifies_fist():
    assert classify_hand(make_hand(set())) == "Fist"


def test_classifies_pointing():
    assert classify_hand(make_hand({"INDEX"})) == "Pointing"


def test_classifies_peace():
    assert classify_hand(make_hand({"INDEX", "MIDDLE"})) == "Peace"
