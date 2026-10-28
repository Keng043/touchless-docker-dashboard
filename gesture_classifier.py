"""Simple rule-based hand-pose classification from MediaPipe landmarks."""

from hand_tracker import HandDetection

# MediaPipe Hands landmark IDs.
_FINGER_TIPS = {"INDEX": 8, "MIDDLE": 12, "RING": 16, "PINKY": 20}
_FINGER_PIPS = {"INDEX": 6, "MIDDLE": 10, "RING": 14, "PINKY": 18}


def classify_hand(landmarks) -> str:
    """Return a basic pose label using normalized landmark positions."""
    points = {point.id: point for point in landmarks}
    required = {0, 4, 5, 8, 9, 12, 13, 16, 17, 18, 20}
    if not required.issubset(points):
        return "Hand detected"

    extended = {
        name: points[tip].y < points[pip].y
        for name, tip in _FINGER_TIPS.items()
        for pip in [_FINGER_PIPS[name]]
    }
    thumb_tip = points[4]
    thumb_ip = points[3]
    thumb_mcp = points[2]
    thumb_extended = abs(thumb_tip.x - points[5].x) > abs(thumb_ip.x - thumb_mcp.x) * 1.15

    index_tip = points[8]
    thumb_gap = ((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2) ** 0.5
    if thumb_gap < 0.055:
        return "Pinch"

    fingers_up = [extended[name] for name in ("INDEX", "MIDDLE", "RING", "PINKY")]
    # A five-finger palm must include an extended thumb; four raised fingers
    # with a tucked thumb should not trigger the stop command.
    if all(fingers_up) and thumb_extended:
        return "Open palm"
    if extended["INDEX"] and extended["MIDDLE"] and not extended["RING"] and not extended["PINKY"]:
        return "Peace"
    if extended["INDEX"] and not any(extended[name] for name in ("MIDDLE", "RING", "PINKY")):
        return "Pointing"
    if thumb_extended and not any(fingers_up) and points[4].y < points[0].y:
        return "Thumbs up"
    if not any(fingers_up):
        return "Fist"
    return "Hand detected"


def classify_detection(hand: HandDetection) -> str:
    """Classify a complete hand detection."""
    return classify_hand(hand.landmarks)
