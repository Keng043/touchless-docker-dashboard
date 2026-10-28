"""Local custom gesture dataset and lightweight nearest-centroid classifier.

Training examples are normalized MediaPipe landmarks stored as JSON Lines.
No image frames, network calls, or external ML packages are required.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

COMMANDS = ("menu", "filter", "music", "system")
LANDMARK_COUNT = 21
FEATURE_SIZE = LANDMARK_COUNT * 3


def landmark_features(landmarks):
    """Create translation/scale-normalized xyz features from 21 landmarks."""
    points = {point.id: point for point in landmarks}
    if not all(index in points for index in range(LANDMARK_COUNT)):
        return None
    wrist = points[0]
    # Wrist-to-middle-MCP distance approximates palm size and is stable enough
    # for different camera distances.
    palm = points[9]
    scale = math.hypot(palm.x - wrist.x, palm.y - wrist.y)
    if scale < 1e-5:
        return None
    features = []
    for index in range(LANDMARK_COUNT):
        point = points[index]
        features.extend((
            (point.x - wrist.x) / scale,
            (point.y - wrist.y) / scale,
            (point.z - wrist.z) / scale,
        ))
    return features


def append_sample(dataset_path, label, command, features):
    """Append one labeled sample locally as a JSONL record."""
    if command not in COMMANDS:
        raise ValueError(f"Unsupported command: {command}")
    if not label or not label.strip():
        raise ValueError("Gesture label cannot be empty")
    if len(features) != FEATURE_SIZE:
        raise ValueError(f"Expected {FEATURE_SIZE} features")
    path = Path(dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "label": label.strip(),
            "command": command,
            "features": [float(value) for value in features],
        }, ensure_ascii=False) + "\n")


def load_samples(dataset_path):
    """Load and validate JSONL samples; malformed rows fail with line context."""
    path = Path(dataset_path)
    if not path.exists():
        return []
    samples = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if (not isinstance(row.get("label"), str) or not row["label"].strip()
                        or row.get("command") not in COMMANDS
                        or len(row.get("features", [])) != FEATURE_SIZE):
                    raise ValueError("invalid label, command, or feature count")
                features = [float(value) for value in row["features"]]
                if not all(math.isfinite(value) for value in features):
                    raise ValueError("features must be finite")
                samples.append({"label": row["label"], "command": row["command"],
                                "features": features})
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid gesture sample on line {line_number}: {error}") from error
    return samples


class CustomGestureModel:
    """Nearest-centroid classifier with a conservative distance rejection gate."""

    def __init__(self, prototypes=None, max_distance=0.65):
        self.prototypes = prototypes or []
        self.max_distance = float(max_distance)

    @classmethod
    def train(cls, samples, max_distance=0.65):
        if not samples:
            raise ValueError("No gesture samples found. Capture examples first.")
        grouped = defaultdict(list)
        for sample in samples:
            if sample["command"] not in COMMANDS:
                raise ValueError(f"Unsupported command: {sample['command']}")
            if len(sample["features"]) != FEATURE_SIZE:
                raise ValueError("Invalid feature count in training sample")
            grouped[(sample["label"], sample["command"])].append(sample["features"])
        prototypes = []
        for (label, command), rows in sorted(grouped.items()):
            centroid = [sum(row[i] for row in rows) / len(rows) for i in range(FEATURE_SIZE)]
            prototypes.append({"label": label, "command": command,
                               "count": len(rows), "centroid": centroid})
        return cls(prototypes, max_distance)

    def predict(self, features):
        """Return (label, command, distance), or None if no safe match."""
        if features is None or len(features) != FEATURE_SIZE or not self.prototypes:
            return None
        ranked = []
        for item in self.prototypes:
            distance = math.sqrt(sum(
                (float(value) - center) ** 2
                for value, center in zip(features, item["centroid"])
            ) / FEATURE_SIZE)
            ranked.append((distance, item))
        distance, best = min(ranked, key=lambda pair: pair[0])
        if distance > self.max_distance:
            return None
        return best["label"], best["command"], distance

    def save(self, model_path):
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "version": 1, "max_distance": self.max_distance,
            "prototypes": self.prototypes,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_path):
        path = Path(model_path)
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1 or not isinstance(payload.get("prototypes"), list):
            raise ValueError("Unsupported or invalid custom gesture model")
        for item in payload["prototypes"]:
            if (item.get("command") not in COMMANDS
                    or len(item.get("centroid", [])) != FEATURE_SIZE):
                raise ValueError("Invalid prototype in custom gesture model")
        return cls(payload["prototypes"], payload.get("max_distance", 0.65))
