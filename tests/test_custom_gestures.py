"""Tests for local custom gesture training and inference."""
import json

import pytest

from custom_gestures import (
    COMMANDS, FEATURE_SIZE, CustomGestureModel, append_sample,
    landmark_features, load_samples,
)
from hand_tracker import HandLandmark


def hand(offset=0.0, scale=1.0):
    points = []
    for index in range(21):
        x = 0.4 + (index % 5) * 0.015 * scale + offset
        y = 0.4 + (index // 5) * 0.02 * scale
        z = index * 0.001 * scale
        points.append(HandLandmark(index, x, y, z, int(x * 640), int(y * 480)))
    # Wrist and middle MCP must have a nonzero distance.
    return points


def test_landmark_features_are_scale_and_translation_normalized():
    base = landmark_features(hand())
    transformed = landmark_features(hand(offset=0.2, scale=1.5))
    assert len(base) == FEATURE_SIZE
    assert transformed == pytest.approx(base, abs=1e-6)


def test_landmark_features_reject_incomplete_hand():
    assert landmark_features(hand()[:-1]) is None


def test_append_and_load_samples_are_local_jsonl(tmp_path):
    path = tmp_path / "data" / "samples.jsonl"
    features = [0.1] * FEATURE_SIZE
    append_sample(path, "heart", "menu", features)
    append_sample(path, "heart", "menu", features)
    loaded = load_samples(path)
    assert len(loaded) == 2
    assert loaded[0]["label"] == "heart"
    assert loaded[0]["command"] == "menu"
    assert loaded[0]["features"] == features


def test_invalid_command_and_feature_count_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        append_sample(tmp_path / "samples.jsonl", "wave", "unknown", [0.0] * FEATURE_SIZE)
    with pytest.raises(ValueError):
        append_sample(tmp_path / "samples.jsonl", "wave", "menu", [0.0])


def test_model_train_predict_save_and_load(tmp_path):
    features = [0.2] * FEATURE_SIZE
    samples = [
        {"label": "heart", "command": "menu", "features": features},
        {"label": "heart", "command": "menu", "features": [v + 0.001 for v in features]},
        {"label": "peace", "command": "filter", "features": [v + 0.3 for v in features]},
    ]
    model = CustomGestureModel.train(samples)
    assert model.predict(features)[0:2] == ("heart", "menu")
    path = tmp_path / "model.json"
    model.save(path)
    restored = CustomGestureModel.load(path)
    assert restored.predict(features)[0:2] == ("heart", "menu")
    assert {item["command"] for item in restored.prototypes} <= set(COMMANDS)


def test_model_rejects_unknown_pose_by_distance():
    features = [0.0] * FEATURE_SIZE
    model = CustomGestureModel.train([
        {"label": "fist", "command": "menu", "features": [1.0] * FEATURE_SIZE}
    ], max_distance=0.1)
    assert model.predict(features) is None


def test_empty_training_data_requires_samples():
    with pytest.raises(ValueError, match="No gesture samples"):
        CustomGestureModel.train([])


def test_corrupt_dataset_reports_line_number(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps({"label": "bad", "command": "menu", "features": [float("nan")] * FEATURE_SIZE}) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_samples(path)
