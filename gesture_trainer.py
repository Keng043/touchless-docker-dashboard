"""Capture custom hand gestures from the webcam and train a local model.

Run: python gesture_trainer.py
Press SPACE to save one sample, Q/ESC to finish capture and train.
"""
from pathlib import Path
import cv2

from custom_gestures import COMMANDS, CustomGestureModel, append_sample, landmark_features, load_samples
from hand_tracker import HandTracker

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "gesture_samples.jsonl"
MODEL = ROOT / "data" / "custom_gestures.json"


def main():
    label = input("Name this gesture (e.g. two-finger-heart): ").strip()
    if not label:
        raise SystemExit("Gesture label cannot be empty.")
    print("Choose command: " + ", ".join(f"{i + 1}={name}" for i, name in enumerate(COMMANDS)))
    try:
        command_index = int(input("Command number: ").strip()) - 1
        command = COMMANDS[command_index]
    except (ValueError, IndexError):
        raise SystemExit("Invalid command number.")
    target = input("Samples to capture [30]: ").strip()
    try:
        target_count = max(5, min(300, int(target or "30")))
    except ValueError:
        raise SystemExit("Sample count must be a number.")

    camera_index = int(__import__("os").getenv("GHOST_CAMERA_INDEX", "1"))
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise SystemExit("Could not open webcam. Check GHOST_CAMERA_INDEX.")
    captured = 0
    try:
        model_path = ROOT / "hand_landmarker.task"
        with HandTracker(model_path=str(model_path), num_hands=1) as tracker:
            timestamp = 0
            while captured < target_count:
                ok, frame = camera.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                timestamp += 34
                hands = tracker.process_frame(frame, timestamp)
                features = landmark_features(hands[0].landmarks) if hands else None
                cv2.putText(frame, f"Gesture: {label} | command: {command}", (18, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 255, 210), 2)
                cv2.putText(frame, f"Samples: {captured}/{target_count} | SPACE capture | Q quit",
                            (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 255, 210), 2)
                cv2.putText(frame, "Move hand slightly between samples; vary angle/distance.",
                            (18, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 255, 210), 1)
                cv2.imshow("PROJECT GHOST - Gesture Trainer", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == 32:
                    if features is None:
                        print("[TRAIN] No complete hand detected; sample skipped.")
                        continue
                    append_sample(DATASET, label, command, features)
                    captured += 1
                    print(f"[TRAIN] Saved sample {captured}/{target_count} for {label} -> {command}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    samples = load_samples(DATASET)
    if not samples:
        print("[TRAIN] No samples saved; model unchanged.")
        return
    trained = CustomGestureModel.train(samples)
    trained.save(MODEL)
    counts = {}
    for sample in samples:
        key = (sample["label"], sample["command"])
        counts[key] = counts.get(key, 0) + 1
    print(f"[TRAIN] Model saved: {MODEL}")
    print(f"[TRAIN] Total samples: {len(samples)}; gesture classes: {len(counts)}")
    for (gesture_label, gesture_command), count in sorted(counts.items()):
        print(f"  {gesture_label} -> {gesture_command}: {count} samples")
    print("[TRAIN] Data and model remain on this computer; no images are saved or uploaded.")


if __name__ == "__main__":
    main()
