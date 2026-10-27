# PROJECT GHOST — Floating Gesture Launcher

PROJECT GHOST is a webcam hand-gesture interface. An open palm activates a compact launcher that floats above the palm and follows its movement.

## Single-hand controls

- **Open palm:** activate the floating launcher.
- **One raised finger:** select **System 1 — Music**.
- **Two raised fingers:** select **System 2**.
- **Fist with the active hand:** confirm the selected system.
- **Fists with both hands:** deactivate the gesture interface.
- Lower the active hand or lose it from view for a short period to close the launcher.

The launcher position is calculated from palm landmarks, smoothed to reduce jitter, and clamped to the camera frame. Gesture labels are stabilized over multiple frames to reduce accidental selection.

## System 1 — Music

On confirmation, the app opens the supplied YouTube link in the default browser:

- Track label: **รอ SIS (TER)**
- Link: https://youtu.be/hqCMb8Vr1Ss?si=f9VvH1pJmNsY7qQf

While the app's music state is active, musical notes are drawn beside the tracked fingertips. The current build does not extract or synchronize YouTube captions, so it does **not** display timed lyrics. Timed karaoke lyrics can be added if an authorized lyric/subtitle file is provided. Closing the gesture interface does not forcibly stop or close the external browser playback.

## System 2

System 2 is a placeholder in this build. The gesture selection and confirmation work, but no application is assigned to it yet.

## Safety and boundaries

Gesture actions are limited to opening the specified YouTube URL after the user confirms System 1. System 2 is a no-op placeholder. The application does not issue Docker commands, shell commands, or other operating-system operations through gestures.

## Setup (Windows)

Python 3.10–3.12 is recommended.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Ensure `hand_landmarker.task` exists in the project root.

## Run

```powershell
.\.venv\Scripts\python.exe main.py
```

The default camera index is `1`, with fallback to `0`. To choose a camera:

```powershell
$env:GHOST_CAMERA_INDEX = "0"
.\.venv\Scripts\python.exe main.py
```

Press `q` or `Esc` in the camera window to exit.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Gesture recognition is heuristic and can be affected by lighting, camera angle, hand rotation, occlusion, and MediaPipe handedness estimates.
