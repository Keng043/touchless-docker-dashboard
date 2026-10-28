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

On confirmation, the app plays a local audio file inside GHOST. It no longer opens YouTube or an external browser.

- Track label: **รอ SIS (TER)**
- Default audio path: `assets/music_track.mp3`
- Optional override: set the `GHOST_AUDIO_PATH` environment variable to your own audio file path.

Put an audio file you are authorized to use at `assets/music_track.mp3`. No song audio is bundled in this repository. If the file is missing, the top status bar shows `AUDIO FILE MISSING`; GHOST keeps running normally. Local playback is controlled by GHOST and is stopped when the two-fist shutdown gesture is detected or the app exits.

While audio is playing, GHOST reads `ghost_music.srt` and draws the caption near the tracked fingertips. Caption timing uses the local audio player's elapsed playback time, so it stays aligned with the app's playback position more reliably than the previous external YouTube launch.

`ghost_music.srt` contains the user-provided text and timestamps. Some phrases may be speech-recognition errors; the text is preserved as supplied and has not been independently verified against the recording.

## Camera quality

GHOST requests 1280×720 capture by default, asks Windows for MJPG to reduce USB bandwidth, and requests a one-frame capture buffer to reduce latency. Camera capture runs in a background thread so inference always uses the newest frame rather than waiting for a camera read on every processing cycle. Hand inference is resized to 480 pixels wide by default to reduce CPU work while normalized landmarks are drawn on the full-resolution camera frame. Set `GHOST_PROCESS_WIDTH=320` for maximum speed (may reduce hand-detection reliability), `480` for the recommended balance, `640` for more detail, or `0` to disable inference resizing. At startup it logs the resolution reported by the camera driver; the driver may ignore unsupported modes. The app does not intentionally upscale the camera frame for display.

For a supported 1080p camera, set these before launch:

```powershell
$env:GHOST_CAMERA_WIDTH = "1920"
$env:GHOST_CAMERA_HEIGHT = "1080"
.\.venv312\Scripts\python.exe main.py
```

If the picture remains pixelated, try `GHOST_CAMERA_INDEX=0` or another camera index, clean the lens, add front-facing light, and check Windows camera privacy/settings. Some built-in webcams advertise HD but output a lower-quality compressed stream; software cannot recover detail the camera never captured.

## Display

The top status strip shows only the program state and FPS. Session/frame counters, bottom legends, hand summary cards, and extra bottom status text have been removed for a cleaner camera view.

## System 2

System 2 is a placeholder in this build. The gesture selection and confirmation work, but no application is assigned to it yet.

## Safety and boundaries

Gesture actions are limited to playing the configured local audio file after the user confirms System 1. System 2 is a no-op placeholder. The application does not issue Docker commands, shell commands, or other operating-system operations through gestures.

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


## Custom gesture training (local webcam)

You can create your own hand pose and assign it to one of four commands: `menu`, `filter`, `music`, or `system`. The trainer stores normalized 3D landmarks only; it does not save webcam images or upload data.

1. Run `python gesture_trainer.py` from the project folder (or use the Python executable from your project virtual environment).
2. Enter a gesture name (for example, `heart-shape`) and select its command number: 1 menu, 2 filter, 3 music, 4 system.
3. Keep the pose in view and press **SPACE** for each sample. Capture at least 30 samples, changing the hand angle and distance slightly. Press **Q** or **Esc** to finish early.
4. GHOST writes examples to `data/gesture_samples.jsonl` and trains `data/custom_gestures.json`. These files are ignored by Git and remain local.
5. Restart `main.py`; it loads the trained model automatically. Hold the trained pose, then relax/change pose before repeating the command. Unknown poses are rejected by a distance threshold.

Command behavior: **menu** toggles the launcher, **filter** advances the portal filter, **music** starts/stops local playback, and **system** currently reports a safe placeholder. System gestures do not execute shell, Docker, or operating-system commands. Built-in gestures remain available when no custom model exists.

Training samples and the model have not been created for you yet: run the webcam trainer to collect examples from your own hand. The webcam integration itself should be verified on the target computer and camera after training.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Gesture recognition is heuristic and can be affected by lighting, camera angle, hand rotation, occlusion, and MediaPipe handedness estimates.
