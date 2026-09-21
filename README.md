# Touchless Docker Dashboard

Gesture-controlled local Docker monitoring and control using OpenCV and MediaPipe.

## What it does
- Tracks a hand with MediaPipe and maps the index fingertip to a virtual cursor.
- Uses thumb/index pinch detection for START, STOP, and RESTART actions.
- Displays container status, CPU usage, and memory usage.
- Polls Docker statistics on a background thread so camera rendering is not blocked.
- Falls back to a simulator when the Docker SDK/daemon is unavailable.

## Architecture
- `camera_streamer.py` — camera lifecycle and frame acquisition.
- `hand_tracker.py` — MediaPipe hand detection and landmark models.
- `docker_service.py` — Docker abstraction, real SDK adapter, and simulator.
- `ui_manager.py` — OpenCV HUD and gesture hit-testing.
- `main.py` — application orchestration and background stats worker.
- `tests/` — service-level regression tests.

## Security notes
This application can control the local Docker daemon. Treat Docker daemon access as a privileged capability.

- Do not expose the Docker socket/API to untrusted networks.
- Do not run the dashboard with broader OS privileges than required.
- Container IDs are passed to the Docker SDK rather than assembled into shell commands.
- The dashboard does not execute arbitrary shell commands from gesture input.
- Review the Docker daemon permissions before production use.

## Setup
Python 3.10+ is recommended. Create a virtual environment, then:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Place the MediaPipe `hand_landmarker.task` model in the project root. It is intentionally ignored by Git because it is a large binary asset.

## Test

```bash
python -m pytest -q
```

## Run

```bash
python main.py
```

Press `q` in the camera window to exit.

## Limitations
- Gesture control is sensitive to lighting, camera placement, and hand pose.
- The current dashboard targets a local desktop Docker daemon.
- No authentication layer is provided; OS/Docker daemon permissions are the security boundary.
- Docker stats are snapshots and may fail for containers that terminate while being sampled.
