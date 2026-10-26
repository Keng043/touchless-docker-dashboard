# Touchless Docker Dashboard

Gesture-controlled local Docker monitoring and control using OpenCV and MediaPipe.

## What it does
- Tracks a hand with MediaPipe and maps the index fingertip to a virtual cursor.
- Uses thumb/index pinch detection for START, STOP, and RESTART actions.
- Displays container status, CPU usage, and memory usage.
- Polls Docker statistics on a background thread so camera rendering is not blocked.
- Falls back to a simulator when the Docker SDK or Docker daemon is unavailable.

## Architecture
- `camera_streamer.py` — camera lifecycle and frame acquisition.
- `hand_tracker.py` — MediaPipe hand detection and landmark models.
- `docker_service.py` — Docker abstraction, real SDK adapter, and simulator.
- `ui_manager.py` — OpenCV HUD and gesture hit-testing.
- `main.py` — application orchestration and background stats worker.
- `tests/` — regression tests for Docker service behavior, camera cleanup, and UI hit-testing.

## Security notes
This application can control the local Docker daemon. Treat Docker daemon access as a privileged capability.

- Do not expose the Docker socket/API to untrusted networks.
- Do not run the dashboard with broader OS privileges than required.
- Container IDs are passed to the Docker SDK rather than assembled into shell commands.
- The dashboard does not execute arbitrary shell commands from gesture input.
- Review Docker daemon permissions before production use.

## Setup
Python 3.10–3.12 is recommended. MediaPipe and OpenCV are version-pinned for compatibility. On Windows, create a clean virtual environment with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the MediaPipe `hand_landmarker.task` model in the project root. It is intentionally ignored by Git because it is a large binary asset.

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Run

```powershell
.\.venv\Scripts\python.exe main.py
```

Press `q` in the camera window to exit.

### Optional performance profiling

Set `GHOST_PROFILE=1` to print rolling FPS and average/p95 timings for camera read, preprocessing, hand detection, UI rendering, and display approximately every five seconds. Profiling is disabled by default.

```powershell
$env:GHOST_PROFILE = "1"
.\.venv\Scripts\python.exe main.py
```

## Limitations
- Gesture control is sensitive to lighting, camera placement, and hand pose.
- The current dashboard targets a local desktop Docker daemon.
- No authentication layer is provided; OS/Docker daemon permissions are the security boundary.
- Docker stats are snapshots and may fail for containers that terminate while being sampled.
- If Docker is unavailable, the dashboard uses mock containers; actions in simulator mode do not affect real containers.
