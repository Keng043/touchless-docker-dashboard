# 🐳 Touchless Docker Dashboard 👁️💻

An interactive, gesture-controlled container management dashboard that utilizes Computer Vision (OpenCV + Google MediaPipe) to orchestrate and monitor local/simulated Docker environments.

This project is built from the ground up using **Object-Oriented Programming (OOP)** and strictly adheres to **SOLID Design Principles** to solve common real-time video processing bottlenecks.

---

## 🚀 Key Features

* **Asynchronous Multi-Threaded Camera Streaming:** Offloads heavy video frame pulling onto a background worker thread to ensure zero frame-drops in the rendering layer.
* **Futuristic Cyberpunk HUD Display:** A stunning heads-up display showing running/stopped containers, animated progress bars for live CPU % and Memory usage metrics.
* **Holographic Gesture Cursor:** Smoothly tracks your index finger tip to act as a virtual mouse pointer.
* **Pinch-to-Click Interaction:** Detects when you pinch your index finger and thumb together to trigger Docker container actions (`START`, `STOP`, `RESTART`) with built-in robust debouncing.
* **Dual-Hand Magic Portal:** Open a magical neon-styled geometric portal by holding up both hands. Features real-time edge particle spark physics (a direct nod to *Dr. Strange / Retrolens* style).
* **Smart Fallback Simulator (Liskov Substitution):** Detects if Docker Daemon or the python `docker` SDK is missing and seamlessly falls back to a highly realistic mock container stack with real-time stats fluctuations and fully interactive feedback.

---

## 📐 Architecture & SOLID Principles

* **Single Responsibility Principle (SRP):**
  * `CameraStreamer`: Strictly handles thread-safe asynchronous camera frame grabbing.
  * `HandTracker`: Purely extracts and scales hand landmark coordinates into decoupled immutable dataclasses.
  * `DockerService`: Exclusively manages container list queries and life-cycle controls.
  * `UIManager`: Handles CV graphics drawing, HUD drawing, and geometric math for portals and cursor collisions.
* **Liskov Substitution Principle (LSP) & Dependency Inversion Principle (DIP):**
  * `IDockerService` defines an interface. Both `RealDockerService` and `MockDockerService` implement it seamlessly, enabling full simulation capabilities when a live daemon isn't present.
  * High-level client code (`main.py`) interacts purely with the interface rather than the concrete implementations.

---

## 📦 Project Directory Structure

```text
touchless-docker-dashboard/
├── camera_streamer.py   # Async video streamer
├── hand_tracker.py      # MediaPipe landmarker & hand domain dataclasses
├── docker_service.py    # Docker SDK manager & mock fallback simulator
├── ui_manager.py        # Cyberpunk HUD graphics & particle portal drawing
├── main.py              # Main orchestrator loop
└── .gitignore           # Standard git ignore patterns
```

---

## ⚙️ Installation & Running

### 1. Prerequisites
Ensure you have Python 3.10+ installed.

### 2. Install Dependencies
```bash
pip install opencv-python mediapipe numpy
# Optional for real Docker integration:
pip install docker
```

### 3. Download the MediaPipe Model Asset
Download the **Hand Landmarker** model task file and place it in the project root directory:
* [Download hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

### 4. Run the Dashboard
```bash
python main.py
```

---

## 🖐️ Interactive Gestures

* **Move Pointer:** Hold up one hand and point with your index finger. Move your hand to move the neon green cursor.
* **Trigger Click:** Pinch your **Thumb and Index finger** together while hovering the cursor over a container action button (`START`, `STOP`, `RESTART`).
* **Open Magic Portal:** Hold up **both hands** in front of the camera to open the glowing edge-sparking portal.

---

## 🧑‍💻 Author
Developed by **Python Software Architect & DevOps Engineer** (Kongkiat).
