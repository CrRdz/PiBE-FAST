# Fall Detection Pi

Real-time human pose and fall detection for Raspberry Pi.

The first version uses TensorFlow Lite + MoveNet SinglePose Lightning to extract
17 body keypoints from each frame. The service then classifies the pose with
local rules (`standing`, `sitting`, `lying`) and uses a temporal state machine to
detect likely falls. A fall is not the same as lying down: the detector looks for
an upright pose, a fast center drop, rapid torso rotation, and sustained lying.

## Project Layout

```text
app/
  main.py                 # Service entry point
  camera.py               # Camera and local video frame sources
  movenet.py              # TFLite MoveNet inference
  pose_classifier.py      # standing / sitting / lying rules
  fall_detector.py        # Fall state machine
  web.py                  # MJPEG preview and status API
  drawing.py              # Skeleton overlay
  config.py               # Tunable thresholds
models/
  movenet_lightning.tflite
data/
  clips/
  keypoints/
  snapshots/
scripts/
  install_pi.sh
  run_local_video.sh
tests/
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install one TFLite interpreter:

```bash
# Local development
python -m pip install tensorflow

# Raspberry Pi, when a wheel is available for your OS/Python version
python -m pip install tflite-runtime
```

Download the MoveNet SinglePose Lightning TFLite model from TensorFlow Hub or
the official TensorFlow examples and save it as:

```text
models/movenet_lightning.tflite
```

## Run With Local Videos

The first development phase should use local videos before connecting the camera:

```bash
python -m app.main --source samples/test_standing.mp4
python -m app.main --source samples/test_sitting.mp4
python -m app.main --source samples/test_lying.mp4
python -m app.main --source samples/test_fall.mp4
```

Open the web preview at:

```text
http://localhost:8080
```

For a quick command wrapper:

```bash
scripts/run_local_video.sh samples/test_fall.mp4
```

## Run With Camera

USB or OpenCV-compatible camera:

```bash
python -m app.main --source camera
```

Raspberry Pi camera via Picamera2:

```bash
python -m app.main --source camera --camera-backend picamera2
```

When using SSH:

```bash
ssh -L 8080:localhost:8080 pi@raspberrypi.local
```

Then open:

```text
http://localhost:8080
```

## Logs and Event Clips

Keypoints are written as JSONL under `data/keypoints/` by default. Each line has
the timestamp, pose, fall flag, quality, metrics, and all keypoints:

```json
{"ts":1719912001.23,"pose":"standing","fall":false,"quality":0.82,"keypoints":[{"name":"left_shoulder","x":0.42,"y":0.31,"score":0.88}]}
```

Video is not saved continuously. To save MP4 event clips only when a likely fall
is detected:

```bash
python -m app.main --source camera --save-event-clips
```

The recorder keeps about 5 seconds before and 5 seconds after the fall event.
OpenCV codec support differs by platform; this first version writes MP4 with the
`mp4v` codec.

## Tune Thresholds

The first-pass rules live in `app/config.py`:

- `PoseClassifierConfig`: keypoint confidence, body aspect ratios, torso angles,
  sitting knee/hip geometry.
- `FallDetectorConfig`: transition window, center drop threshold, torso angle
  change, lying hold duration, recovery duration.

Camera placement strongly affects these values. Start with logs from controlled
samples: standing 30s, sitting 30s, lying 30s, slow lying down 30s, and several
simulated falls.

## Tests

```bash
python -m unittest discover -s tests -v
```

The current tests cover the rule classifier and the temporal fall detector. They
do not require OpenCV, TensorFlow, a camera, or the model file.

## Git Hygiene

Do not commit real videos, privacy-sensitive snapshots, `.env`, or `.venv`.
Generated clips, snapshots, keypoint logs, and common image/video files are
ignored by `.gitignore`.

