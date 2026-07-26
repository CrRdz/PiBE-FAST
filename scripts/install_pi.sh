#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv ffmpeg libatlas-base-dev

python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m pip install tflite-runtime || {
  echo "tflite-runtime wheel was not available for this platform."
  echo "Install TensorFlow Lite runtime manually for your Raspberry Pi OS/Python version."
}

mkdir -p data/clips data/keypoints data/snapshots models samples
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install_speech_pi.sh" --skip-apt-update
echo "Done. Put movenet_lightning.tflite in models/ before running the service."
