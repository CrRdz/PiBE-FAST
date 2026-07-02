#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: scripts/run_local_video.sh samples/test_fall.mp4"
  exit 1
fi

python -m app.main --source "$1"

