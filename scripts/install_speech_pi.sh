#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIBEFAST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WHISPER_ROOT="${PIBEFAST_WHISPER_ROOT:-/home/pi/whisper.cpp}"
WHISPER_CLI="${WHISPER_ROOT}/build/bin/whisper-cli"
MODEL_SOURCE="${WHISPER_ROOT}/models/ggml-base.bin"
MODEL_TARGET="${PIBEFAST_ROOT}/models/ggml-base.bin"
MODEL_SHA1="465707469ff3a37a2b9b8d8f89f2f99de7299dac"

if [[ "${1:-}" != "--skip-apt-update" ]]; then
  sudo apt update
fi
sudo apt install -y alsa-utils cmake build-essential git

if [[ -d "${WHISPER_ROOT}/.git" ]]; then
  git -C "${WHISPER_ROOT}" pull --ff-only
elif [[ -e "${WHISPER_ROOT}" ]]; then
  echo "Cannot install whisper.cpp: ${WHISPER_ROOT} exists but is not a Git checkout." >&2
  exit 1
else
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "${WHISPER_ROOT}"
fi

cmake -S "${WHISPER_ROOT}" -B "${WHISPER_ROOT}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${WHISPER_ROOT}/build" --config Release -j2
"${WHISPER_CLI}" --help >/dev/null

mkdir -p "${PIBEFAST_ROOT}/models"
if [[ ! -f "${MODEL_TARGET}" ]]; then
  "${WHISPER_ROOT}/models/download-ggml-model.sh" base
  install -m 0644 "${MODEL_SOURCE}" "${MODEL_TARGET}"
fi
echo "${MODEL_SHA1}  ${MODEL_TARGET}" | sha1sum --check -

if command -v whisper-cli >/dev/null 2>&1; then
  echo "whisper-cli: $(command -v whisper-cli)"
elif [[ ! -e /usr/local/bin/whisper-cli && ! -L /usr/local/bin/whisper-cli ]]; then
  sudo ln -s "${WHISPER_CLI}" /usr/local/bin/whisper-cli
  echo "whisper-cli: /usr/local/bin/whisper-cli"
else
  echo "warning: /usr/local/bin/whisper-cli already exists; use ${WHISPER_CLI}" >&2
fi

echo
echo "ALSA capture devices:"
if ! arecord -l; then
  echo "warning: ALSA is installed, but no capture device is currently detected." >&2
  echo "Connect a USB/I2S microphone, then run: arecord -l" >&2
fi

echo
echo "Speech dependencies are ready."
echo "Microphone test:"
echo "  arecord -D default -f S16_LE -r 16000 -c 1 -d 3 /tmp/pibefast-mic-test.wav"
echo "Service options:"
echo "  --speech-device default --whisper-cli ${WHISPER_CLI} --speech-model ${MODEL_TARGET}"
