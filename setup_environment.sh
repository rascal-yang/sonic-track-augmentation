#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SIM_PYTHON="${SIM_PYTHON:-python3.11}"
if ! command -v ffprobe >/dev/null || ! command -v ffmpeg >/dev/null; then
    printf '%s\n' 'Install system ffmpeg first: sudo apt-get install ffmpeg' >&2
    exit 2
fi
"$SIM_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"'
"$SIM_PYTHON" -m venv "$PROJECT_DIR/.venv"
SIM_PIP_PYTHON="$PROJECT_DIR/.venv/bin/python"
"$SIM_PIP_PYTHON" -m pip install --upgrade pip
"$SIM_PIP_PYTHON" -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
"$SIM_PIP_PYTHON" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
"$SIM_PIP_PYTHON" -m pip install -e "$PROJECT_DIR/dependencies/IsaacLab/source/isaaclab"
"$SIM_PIP_PYTHON" -m pip install -e "$PROJECT_DIR/dependencies/IsaacLab/source/isaaclab_assets"
"$SIM_PIP_PYTHON" -m pip install -r "$PROJECT_DIR/requirements-runtime.txt"
"$SIM_PIP_PYTHON" "$PROJECT_DIR/launch.py" doctor
