#!/usr/bin/env bash
set -euo pipefail

STEGF_PRE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STEGF_PRE_REPO_ROOT"

STEGF_PRE_WITH_MIDAS=1
STEGF_PRE_PROFILE="local"
while (( $# > 0 )); do
  case "$1" in
    --profile)
      if (( $# < 2 )); then
        echo "[STEGF] --profile requires local or server" >&2
        exit 2
      fi
      STEGF_PRE_PROFILE="$2"
      shift 2
      ;;
    --without-midas)
      STEGF_PRE_WITH_MIDAS=0
      shift
      ;;
    -h|--help)
      echo "Usage: bash script/setup_preprocess.sh [--profile local|server] [--without-midas]"
      exit 0
      ;;
    *)
      echo "[STEGF] Unknown setup argument: $1" >&2
      echo "Usage: bash script/setup_preprocess.sh [--profile local|server] [--without-midas]" >&2
      exit 2
      ;;
  esac
done

case "$STEGF_PRE_PROFILE" in
  local)
    STEGF_PRE_ENV_FILE="script/preprocess_environment.yml"
    ;;
  server)
    STEGF_PRE_ENV_FILE="script/preprocess_environment_server.yml"
    ;;
  *)
    echo "[STEGF] Unknown preprocessing profile: $STEGF_PRE_PROFILE" >&2
    echo "[STEGF] Expected local or server" >&2
    exit 2
    ;;
esac

if [[ -z "${DISPLAY:-}" ]]; then
  if command -v xvfb-run >/dev/null 2>&1; then
    echo "[STEGF] Headless host ready: xvfb-run is available"
  else
    echo "[STEGF] Headless host detected: install xvfb and xauth if COLMAP requires a virtual display"
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[STEGF] conda was not found in PATH." >&2
  exit 1
fi

STEGF_PRE_CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$STEGF_PRE_CONDA_BASE/etc/profile.d/conda.sh"

run_conda_env_with_retries() {
  local attempt=1
  local max_attempts=3

  while true; do
    if CONDA_REPORT_ERRORS=false \
       CONDA_REMOTE_CONNECT_TIMEOUT_SECS=30 \
       CONDA_REMOTE_READ_TIMEOUT_SECS=120 \
       CONDA_REMOTE_MAX_RETRIES=5 \
       "$@"; then
      return 0
    fi

    if (( attempt >= max_attempts )); then
      echo "[STEGF] Conda environment setup failed after $max_attempts attempts" >&2
      return 1
    fi

    echo "[STEGF] Conda index download failed; clearing the index cache and retrying ($((attempt + 1))/$max_attempts)"
    conda clean --index-cache --yes >/dev/null 2>&1 || true
    attempt=$((attempt + 1))
  done
}

if conda env list | awk '{print $1}' | grep -Fxq STEGF-preprocess; then
  echo "[STEGF] Updating existing preprocessing environment: STEGF-preprocess (profile=$STEGF_PRE_PROFILE)"
  run_conda_env_with_retries \
    conda env update --name STEGF-preprocess --file "$STEGF_PRE_ENV_FILE"
else
  echo "[STEGF] Creating preprocessing environment: STEGF-preprocess (profile=$STEGF_PRE_PROFILE)"
  run_conda_env_with_retries \
    conda env create --file "$STEGF_PRE_ENV_FILE"
fi

conda activate STEGF-preprocess

echo "[STEGF] Installing preprocessing dependencies"
python -m pip install \
  opencv-python-headless==4.8.1.78 \
  open3d==0.17.0 \
  tqdm==4.66.5

if [[ "$STEGF_PRE_WITH_MIDAS" -eq 0 ]]; then
  echo "[STEGF] Skipping optional MiDaS dependencies and model download"
else
  echo "[STEGF] Installing CUDA PyTorch and MiDaS dependencies"
  if [[ "$STEGF_PRE_PROFILE" == "server" ]]; then
    python -m pip install \
      torch==2.0.0 \
      torchvision==0.15.0 \
      torchaudio==2.0.0 \
      --index-url https://download.pytorch.org/whl/cu118
  else
    python -m pip install \
      torch==2.7.1 \
      torchvision==0.22.1 \
      --index-url https://download.pytorch.org/whl/cu128
  fi
  python -m pip install \
    timm==0.6.12 \
    einops==0.6.1 \
    imutils==0.5.4

  STEGF_PRE_MIDAS_DIR="$STEGF_PRE_REPO_ROOT/thirdparty/MiDaS"
  STEGF_PRE_MIDAS_COMMIT="454597711a62eabcbf7d1e89f3fb9f569051ac9b"
  if [[ ! -d "$STEGF_PRE_MIDAS_DIR" ]]; then
    echo "[STEGF] Cloning the pinned MiDaS source"
    git clone https://github.com/isl-org/MiDaS.git "$STEGF_PRE_MIDAS_DIR"
    git -C "$STEGF_PRE_MIDAS_DIR" checkout "$STEGF_PRE_MIDAS_COMMIT"
  elif [[ ! -f "$STEGF_PRE_MIDAS_DIR/midas/model_loader.py" ]]; then
    echo "[STEGF] Existing path is not a MiDaS checkout: $STEGF_PRE_MIDAS_DIR" >&2
    exit 1
  elif [[ -d "$STEGF_PRE_MIDAS_DIR/.git" ]] && \
       [[ "$(git -C "$STEGF_PRE_MIDAS_DIR" rev-parse HEAD)" != "$STEGF_PRE_MIDAS_COMMIT" ]]; then
    echo "[STEGF] Existing MiDaS checkout is not at the pinned commit: $STEGF_PRE_MIDAS_DIR" >&2
    exit 1
  else
    echo "[STEGF] Using existing MiDaS checkout: $STEGF_PRE_MIDAS_DIR"
  fi

  STEGF_PRE_WEIGHT_DIR="$STEGF_PRE_MIDAS_DIR/weights"
  STEGF_PRE_WEIGHT_PATH="$STEGF_PRE_WEIGHT_DIR/dpt_beit_large_512.pt"
  mkdir -p "$STEGF_PRE_WEIGHT_DIR"
  if [[ ! -s "$STEGF_PRE_WEIGHT_PATH" ]]; then
    echo "[STEGF] Downloading MiDaS dpt_beit_large_512 weights"
    STEGF_PRE_WEIGHT_URL="https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt"
    python - "$STEGF_PRE_WEIGHT_URL" "$STEGF_PRE_WEIGHT_PATH" <<'PY'
import sys
import urllib.request

url, destination = sys.argv[1:]
temporary = destination + ".downloading"
urllib.request.urlretrieve(url, temporary)
with open(temporary, "rb") as source:
    if source.read(2) != b"PK":
        raise RuntimeError("Downloaded MiDaS weights are not a PyTorch ZIP checkpoint")
import os
os.replace(temporary, destination)
PY
  else
    echo "[STEGF] MiDaS weights already exist: $STEGF_PRE_WEIGHT_PATH"
  fi
fi

python - <<'PY'
import shutil
import subprocess

import cv2
import numpy
import open3d

colmap = shutil.which("colmap")
if colmap is None:
    raise RuntimeError("COLMAP was not installed into STEGF-preprocess")
version_result = subprocess.run(
    [colmap, "-h"],
    capture_output=True,
    text=True,
)
if version_result.returncode != 0:
    detail = version_result.stderr.strip() or version_result.stdout.strip()
    raise RuntimeError(
        f"COLMAP could not start (exit {version_result.returncode}): {detail}"
    )
colmap_version = version_result.stdout.splitlines()[0]
print(
    "[STEGF] Preprocessing environment ready: "
    f"numpy={numpy.__version__}, opencv={cv2.__version__}, "
    f"open3d={open3d.__version__}, {colmap_version}, colmap={colmap}"
)
PY

if [[ "$STEGF_PRE_WITH_MIDAS" -eq 1 ]]; then
  python - <<'PY'
from importlib.metadata import version

import torch
import torchvision

cuda_summary = "cuda=unavailable"
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    required_arch = f"sm_{capability[0]}{capability[1]}"
    supported_arches = set(torch.cuda.get_arch_list())
    if required_arch not in supported_arches:
        raise RuntimeError(
            f"PyTorch {torch.__version__} does not support {required_arch} "
            f"({torch.cuda.get_device_name()})"
        )
    cuda_summary = f"gpu={torch.cuda.get_device_name()}, arch={required_arch}"

print(
    "[STEGF] MiDaS environment ready: "
    f"torch={torch.__version__}, torchvision={torchvision.__version__}, "
    f"timm={version('timm')}, {cuda_summary}"
)
PY
fi
