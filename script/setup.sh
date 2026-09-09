#!/usr/bin/env bash
set -euo pipefail

STEGF_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STEGF_REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "[STEGF] conda was not found in PATH." >&2
  exit 1
fi

STEGF_CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$STEGF_CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq STEGF; then
  echo "[STEGF] Updating existing Conda environment: STEGF"
  conda env update --name STEGF --file script/environment.yml
else
  echo "[STEGF] Creating Conda environment: STEGF"
  conda env create --file script/environment.yml
fi

conda activate STEGF

echo "[STEGF] Installing PyTorch 2.0.0 with CUDA 11.8 runtime"
python -m pip install \
  torch==2.0.0 \
  torchvision==0.15.0 \
  torchaudio==2.0.0 \
  --index-url https://download.pytorch.org/whl/cu118

echo "[STEGF] Installing Python dependencies"
python -m pip install \
  opencv-python \
  natsort \
  scipy \
  kornia \
  plyfile==0.8.1 \
  tqdm \
  scikit-image

if ! command -v nvcc >/dev/null 2>&1; then
  echo "[STEGF] nvcc was not found in PATH; a CUDA compiler is required to build the local extensions." >&2
  exit 1
fi

echo "[STEGF] Building local CUDA extensions"
python -m pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9
python -m pip install thirdparty/gaussian_splatting/submodules/simple-knn
python -m pip install -e thirdparty/mmcv -v

python - <<'PY'
import sys
from importlib.metadata import version

import torch
import torchaudio
import torchvision
from diff_gaussian_rasterization_ch9 import GaussianRasterizer
from mmcv.ops import knn
from simple_knn._C import distCUDA2

expected_versions = {
    "torch": "2.0.0",
    "torchvision": "0.15.0",
    "torchaudio": "2.0.0",
    "plyfile": "0.8.1",
}
actual_versions = {
    "torch": torch.__version__.split("+", 1)[0],
    "torchvision": torchvision.__version__.split("+", 1)[0],
    "torchaudio": torchaudio.__version__.split("+", 1)[0],
    "plyfile": version("plyfile"),
}
if sys.version_info[:2] != (3, 8):
    raise RuntimeError(f"Expected Python 3.8, got {sys.version.split()[0]}")
if actual_versions != expected_versions:
    raise RuntimeError(
        f"Unexpected package versions: expected={expected_versions}, "
        f"actual={actual_versions}"
    )
if torch.version.cuda != "11.8":
    raise RuntimeError(
        f"Expected the PyTorch CUDA 11.8 build, got {torch.version.cuda}"
    )

print(
    "[STEGF] Python environment ready: "
    f"torch={torch.__version__}, "
    f"torchvision={torchvision.__version__}, "
    f"torchaudio={torchaudio.__version__}, "
    f"plyfile={version('plyfile')}, "
    f"cuda={torch.version.cuda}"
)
print("[STEGF] CUDA extensions imported successfully.")
PY
