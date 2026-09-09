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

python -m pip install thirdparty/gaussian_splatting/submodules/gaussian_rasterization_ch9
python -m pip install thirdparty/gaussian_splatting/submodules/simple-knn
python -m pip install -e thirdparty/mmcv -v

python - <<'PY'
import torch
import torchvision
from diff_gaussian_rasterization_ch9 import GaussianRasterizer
from mmcv.ops import knn
from simple_knn._C import distCUDA2

print(f"[STEGF] Python environment ready: torch={torch.__version__}, torchvision={torchvision.__version__}, cuda={torch.version.cuda}")
print("[STEGF] CUDA extensions imported successfully.")
PY
