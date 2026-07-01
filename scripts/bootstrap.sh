#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$ROOT/.venv}"
# Some managed Jupyter images force pip's user mode globally. Packages in a
# virtual environment must never use --user, so override that inherited setting.
export PIP_USER=0

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)'; then
  echo "Python 3.11, 3.12, or 3.13 is required." >&2
  exit 1
fi

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade "pip>=25" setuptools wheel

if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  "$VENV/bin/python" -m pip install --index-url "$TORCH_INDEX_URL" \
    "torch>=2.7,<3" "torchvision>=0.22,<1"
else
  DRIVER_VERSION=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
      | head -n 1 | tr -d '[:space:]')"
  fi
  DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
  if [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] && (( DRIVER_MAJOR < 580 )); then
    echo "NVIDIA driver $DRIVER_VERSION detected; installing CUDA 11.8-compatible PyTorch."
    "$VENV/bin/python" -m pip install --index-url \
      https://download.pytorch.org/whl/cu118 \
      "torch==2.7.1" "torchvision==0.22.1"
  else
    "$VENV/bin/python" -m pip install "torch>=2.7,<3" "torchvision>=0.22,<1"
  fi
fi
"$VENV/bin/python" -m pip install -e "$ROOT[gpu]"

"$VENV/bin/python" - <<'PY'
import torch
import transformers
print(f"Installed torch={torch.__version__}, transformers={transformers.__version__}")
if not torch.cuda.is_available():
    raise SystemExit(
        "PyTorch installed, but CUDA is unavailable. Set TORCH_INDEX_URL to the "
        "correct wheel index for this server and rerun bootstrap."
    )
print("CUDA:", torch.version.cuda)
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {props.name}, {props.total_memory / 1024**3:.1f} GiB")
PY

echo "Environment ready. Run: source \"$VENV/bin/activate\""
