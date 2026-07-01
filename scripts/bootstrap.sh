#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$ROOT/.venv}"

if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)'; then
  echo "Python 3.11, 3.12, or 3.13 is required." >&2
  exit 1
fi

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade "pip>=25" setuptools wheel

if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  "$VENV/bin/python" -m pip install --index-url "$TORCH_INDEX_URL" \
    "torch>=2.8,<3" "torchvision>=0.23,<1"
  "$VENV/bin/python" -m pip install -e "$ROOT[gpu]"
else
  "$VENV/bin/python" -m pip install -e "$ROOT[gpu]"
fi

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

