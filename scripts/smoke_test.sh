#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m pytest -q
tempbench --models mistral-7b-v0.3-instruct \
  --prompts contradiction_repair plan
tempbench --models mistral-7b-v0.3-instruct preflight --remote

echo "Smoke test passed. This verifies configuration/access, not a generation."

