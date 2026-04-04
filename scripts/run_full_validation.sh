#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate || source .venv/Scripts/activate

pip install -U pip
pip install -e ".[dev,vision,tui,neat,umap]"

pytest tests/ -v

eden train --dataset synth --epochs 5 --dry-run

eden tui --dataset fashion_mnist || true

eden benchmark --all --seeds 1 --epochs 2 --output eden_results/full --quick

eden ablation --mechanism contact_inhibition --seeds 1 --epochs 2

eden sensitivity --param tau --range 0.8,1.2

echo "Validation complete. See eden_results/"
