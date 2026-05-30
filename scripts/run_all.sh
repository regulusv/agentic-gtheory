#!/usr/bin/env bash
# Quickstart: synthetic data -> G-Study -> D-Study -> figures + Table 1.
# Reproduces the full pipeline WITHOUT real ToolBench data or judge API calls.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-config.yaml}"

echo "==> [1/2] generating synthetic data"
python scripts/make_synthetic.py --config "$CONFIG"

echo "==> [2/2] running analysis pipeline"
python scripts/run_pipeline.py --config "$CONFIG"

echo "==> done. See outputs/ for table_1.tex, fig1_domain_variance.png, fig2_dstudy.png"
