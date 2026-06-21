#!/usr/bin/env bash
# End-to-end smoke test: runs the full pipeline (stages 1-4, enhanced mode) on a
# SINGLE page to verify everything wires together. Stage 5 (serving) is an
# optional feature, not part of the pipeline, so it is not run here.
#
# This is a wiring check, not a quality check: with one page the metrics are
# meaningless. Artifacts are written under data/smoke/ so a real run's data/ is
# never touched. Requires a GPU and downloads the models on first use.
set -euo pipefail
cd "$(dirname "$0")/.."

# Build a temporary base config (max_pages=1, tiny QA set, redirected paths)
# without mutating or duplicating configs/base.yaml.
SMOKE_CFG="$(mktemp)"
trap 'rm -f "$SMOKE_CFG"' EXIT

python - "$SMOKE_CFG" << 'PY'
import sys, yaml
cfg = yaml.safe_load(open("configs/base.yaml"))
cfg["ingest"]["max_pages"] = 1
cfg["ingest"]["streaming"] = True  # stream one page; don't download 1.22 GB
cfg["eval"]["num_questions"] = 2
cfg["eval"]["judge_sample_size"] = 2
# Redirect every artifact path under data/smoke/ (already git-ignored via data/).
cfg["paths"] = {k: v.replace("data", "data/smoke", 1) for k, v in cfg["paths"].items()}
yaml.safe_dump(cfg, open(sys.argv[1], "w"))
PY

echo "== Stage 1: ingest + preprocess =="
python scripts/01_ingest_preprocess.py --config "$SMOKE_CFG" --mode enhanced
echo "== Stage 2: build index =="
python scripts/02_build_index.py --config "$SMOKE_CFG" --mode enhanced
echo "== Stage 3: build knowledge graph =="
python scripts/03_build_kg.py --config "$SMOKE_CFG" --mode enhanced
echo "== Stage 4: evaluate (baseline vs enhanced) =="
python scripts/04_run_eval.py --config "$SMOKE_CFG" --variants baseline,enhanced

echo "Smoke test passed."
