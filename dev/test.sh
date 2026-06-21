#!/usr/bin/env bash
# Run the unit-test suite (pure-logic tests; no GPU or model downloads needed).
# Needs the 'dev' extra (pip install -e ".[dev]").
set -euo pipefail
cd "$(dirname "$0")/.."
pytest "$@"
