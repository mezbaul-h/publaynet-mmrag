#!/usr/bin/env bash
# Check lint and formatting without modifying files. Needs the 'dev' extra
# (pip install -e ".[dev]"). Exits non-zero on any violation.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== ruff check =="
ruff check .
echo "== ruff format --check =="
ruff format --check .
echo "Lint OK."
