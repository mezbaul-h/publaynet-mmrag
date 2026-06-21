#!/usr/bin/env bash
# Auto-fix lint violations and reformat in place. Needs the 'dev' extra.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== ruff check --fix =="
ruff check --fix .
echo "== ruff format =="
ruff format .
echo "Lint fixed."
