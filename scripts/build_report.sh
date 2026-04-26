#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python report/build_tables.py
cd report && latexmk -pdf report.tex
