#!/usr/bin/env bash
set -euo pipefail

python -m src.data.download
python -m src.data.validation

dvc add data/raw/credit_default.csv
dvc repro
