#!/usr/bin/env bash
set -euo pipefail

if [ ! -f models/credit_default_model.joblib ]; then
    echo 'Сначала выполните dvc repro, чтобы получить модель'
    exit 1
fi

docker build -t credit-scoring-api .
docker run --rm -p 8000:8000 credit-scoring-api
