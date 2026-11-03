"""Один пример запроса к /predict."""

import pandas as pd
import requests

from src.data.download import TARGET
from src.features.build_features import RAW_FEATURES

sample = pd.read_csv("data/processed/test.csv").iloc[0]
payload = sample[RAW_FEATURES].to_dict()
result = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
result.raise_for_status()
print(result.json())
