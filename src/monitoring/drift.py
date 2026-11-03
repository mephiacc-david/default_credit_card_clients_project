"""Отправка тестовых строк в API и расчёт PSI по скору."""

import argparse
import numpy as np
import pandas as pd
import requests

from src.features.build_features import RAW_FEATURES


def calculate_psi(reference, current, bins=10):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Для PSI нужны непустые выборки")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        raise ValueError("Недостаточно различных значений для расчёта PSI")
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts = np.histogram(reference, bins=edges)[0] / len(reference)
    current_counts = np.histogram(current, bins=edges)[0] / len(current)
    reference_counts = np.clip(reference_counts, 1e-6, None)
    current_counts = np.clip(current_counts, 1e-6, None)
    difference = current_counts - reference_counts
    ratio = current_counts / reference_counts
    return float(np.sum(difference * np.log(ratio)))


def simulate_requests(url="http://localhost:8000", sample_size=100):
    test_data = pd.read_csv("data/processed/test.csv")
    reference = pd.read_csv("reports/reference_probabilities.csv")["probability"]
    sample = test_data.sample(min(sample_size, len(test_data)), random_state=42)

    probabilities = []
    for _, row in sample.iterrows():
        payload = row[RAW_FEATURES].to_dict()
        response = requests.post(f"{url}/predict", json=payload, timeout=15)
        response.raise_for_status()
        probabilities.append(response.json()["default_probability"])

    psi = calculate_psi(reference, probabilities)
    print(f"Отправлено запросов: {len(probabilities)}")
    print(f"PSI по вероятности дефолта: {psi:.4f}")
    return psi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()
    simulate_requests(url=args.url, sample_size=args.sample_size)
