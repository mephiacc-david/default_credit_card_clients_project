import numpy as np

from src.models.metrics import calculate_metrics


def test_calculate_metrics():
    true_labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = calculate_metrics(true_labels, probabilities)
    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
