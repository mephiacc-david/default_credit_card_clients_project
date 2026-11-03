import numpy as np

from src.monitoring.drift import calculate_psi


def test_psi_for_same_data_is_zero():
    sample = np.arange(1, 101)
    assert abs(calculate_psi(sample, sample)) < 1e-8


def test_psi_for_different_data_is_positive():
    first = np.arange(1, 101)
    second = np.arange(51, 151)
    assert calculate_psi(first, second) > 0
