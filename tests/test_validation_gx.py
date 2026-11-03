"""Проверки валидации Great Expectations."""

import pytest

from src.data.download import TARGET
from src.data.validation_gx import validate_data


def test_gx_accepts_valid_data(sample_data):
    assert validate_data(sample_data, expected_rows=len(sample_data)) == 30


def test_gx_rejects_invalid_target(sample_data):
    sample_data.loc[0, TARGET] = 2
    with pytest.raises(ValueError, match="expect_column_values_to_be_in_set"):
        validate_data(sample_data, expected_rows=len(sample_data))
