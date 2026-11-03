import pytest
from pandera.errors import SchemaErrors

from src.data.validation import validate_data


def test_valid_data_passes(sample_data):
    validated_data = validate_data(sample_data)
    assert len(validated_data) == len(sample_data)


def test_wrong_age_fails(sample_data):
    sample_data.loc[0, "AGE"] = 150
    with pytest.raises(SchemaErrors):
        validate_data(sample_data)


def test_missing_value_fails(sample_data):
    sample_data.loc[0, "AGE"] = None
    with pytest.raises(SchemaErrors):
        validate_data(sample_data)


def test_missing_column_fails(sample_data):
    sample_data = sample_data.drop(columns=["PAY_AMT1"])
    with pytest.raises(SchemaErrors):
        validate_data(sample_data)
