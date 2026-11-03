"""Проверки входного датасета через Pandera."""

import argparse

import pandas as pd

from src.data.download import RAW_PATH, TARGET

PAY_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMOUNT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]


def create_schema():
    import pandera.pandas as pa

    columns = {
        "LIMIT_BAL": pa.Column(float, pa.Check.gt(0), nullable=False, coerce=True),
        "SEX": pa.Column(int, pa.Check.isin([1, 2]), nullable=False),
        "EDUCATION": pa.Column(int, pa.Check.in_range(0, 6), nullable=False),
        "MARRIAGE": pa.Column(int, pa.Check.in_range(0, 3), nullable=False),
        "AGE": pa.Column(int, pa.Check.in_range(18, 100), nullable=False),
        TARGET: pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    }
    for column in PAY_COLUMNS:
        columns[column] = pa.Column(
            int, pa.Check.in_range(-2, 9), nullable=False
        )
    for column in BILL_COLUMNS:
        columns[column] = pa.Column(float, nullable=False, coerce=True)
    for column in PAY_AMOUNT_COLUMNS:
        columns[column] = pa.Column(
            float, pa.Check.ge(0), nullable=False, coerce=True
        )
    return pa.DataFrameSchema(columns, strict=False)


def validate_data(data):
    return create_schema().validate(data, lazy=True)


def validate_file(path=RAW_PATH):
    data = pd.read_csv(path)
    validate_data(data)
    print(f"Проверка Pandera пройдена: {len(data)} строк")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(RAW_PATH))
    arguments = parser.parse_args()
    validate_file(arguments.path)
