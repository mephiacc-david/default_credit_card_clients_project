"""Проверка исходных данных с помощью Great Expectations."""

import argparse

import great_expectations as gx
import pandas as pd
from great_expectations.data_context.types.base import ProgressBarsConfig

from src.data.download import RAW_PATH, TARGET, UCI_COLUMN_NAMES


def validate_data(data: pd.DataFrame, expected_rows: int = 30000) -> int:
    """Проверить схему и ключевые ограничения; вернуть число проверок."""
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(globally=False)
    source = context.data_sources.add_pandas(name="credit_default")
    asset = source.add_dataframe_asset(name="raw_clients")
    batch_definition = asset.add_batch_definition_whole_dataframe("all_rows")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": data})

    expectations = [
        gx.expectations.ExpectTableRowCountToEqual(value=expected_rows),
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=UCI_COLUMN_NAMES + [TARGET], exact_match=False
        ),
    ]
    expectations.extend(
        gx.expectations.ExpectColumnValuesToNotBeNull(column=column)
        for column in UCI_COLUMN_NAMES + [TARGET]
    )
    expectations.extend(
        [
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=TARGET, value_set=[0, 1]
            ),
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="AGE", min_value=18, max_value=100
            ),
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="LIMIT_BAL", min_value=0, strict_min=True
            ),
            gx.expectations.ExpectColumnValuesToBeInSet(column="SEX", value_set=[1, 2]),
        ]
    )

    suite = gx.ExpectationSuite(name="credit_default_raw")
    for expectation in expectations:
        suite.add_expectation(expectation)

    result = batch.validate(suite)
    if not result.success:
        failures = [
            check.expectation_config.type
            for check in result.results
            if not check.success
        ]
        raise ValueError(
            "Great Expectations: не пройдены проверки: " + ", ".join(failures)
        )
    return len(expectations)


def validate_file(path=RAW_PATH) -> None:
    data = pd.read_csv(path)
    count = validate_data(data)
    print(f"Great Expectations: пройдено {count} проверок, строк: {len(data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(RAW_PATH))
    arguments = parser.parse_args()
    validate_file(arguments.path)
