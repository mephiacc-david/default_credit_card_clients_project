"""Простые дополнительные признаки по истории платежей."""

import numpy as np
import pandas as pd

from src.data.validation import BILL_COLUMNS, PAY_AMOUNT_COLUMNS, PAY_COLUMNS

NUMERIC_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    *PAY_COLUMNS,
    *BILL_COLUMNS,
    *PAY_AMOUNT_COLUMNS,
    "payment_delay_mean",
    "payment_delay_count",
    "bill_amount_mean",
    "payment_amount_mean",
]
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE", "age_group"]
RAW_FEATURES = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    *PAY_COLUMNS,
    *BILL_COLUMNS,
    *PAY_AMOUNT_COLUMNS,
]


def create_features(data):
    result = data.copy()
    result["payment_delay_mean"] = result[PAY_COLUMNS].mean(axis=1)
    result["payment_delay_count"] = (result[PAY_COLUMNS] > 0).sum(axis=1)
    result["bill_amount_mean"] = result[BILL_COLUMNS].mean(axis=1)
    result["payment_amount_mean"] = result[PAY_AMOUNT_COLUMNS].mean(axis=1)
    result["age_group"] = pd.cut(
        result["AGE"],
        bins=[17, 29, 44, 59, np.inf],
        labels=["18-29", "30-44", "45-59", "60+"],
    ).astype(str)
    return result
