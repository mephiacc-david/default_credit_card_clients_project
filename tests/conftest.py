import numpy as np
import pandas as pd
import pytest

from src.data.download import TARGET
from src.features.build_features import RAW_FEATURES


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    rows = 50
    data = pd.DataFrame(index=range(rows))
    data["LIMIT_BAL"] = rng.integers(10000, 500000, rows)
    data["SEX"] = rng.integers(1, 3, rows)
    data["EDUCATION"] = rng.integers(1, 5, rows)
    data["MARRIAGE"] = rng.integers(1, 4, rows)
    data["AGE"] = rng.integers(21, 70, rows)
    for name in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        data[name] = rng.integers(-2, 5, rows)
    for index in range(1, 7):
        data[f"BILL_AMT{index}"] = rng.integers(-10000, 100000, rows)
        data[f"PAY_AMT{index}"] = rng.integers(0, 20000, rows)
    data[TARGET] = (data["PAY_0"] > 0).astype(int)
    return data[RAW_FEATURES + [TARGET]]
