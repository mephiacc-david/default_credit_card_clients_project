"""Очистка и разделение датасета на train и test."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.download import RAW_PATH, TARGET
from src.data.validation import validate_data

PROCESSED_DIR = Path("data/processed")


def clean_data(data):
    data = data.drop(columns=["ID"], errors="ignore").copy()
    data["EDUCATION"] = data["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    data["MARRIAGE"] = data["MARRIAGE"].replace({0: 3})
    return data


def prepare_data(raw_path=RAW_PATH, output_dir=PROCESSED_DIR):
    raw_data = pd.read_csv(raw_path)
    validate_data(raw_data)
    data = clean_data(raw_data)
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data[TARGET]
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(output_dir / "train.csv", index=False)
    test_data.to_csv(output_dir / "test.csv", index=False)
    print(f"Train: {len(train_data)}, test: {len(test_data)}")
    return train_data, test_data


if __name__ == "__main__":
    prepare_data()
