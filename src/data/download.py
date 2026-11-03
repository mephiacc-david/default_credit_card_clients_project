"""Загрузка открытого датасета UCI (идентификатор 350)."""

from pathlib import Path

import pandas as pd

TARGET = "default.payment.next.month"
RAW_PATH = Path("data/raw/credit_default.csv")
UCI_COLUMN_NAMES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5",
    "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4",
    "PAY_AMT5", "PAY_AMT6",
]
MIRROR_URL = (
    "https://huggingface.co/datasets/scikit-learn/credit-card-clients/"
    "resolve/main/UCI_Credit_Card.csv"
)


def download_dataset(output_path=RAW_PATH):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return output_path

    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=350)
        data = dataset.data.features.copy()
        column_mapping = {
            f"X{index}": name
            for index, name in enumerate(UCI_COLUMN_NAMES, start=1)
        }
        data = data.rename(columns=column_mapping)
        data[TARGET] = dataset.data.targets.iloc[:, 0].to_numpy()
    except Exception as error:
        print(f"UCI недоступен: {error}")
        print("Загружаю копию датасета из Hugging Face")
        data = pd.read_csv(MIRROR_URL)

    if (
        len(data) != 30000
        or TARGET not in data.columns
        or not set(UCI_COLUMN_NAMES).issubset(data.columns)
    ):
        raise ValueError("Формат датасета отличается от ожидаемого")

    data.to_csv(output_path, index=False)
    print(f"Сохранил {len(data)} строк в {output_path}")
    return output_path


if __name__ == "__main__":
    download_dataset()
