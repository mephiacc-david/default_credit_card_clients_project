"""Пять экспериментов, подбор параметров и сохранение модели."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV, train_test_split

from src.data.download import TARGET
from src.models.metrics import calculate_metrics
from src.models.pipeline import create_pipeline

REPORTS_DIR = Path("reports")
MODEL_PATH = Path("models/credit_default_model.joblib")

EXPERIMENTS = [
    ("logistic_C_0.1", "logistic", {"C": 0.1}),
    ("logistic_C_1", "logistic", {"C": 1.0}),
    (
        "logistic_balanced",
        "logistic",
        {"C": 1.0, "class_weight": "balanced"},
    ),
    (
        "random_forest",
        "random_forest",
        {"n_estimators": 80, "max_depth": 9, "min_samples_leaf": 8},
    ),
    (
        "gradient_boosting",
        "gradient_boosting",
        {"max_iter": 90, "learning_rate": 0.07, "max_leaf_nodes": 15},
    ),
]


def plot_roc(y_true, probabilities, output_path):
    display = RocCurveDisplay.from_predictions(y_true, probabilities)
    display.ax_.grid(alpha=0.3)
    display.figure_.tight_layout()
    display.figure_.savefig(output_path, dpi=130)
    plt.close(display.figure_)


def get_grid(model_name):
    if model_name == "logistic":
        return {"classifier__C": [0.1, 1.0, 3.0]}
    if model_name == "random_forest":
        return {"classifier__max_depth": [7, 11, None]}
    return {"classifier__max_leaf_nodes": [10, 15, 25]}


def run_training(
    train_path="data/processed/train.csv", test_path="data/processed/test.csv"
):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    x_train = train_data.drop(columns=[TARGET])
    y_train = train_data[TARGET]
    x_test = test_data.drop(columns=[TARGET])
    y_test = test_data[TARGET]

    x_fit, x_valid, y_fit, y_valid = train_test_split(
        x_train, y_train, test_size=0.25, stratify=y_train, random_state=42
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(Path("mlruns").resolve().as_uri())
    mlflow.set_experiment("credit_default_scoring")

    experiment_rows = []
    for experiment_name, model_name, parameters in EXPERIMENTS:
        print(f"Обучаю {experiment_name}")
        pipeline = create_pipeline(model_name, **parameters)
        with mlflow.start_run(run_name=experiment_name):
            mlflow.log_param("algorithm", model_name)
            mlflow.log_params({f"model_{key}": val for key, val in parameters.items()})
            pipeline.fit(x_fit, y_fit)
            probabilities = pipeline.predict_proba(x_valid)[:, 1]
            metrics = calculate_metrics(y_valid, probabilities)
            mlflow.log_metrics({f"valid_{key}": val for key, val in metrics.items()})
            roc_path = REPORTS_DIR / "validation_roc.png"
            plot_roc(y_valid, probabilities, roc_path)
            mlflow.log_artifact(str(roc_path))
            mlflow.sklearn.log_model(pipeline, name="model")
            experiment_rows.append(
                {"experiment": experiment_name, "algorithm": model_name, **metrics}
            )

    results = pd.DataFrame(experiment_rows)
    results = results.sort_values("roc_auc", ascending=False)
    results.to_csv(REPORTS_DIR / "experiments.csv", index=False)

    best_name = results.iloc[0]["experiment"]
    best_experiment = next(item for item in EXPERIMENTS if item[0] == best_name)
    _, best_algorithm, best_parameters = best_experiment

    print(f"Подбираю параметры для {best_name}")
    search = GridSearchCV(
        estimator=create_pipeline(best_algorithm, **best_parameters),
        param_grid=get_grid(best_algorithm),
        cv=3,
        scoring="roc_auc",
        n_jobs=2,
        refit=True,
    )

    with mlflow.start_run(run_name="final_grid_search"):
        search.fit(x_train, y_train)
        model = search.best_estimator_
        probabilities = model.predict_proba(x_test)[:, 1]
        test_metrics = calculate_metrics(y_test, probabilities)

        mlflow.log_param("algorithm", best_algorithm)
        mlflow.log_params(
            {f"best_{key}": value for key, value in search.best_params_.items()}
        )
        mlflow.log_metric("cv_roc_auc", float(search.best_score_))
        mlflow.log_metrics({f"test_{key}": val for key, val in test_metrics.items()})

        plot_roc(y_test, probabilities, REPORTS_DIR / "roc_curve.png")
        mlflow.log_artifact(str(REPORTS_DIR / "roc_curve.png"))
        mlflow.log_artifact(str(REPORTS_DIR / "experiments.csv"))
        mlflow.sklearn.log_model(model, name="model")

    joblib.dump(model, MODEL_PATH)
    reference = model.predict_proba(x_train)[:, 1]
    pd.DataFrame({"probability": reference}).to_csv(
        REPORTS_DIR / "reference_probabilities.csv", index=False
    )
    output_metrics = {
        **test_metrics,
        "best_experiment": best_name,
        "best_algorithm": best_algorithm,
        "cv_roc_auc": float(search.best_score_),
        "best_params": search.best_params_,
        "test_rows": int(len(x_test)),
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(output_metrics, file, indent=2, ensure_ascii=False)
    print("Тестовые метрики:", test_metrics)
    return results, output_metrics


if __name__ == "__main__":
    run_training()
