"""Sklearn Pipeline с предобработкой и моделью."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    create_features,
)


def create_pipeline(model_name="logistic", **parameters):
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    if model_name == "logistic":
        model = LogisticRegression(max_iter=800, random_state=42, **parameters)
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **parameters)
    elif model_name == "gradient_boosting":
        model = HistGradientBoostingClassifier(random_state=42, **parameters)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    return Pipeline(
        [
            ("features", FunctionTransformer(create_features, validate=False)),
            ("preprocessing", preprocessing),
            ("classifier", model),
        ]
    )
