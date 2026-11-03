import numpy as np
from fastapi.testclient import TestClient

from src.api.app import app, get_model


class DummyModel:
    def predict_proba(self, data):
        return np.array([[0.25, 0.75]])


def test_predict_endpoint(sample_data):
    app.dependency_overrides[get_model] = lambda: DummyModel()
    try:
        client = TestClient(app)
        payload = (
            sample_data.drop(columns=["default.payment.next.month"]).iloc[0].to_dict()
        )
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["default_prediction"] == 1
        assert response.json()["default_probability"] == 0.75
    finally:
        app.dependency_overrides.clear()


def test_predict_with_trained_pipeline(sample_data):
    from src.data.download import TARGET
    from src.models.pipeline import create_pipeline

    features = sample_data.drop(columns=[TARGET])
    model = create_pipeline("logistic", C=0.5)
    model.fit(features, sample_data[TARGET])
    app.dependency_overrides[get_model] = lambda: model
    try:
        client = TestClient(app)
        response = client.post("/predict", json=features.iloc[0].to_dict())
        assert response.status_code == 200
        probability = response.json()["default_probability"]
        assert 0 <= probability <= 1
    finally:
        app.dependency_overrides.clear()


def test_wrong_age_returns_422(sample_data):
    app.dependency_overrides[get_model] = lambda: DummyModel()
    try:
        client = TestClient(app)
        payload = (
            sample_data.drop(columns=["default.payment.next.month"]).iloc[0].to_dict()
        )
        payload["AGE"] = 150
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
    finally:
        app.dependency_overrides.clear()
