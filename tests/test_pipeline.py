from src.data.download import TARGET
from src.models.pipeline import create_pipeline


def test_pipeline_trains_and_predicts(sample_data):
    model = create_pipeline("logistic", C=0.5)
    features = sample_data.drop(columns=[TARGET])
    labels = sample_data[TARGET]
    model.fit(features, labels)
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (len(sample_data), 2)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
