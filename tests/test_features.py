from src.features.build_features import create_features


def test_new_features(sample_data):
    features = create_features(sample_data)
    assert "payment_delay_mean" in features.columns
    assert "bill_amount_mean" in features.columns
    assert "age_group" in features.columns
    assert features["payment_delay_count"].between(0, 6).all()
    assert len(features) == len(sample_data)
