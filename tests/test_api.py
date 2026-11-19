from fastapi.testclient import TestClient

from serve import api


class StubModel:
    def predict(self, features):
        assert list(features.columns) == ["trip_distance", "total_amount"]
        assert features.iloc[0].to_dict() == {
            "trip_distance": 2.4,
            "total_amount": 18.5,
        }
        return [12.5]


def test_predict_uses_training_features(monkeypatch):
    model_uri = "models:/rfr-ride-duration@production"
    monkeypatch.setattr(api, "get_model_bundle", lambda: (model_uri, StubModel()))

    response = TestClient(api.app).post(
        "/predict",
        json={"trip_distance": 2.4, "total_amount": 18.5},
    )

    assert response.status_code == 200
    assert response.json() == {"prediction": 12.5, "model_uri": model_uri}


def test_predict_requires_both_model_features():
    response = TestClient(api.app).post("/predict", json={"trip_distance": 2.4})

    assert response.status_code == 422
