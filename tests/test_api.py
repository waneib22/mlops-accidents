from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


SAMPLE_FEATURES = {
    "place": 10, "catu": 3, "sexe": 1, "secu1": 0.0,
    "year_acc": 2021, "victim_age": 60, "catv": 2, "obsm": 1,
    "motor": 1, "catr": 3, "circ": 2, "surf": 1, "situ": 1,
    "vma": 50, "jour": 7, "mois": 12, "lum": 5, "dep": 77,
    "com": 77317, "agg_": 2, "int": 1, "atm": 0, "col": 6,
    "lat": 48.60, "long": 2.89, "hour": 17,
    "nb_victim": 2, "nb_vehicules": 1,
}


@pytest.fixture
def client_with_model():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    mock_model.get_params.return_value = {"n_estimators": 100, "random_state": 42}

    import src.api.metrics as state
    from src.api.main import app
    state.ml_model["classifier"] = mock_model
    with TestClient(app) as c:
        state.ml_model["classifier"] = mock_model
        yield c


@pytest.fixture
def client_without_model():
    import src.api.metrics as state
    from src.api.main import app
    state.ml_model["classifier"] = None
    with TestClient(app) as c:
        state.ml_model["classifier"] = None
        yield c


class TestRootEndpoint:
    def test_root_redirects_to_docs(self, client_with_model):
        response = client_with_model.get("/", follow_redirects=False)
        assert response.status_code in [307, 308]
        assert "/docs" in response.headers["location"]


class TestHealthEndpoint:
    def test_health_ok_when_model_loaded(self, client_with_model):
        response = client_with_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["model_type"] is not None
        assert data["n_features"] == 28
        assert data["uptime_seconds"] >= 0
        assert data["api_version"] == "1.0.0"

    def test_health_degraded_when_no_model(self, client_without_model):
        response = client_without_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False
        assert data["model_type"] is None
        assert data["n_features"] == 28


class TestPrometheusEndpoint:
    def test_metrics_returns_prometheus_format(self, client_with_model):
        response = client_with_model.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert b"accidents_" in response.content

    def test_metrics_contains_prediction_counter(self, client_with_model):
        client_with_model.post("/predict", json=SAMPLE_FEATURES)
        response = client_with_model.get("/metrics")
        assert b"accidents_predictions_total" in response.content

    def test_metrics_contains_model_gauge(self, client_with_model):
        response = client_with_model.get("/metrics")
        assert b"accidents_model_loaded" in response.content


class TestStatsEndpoint:
    def test_stats_returns_expected_schema(self, client_with_model):
        response = client_with_model.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "predictions_by_label" in data
        assert "uptime_seconds" in data
        assert "prioritaire" in data["predictions_by_label"]
        assert "non_prioritaire" in data["predictions_by_label"]

    def test_stats_increments_on_predict(self, client_with_model):
        import src.api.metrics as state
        before = state.stats["total"]
        client_with_model.post("/predict", json=SAMPLE_FEATURES)
        assert state.stats["total"] == before + 1


class TestModelInfoEndpoint:
    def test_model_info_returns_metadata(self, client_with_model):
        response = client_with_model.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["n_features"] == 28
        assert len(data["features"]) == 28
        assert "type" in data
        assert "params" in data

    def test_model_info_returns_503_without_model(self, client_without_model):
        response = client_without_model.get("/model/info")
        assert response.status_code == 503


class TestRetrainEndpoint:
    def test_retrain_returns_202(self, client_with_model):
        with patch("src.api.routers.monitoring._run_training"):
            response = client_with_model.post("/retrain")
        assert response.status_code == 202
        assert response.json()["status"] == "accepted"

    def test_retrain_increments_counter(self, client_with_model):
        from src.api.metrics import RETRAIN_COUNTER
        before = RETRAIN_COUNTER._value.get()
        with patch("src.api.routers.monitoring._run_training"):
            client_with_model.post("/retrain")
        assert RETRAIN_COUNTER._value.get() == before + 1


class TestPredictEndpoint:
    def test_predict_returns_valid_response(self, client_with_model):
        response = client_with_model.post("/predict", json=SAMPLE_FEATURES)
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in [0, 1]
        assert data["label"] in ["prioritaire", "non-prioritaire"]
        assert 0.0 <= data["probability"] <= 1.0
        assert data["confidence"] in ["high", "medium", "low"]

    def test_predict_confidence_matches_probability(self, client_with_model):
        response = client_with_model.post("/predict", json=SAMPLE_FEATURES)
        data = response.json()
        p = data["probability"]
        if p >= 0.80:
            assert data["confidence"] == "high"
        elif p >= 0.60:
            assert data["confidence"] == "medium"
        else:
            assert data["confidence"] == "low"

    def test_predict_label_matches_prediction(self, client_with_model):
        response = client_with_model.post("/predict", json=SAMPLE_FEATURES)
        data = response.json()
        expected = "prioritaire" if data["prediction"] == 1 else "non-prioritaire"
        assert data["label"] == expected

    def test_predict_without_model_returns_503(self, client_without_model):
        response = client_without_model.post("/predict", json=SAMPLE_FEATURES)
        assert response.status_code == 503

    def test_predict_missing_field_returns_422(self, client_with_model):
        incomplete = {k: v for k, v in SAMPLE_FEATURES.items() if k != "vma"}
        response = client_with_model.post("/predict", json=incomplete)
        assert response.status_code == 422
