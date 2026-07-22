from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi.testclient import TestClient
from api.main_api import app
import api.main_api as state
from fastapi import HTTPException

client = TestClient(app)

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
    #mock_model.predict.return_value = np.array([1])
    # renvoie un tableau de "1" de la même longueur que l'entrée
    # (utile pour /predict avec 1 ligne ET /metrics avec X_test entier):
    mock_model.predict.side_effect = lambda X: np.arange(len(X)) % 2    #les deux classes sont représentées et precision_score/recall_score ont de quoi calculer sur les deux labels.
    old_model = state.model #vrai modele actuel qu'on va remplacer par le modele mock , puis on retourne au vrai modele pour ne pas influencer les autres tests
    state.model = mock_model
    with TestClient(state.app) as client:
        yield client

    state.model = old_model


@pytest.fixture
def client_without_model():
    old_model = state.model
    state.model = None
    with TestClient(state.app) as client:
        yield client

    state.model = old_model


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
        assert data["api"] == "ok"
        assert data["model_loaded"] is True
        assert data["version"] == "1.0.0"
        

    def test_health_degraded_when_no_model(self, client_without_model):
        response = client_without_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is False



class TestMetrics:
    def test_metrics_returns_expected_schema(self, client_with_model):

        response = client_with_model.get("/metrics")
        assert response.status_code == 200
        data = response.json()

        assert "accuracy" in data
        assert "precision" in data
        assert "recall" in data
        assert "f1_score" in data


class TestRetrainEndpoint:
    def test_retrain_returns_202(self, client_with_model):
        with patch("api.main_api.train"), \
             patch("api.main_api.mlflow.pyfunc.load_model", return_value=MagicMock()):
            response = client_with_model.post("/retrain")
        assert response.status_code == 202
        assert response.json()["status"] == "accepted"


class TestPredictEndpoint:
    def test_predict_returns_valid_response(self, client_with_model):
        response = client_with_model.post("/predict", json=SAMPLE_FEATURES)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in [0, 1]


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

# pour tester ce fichier test avec les logs:
# uv run pytest tests/test_api.py -vv -s

#pour lancer tous les tests : uv run pytest tests