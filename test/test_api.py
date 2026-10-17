from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import app as api_module
from app import app


class ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, frame):
        return np.array([self.value] * len(frame))


TEST_BUNDLE = {
    "features": ["age", "studytime", "failures", "absences", "G1", "G2"],
    "models": {
        "linear_regression": ConstantModel(9.5),
        "random_forest": ConstantModel(16.2),
    },
    "metrics": {
        "linear_regression": {"mae": 3.2, "rmse": 4.1, "r2": 0.5},
        "random_forest": {"mae": 2.7, "rmse": 3.5, "r2": 0.62},
    },
    "best_model": "random_forest",
    "target": "G3",
    "dataset": "Test fixture",
    "sample_count": 316,
    "model_source": "test fixture",
    "sklearn_version": "test",
    "feature_importance": [
        {"feature": "G2", "importance": 0.55},
        {"feature": "G1", "importance": 0.31},
    ],
}


client = TestClient(app)


def setup_function():
    api_module.get_model_bundle = lambda: TEST_BUNDLE


def teardown_function():
    # Do not leak the monkeypatch into subsequent tests.
    from services.model_service import get_model_bundle
    api_module.get_model_bundle = get_model_bundle


def sample_payload(**overrides):
    payload = {
        "age": 17,
        "studytime": 3,
        "failures": 0,
        "absences": 5,
        "G1": 11,
        "G2": 11,
        "model": "best",
    }
    payload.update(overrides)
    return payload


def test_health_does_not_require_a_model():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_info_reports_notebook_features_and_target():
    response = client.get("/api/info")
    assert response.status_code == 200
    body = response.json()
    assert body["target"].startswith("G3")
    assert body["features"] == ["age", "studytime", "failures", "absences", "G1", "G2"]


def test_prediction_uses_best_mae_model_and_includes_both_models():
    response = client.post("/api/predict", json=sample_payload())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prediction"] == 16.2
    assert body["category"] == "High Performance"
    assert body["selected_model"] == "random_forest"
    assert body["predictions"]["linear_regression"] == 9.5
    assert body["predictions"]["random_forest"] == 16.2


def test_prediction_can_select_linear_regression():
    response = client.post("/api/predict", json=sample_payload(model="linear_regression"))
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 9.5
    assert body["category"] == "Low Performance"


def test_validation_rejects_out_of_range_features():
    response = client.post("/api/predict", json=sample_payload(age=99))
    assert response.status_code == 422


def test_batch_prediction_returns_all_rows():
    response = client.post(
        "/api/batch-predict",
        json={
            "model": "best",
            "students": [
                {"age": 17, "studytime": 3, "failures": 0, "absences": 5, "G1": 11, "G2": 11},
                {"age": 16, "studytime": 2, "failures": 1, "absences": 8, "G1": 9, "G2": 10},
            ],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2
    assert body["results"][0]["prediction"] == 16.2


def test_metrics_route_returns_evaluation_and_feature_importance():
    response = client.get("/api/metrics")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["best_model"] == "random_forest"
    assert body["metrics"]["random_forest"]["mae"] == 2.7
    assert body["feature_importance"][0]["feature"] == "G2"


def test_grade_output_is_clamped_to_20():
    bundle = dict(TEST_BUNDLE)
    bundle["models"] = {"linear_regression": ConstantModel(26), "random_forest": ConstantModel(26)}
    api_module.get_model_bundle = lambda: bundle
    response = client.post("/api/predict", json=sample_payload())
    assert response.status_code == 200
    assert response.json()["prediction"] == 20


def test_local_homepage_can_serve_index_html():
    response = client.get("/", follow_redirects=True)
    assert response.status_code == 200
    assert "GradeWise" in response.text
