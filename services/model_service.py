"""Model loading, training fallback, and inference for GradeWise.

Best deployment path: export the trained objects from Colab into
models/student_performance_models.joblib and commit that file. If no artifact
exists, this module tries to download the original UCI dataset and trains the
same Linear Regression and Random Forest models on first use.
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["age", "studytime", "failures", "absences", "G1", "G2"]
MODEL_PATH = ROOT / "models" / "student_performance_models.joblib"
LOCAL_DATA_PATH = ROOT / "data" / "student-mat.csv"
UCI_ZIP_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
MIRROR_URLS = [
    "https://raw.githubusercontent.com/Sauharda-py/linear-regression-student-performance/main/student-mat.csv",
    "https://raw.githubusercontent.com/rupakc/UCI-Data-Analysis/master/Student%20Dataset/Student/student-mat.csv",
]
logger = logging.getLogger("gradewise.model")


class ModelUnavailableError(RuntimeError):
    """Raised when neither a saved bundle nor training data is available."""


def _read_data_bytes(url: str, timeout: float = 8.0) -> bytes:
    request = Request(url, headers={"User-Agent": "GradeWise-Student-Performance-Demo/1.0"})
    with urlopen(request, timeout=timeout) as response:  # nosec: known public dataset URLs only
        content = response.read(5 * 1024 * 1024 + 1)
    if len(content) > 5 * 1024 * 1024:
        raise ValueError("Dataset response was unexpectedly large.")
    if not content:
        raise ValueError("Dataset source returned an empty response.")
    return content


def _read_dataframe_from_bytes(raw: bytes):
    import pandas as pd

    stream = io.BytesIO(raw)
    if zipfile.is_zipfile(stream):
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            matches = [name for name in archive.namelist() if Path(name).name.lower() == "student-mat.csv"]
            if not matches:
                raise ValueError("The downloaded archive did not contain student-mat.csv.")
            with archive.open(matches[0]) as csv_file:
                return pd.read_csv(csv_file, sep=";")
    return pd.read_csv(io.BytesIO(raw), sep=";")


def _load_training_data():
    import pandas as pd

    if LOCAL_DATA_PATH.is_file():
        df = pd.read_csv(LOCAL_DATA_PATH, sep=";")
        return df, "local data/student-mat.csv"

    failures = []
    for url in [UCI_ZIP_URL, *MIRROR_URLS]:
        try:
            raw = _read_data_bytes(url, timeout=8.0 if "archive.ics.uci.edu" in url else 6.0)
            df = _read_dataframe_from_bytes(raw)
            if not set(FEATURES + ["G3"]).issubset(df.columns):
                raise ValueError("Dataset does not contain all required feature columns.")
            logger.info("Loaded training data from %s", url)
            return df, url
        except Exception as exc:
            failures.append(f"{url}: {type(exc).__name__}: {exc}")
            logger.warning("Dataset source failed (%s): %s", url, exc)

    details = " | ".join(failures)
    raise ModelUnavailableError(
        "No trained model artifact was found and the UCI dataset could not be retrieved. "
        "Run the deployment notebook's export cell and add the .joblib file to models/. "
        + details
    )


def _metric_dict(y_true, predictions) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def _train_bundle() -> dict[str, Any]:
    import pandas as pd
    import sklearn
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    df, source = _load_training_data()
    df = df.drop_duplicates().copy()
    # Match the supplied Colab notebook's feature list and holdout split.
    X = df[FEATURES].apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(df["G3"], errors="raise")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    linear = LinearRegression()
    linear.fit(X_train, y_train)
    random_forest = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    random_forest.fit(X_train, y_train)

    linear_predictions = linear.predict(X_test)
    rf_predictions = random_forest.predict(X_test)
    metrics = {
        "linear_regression": _metric_dict(y_test, linear_predictions),
        "random_forest": _metric_dict(y_test, rf_predictions),
    }
    best_model = min(metrics, key=lambda key: metrics[key]["mae"])
    feature_importance = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in sorted(
            zip(FEATURES, random_forest.feature_importances_),
            key=lambda item: item[1], reverse=True
        )
    ]
    return {
        "features": FEATURES.copy(),
        "models": {"linear_regression": linear, "random_forest": random_forest},
        "metrics": metrics,
        "best_model": best_model,
        "target": "G3",
        "dataset": "UCI Student Performance — Mathematics (student-mat.csv)",
        "sample_count": int(len(df)),
        "model_source": f"Trained on request from {source}",
        "sklearn_version": sklearn.__version__,
        "feature_importance": feature_importance,
    }


def _load_saved_bundle() -> dict[str, Any] | None:
    if not MODEL_PATH.is_file():
        return None
    import joblib
    import sklearn

    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("models"), dict):
        raise ModelUnavailableError("The saved model file has an unexpected format.")
    if not set(FEATURES).issubset(bundle.get("features", [])):
        raise ModelUnavailableError("The saved model is missing one or more required features.")
    for name in ("linear_regression", "random_forest"):
        if name not in bundle["models"]:
            raise ModelUnavailableError(f"The saved model bundle is missing {name}.")
    # Cross-version scikit-learn model loading is not guaranteed. If versions
    # differ, warn in logs; fall back to fresh training only if loading fails.
    model_version = bundle.get("sklearn_version")
    if model_version and model_version != sklearn.__version__:
        logger.warning(
            "Saved model was produced with scikit-learn %s; runtime is %s. "
            "Compatibility is not guaranteed; validate predictions.",
            model_version, sklearn.__version__
        )
    bundle.setdefault("target", "G3")
    bundle.setdefault("dataset", "UCI Student Performance — Mathematics (student-mat.csv)")
    bundle.setdefault("sample_count", None)
    bundle.setdefault("model_source", "Saved model exported from Google Colab")
    bundle.setdefault("sklearn_version", model_version or sklearn.__version__)
    bundle.setdefault("metrics", {})
    bundle.setdefault("features", FEATURES.copy())
    bundle.setdefault("feature_importance", [])
    if not bundle.get("best_model") and bundle.get("metrics"):
        viable = [key for key in ("linear_regression", "random_forest") if key in bundle["metrics"]]
        if viable:
            bundle["best_model"] = min(viable, key=lambda key: bundle["metrics"][key].get("mae", math.inf))
    return bundle


@lru_cache(maxsize=1)
def get_model_bundle() -> dict[str, Any]:
    """Load the exported Colab bundle or train from the original dataset once per warm instance."""
    try:
        saved = _load_saved_bundle()
        if saved is not None:
            logger.info("Loaded model bundle from %s", MODEL_PATH)
            return saved
    except Exception as exc:
        logger.exception("Could not load saved Colab model; trying dataset training instead: %s", exc)
    return _train_bundle()


def _category(score: float) -> str:
    # Keep the thresholds from the user's Colab notebook.
    if score >= 15:
        return "High Performance"
    if score >= 10:
        return "Average Performance"
    return "Low Performance"


def predict_grade(bundle: dict[str, Any], features: dict[str, Any], selected_model: str = "best") -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    models = bundle["models"]
    model_name = bundle.get("best_model", "random_forest") if selected_model == "best" else selected_model
    if model_name not in models:
        raise ValueError(f"Model '{model_name}' is not available.")
    row = pd.DataFrame([[features[name] for name in FEATURES]], columns=FEATURES)
    all_predictions: dict[str, float] = {}
    for name, model in models.items():
        raw = float(model.predict(row)[0])
        # G3 is defined on the 0-20 grading scale; bound output to that scale.
        all_predictions[name] = round(float(np.clip(raw, 0, 20)), 2)
    estimate = all_predictions[model_name]
    metrics = bundle.get("metrics", {})
    return {
        "selected_model": model_name,
        "selected_model_label": "Random Forest Regression" if model_name == "random_forest" else "Linear Regression",
        "prediction": estimate,
        "category": _category(estimate),
        "predictions": all_predictions,
        "metrics": metrics,
        "best_model": bundle.get("best_model", model_name),
        "feature_importance": bundle.get("feature_importance", []),
        "input_features": features,
        "note": (
            "An estimate based on historical data, not a guarantee of actual results. "
            "The underlying dataset is limited and the prediction should not be used to label or penalize a student."
        ),
    }
