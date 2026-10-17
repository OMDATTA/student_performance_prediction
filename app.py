"""GradeWise Student Performance Predictor — FastAPI backend for Vercel.

The browser UI is deployed as static files from public/. The API implements
student final-grade predictions using the same six input features and two
regression models as the accompanying Colab notebook.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from services.model_service import ModelUnavailableError, get_model_bundle, predict_grade

BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR / "public"
ON_VERCEL = bool(os.getenv("VERCEL"))

app = FastAPI(
    title="GradeWise Student Performance Predictor API",
    description=(
        "Predict a student's final grade (G3, 0–20 scale) from age, weekly "
        "study time, past failures, absences, and first-/second-period grades. "
        "For educational demonstration only; not a high-stakes decision tool."
    ),
    version="1.0.0",
)

ModelChoice = Literal["best", "linear_regression", "random_forest"]


class StudentFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: int = Field(ge=15, le=22, description="Student age in years")
    studytime: int = Field(ge=1, le=4, description="Study time category from the notebook")
    failures: int = Field(ge=0, le=4, description="Past class failures on the dataset scale")
    absences: int = Field(ge=0, le=93, description="Number of school absences")
    G1: float = Field(ge=0, le=20, description="First-period grade")
    G2: float = Field(ge=0, le=20, description="Second-period grade")


class PredictionRequest(StudentFeatures):
    model: ModelChoice = "best"


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    students: list[StudentFeatures] = Field(min_length=1, max_length=100)
    model: ModelChoice = "best"


def _get_bundle_or_http_error():
    try:
        return get_model_bundle()
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The prediction model is not available yet. Add the model bundle "
                "exported by the Colab notebook to models/student_performance_models.joblib, "
                "or allow the service to download the UCI dataset when a prediction is requested. "
                f"Details: {exc}"
            ),
        ) from exc
    except Exception as exc:
        # Avoid exposing internal stack traces but keep the underlying issue in Vercel logs.
        import logging
        logging.getLogger("gradewise").exception("Prediction model failed to load")
        raise HTTPException(
            status_code=503,
            detail="The prediction model could not be loaded. Check Vercel Runtime Logs.",
        ) from exc


@app.get("/api/health", tags=["System"])
def health():
    """Fast health check that does not trigger model downloads or training."""
    return {
        "status": "ok",
        "service": "GradeWise Student Performance Predictor",
        "runtime": "FastAPI on Vercel",
        "model_loading": "lazy on first prediction or metrics request",
    }


@app.get("/api/info", tags=["System"])
def info():
    return {
        "service": "GradeWise",
        "target": "G3 — final grade on the 0–20 scale",
        "features": ["age", "studytime", "failures", "absences", "G1", "G2"],
        "models": ["linear_regression", "random_forest"],
        "data_source": "UCI Student Performance dataset, student-mat.csv",
        "privacy": "Submitted values are used for the request and are not saved by this app.",
        "notice": "Estimates are educational demonstrations, not guaranteed outcomes.",
    }


@app.post("/api/predict", tags=["Prediction"])
async def predict(payload: PredictionRequest):
    bundle = await run_in_threadpool(_get_bundle_or_http_error)
    features = payload.model_dump(exclude={"model"})
    try:
        result = await run_in_threadpool(predict_grade, bundle, features, payload.model)
    except Exception as exc:
        import logging
        logging.getLogger("gradewise").exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Check runtime logs.") from exc
    return {"success": True, **result}


@app.post("/api/batch-predict", tags=["Prediction"])
async def batch_predict(payload: BatchPredictionRequest):
    bundle = await run_in_threadpool(_get_bundle_or_http_error)
    predictions = []
    try:
        for index, student in enumerate(payload.students, start=1):
            row = student.model_dump()
            result = await run_in_threadpool(predict_grade, bundle, row, payload.model)
            predictions.append({"row": index, **row, **result})
    except Exception as exc:
        import logging
        logging.getLogger("gradewise").exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Batch prediction failed. Check runtime logs.") from exc
    return {"success": True, "count": len(predictions), "model": payload.model, "results": predictions}


@app.get("/api/metrics", tags=["Model insights"])
async def model_metrics():
    bundle = await run_in_threadpool(_get_bundle_or_http_error)
    return {
        "success": True,
        "dataset": bundle.get("dataset", "UCI Student Performance dataset"),
        "sample_count": bundle.get("sample_count"),
        "features": bundle.get("features", ["age", "studytime", "failures", "absences", "G1", "G2"]),
        "target": bundle.get("target", "G3"),
        "best_model": bundle.get("best_model"),
        "metrics": bundle.get("metrics", {}),
        "feature_importance": bundle.get("feature_importance", []),
        "model_source": bundle.get("model_source", "Colab model export or dataset training"),
        "sklearn_version": bundle.get("sklearn_version"),
    }


@app.get("/", include_in_schema=False)
def homepage():
    """Vercel serves public assets through its CDN; local runs serve them from disk."""
    if ON_VERCEL:
        return RedirectResponse(url="/index.html", status_code=307)
    index_file = PUBLIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file, media_type="text/html")
    return {"service": "GradeWise", "message": "Website files not found; expected public/index.html."}


# Keep this mount local-only. On Vercel, public/ is served by the CDN and must
# not be mounted from inside the function's filesystem.
if not ON_VERCEL and PUBLIC_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="website")
