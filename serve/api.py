from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.common.config import get_settings
from src.common.features import FEATURE_COLUMNS
from src.common.model_registry import (
    configure_tracking,
    load_pyfunc_model,
    resolve_model_uri,
)


class PredictionRequest(BaseModel):
    trip_distance: float = Field(..., description="Trip distance in miles")
    total_amount: float = Field(..., description="Total trip amount in dollars")


class PredictionResponse(BaseModel):
    prediction: float
    model_uri: str


app = FastAPI(title="Yellow NYC Taxi Duration API", version="0.1.0")


@lru_cache(maxsize=1)
def get_model_bundle() -> tuple[str, Any]:
    """Load the active MLflow model once and reuse it across API requests."""
    settings = get_settings()
    configure_tracking(settings.mlflow_tracking_uri)
    model_uri = resolve_model_uri(
        tracking_uri=settings.mlflow_tracking_uri,
        model_name=settings.registered_model_name,
        model_alias=settings.model_alias,
    )
    return model_uri, load_pyfunc_model(model_uri)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    model_uri, model = get_model_bundle()
    features = pd.DataFrame(
        [
            {
                "trip_distance": request.trip_distance,
                "total_amount": request.total_amount,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
    prediction = model.predict(features)
    return PredictionResponse(prediction=float(prediction[0]), model_uri=model_uri)
