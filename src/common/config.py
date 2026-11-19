import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# Reduce utility log noise without hiding warnings about model packaging.
logging.getLogger("mlflow.utils").setLevel(logging.WARNING)


@dataclass(frozen=True)
class ProjectSettings:
    """Centralized settings used by the scripts and notebooks."""

    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "yellow-taxi-duration")
    registered_model_name: str = os.getenv("MLFLOW_MODEL_NAME", "rfr-ride-duration")
    model_alias: str | None = os.getenv("MODEL_ALIAS", "production")

    train_data_uri: str = os.getenv(
        "TRAIN_DATA_URI",
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet",
    )


def get_settings() -> ProjectSettings:
    """Return project settings loaded from environment variables."""
    return ProjectSettings()
