import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Reduce utility log noise without hiding warnings about model packaging.
logging.getLogger("mlflow.utils").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(
    path_value: str | None, *, default: Path | None = None
) -> Path:
    """Resolve a path relative to the project root when not absolute."""
    if not path_value:
        if default is None:
            raise ValueError("Expected a path value or default path.")
        return default

    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@dataclass(frozen=True)
class ProjectSettings:
    """Centralized settings used by the scripts and notebooks."""

    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "yellow-taxi-duration")
    registered_model_name: str = os.getenv("MLFLOW_MODEL_NAME", "rfr-ride-duration")
    model_alias: str | None = os.getenv("MODEL_ALIAS", "production")

    train_data_uri: str = os.getenv(
        "TRAIN_DATA_URI",
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet",
    )
    
    artifacts_dir: Path = field(
        default_factory=lambda: resolve_project_path(
            os.getenv("MLFLOW_ARTIFACTS_DIR"),
            default=PROJECT_ROOT / "mlartifacts",
        )
    )


def get_settings() -> ProjectSettings:
    """Return project settings loaded from environment variables."""
    return ProjectSettings()
