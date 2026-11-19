import uuid
from collections.abc import Iterable

import pandas as pd

FEATURE_COLUMNS = ["trip_distance", "total_amount"]
TARGET_COLUMN = "trip_duration_minutes"
RIDE_ID_COLUMN = "ride_id"


def generate_ride_ids(n_rows: int) -> list[str]:
    """Generate new UUIDs for rows without IDs; existing ride IDs are preserved."""
    return [str(uuid.uuid4()) for _ in range(n_rows)]


def load_dataframe(source_uri: str) -> pd.DataFrame:
    """Load a Parquet dataset from a local path or URL."""
    return pd.read_parquet(source_uri)


def prepare_dataframe(df: pd.DataFrame, *, include_target: bool) -> pd.DataFrame:
    """Apply the shared feature engineering used by training and inference."""
    prepared = df.copy()

    if include_target:
        required_columns = {"tpep_pickup_datetime", "tpep_dropoff_datetime"}
        missing_columns = required_columns.difference(prepared.columns)
        if missing_columns:
            raise ValueError(
                "Cannot compute trip duration without these columns: "
                + ", ".join(sorted(missing_columns))
            )
        duration = prepared["tpep_dropoff_datetime"] - prepared["tpep_pickup_datetime"]
        prepared[TARGET_COLUMN] = duration.dt.total_seconds() / 60

        # keep only trips lasting 1min-6hours
        prepared = prepared[
            (prepared[TARGET_COLUMN] >= 1) & (prepared[TARGET_COLUMN] <= 6*60)
        ]

        # keep only trips with distance 0.1- 6h*60mph miles
        prepared = prepared[prepared["trip_distance"].between(0.1, 6*60)]

        # keep only trips with total amount 0-1000$
        prepared = prepared[prepared["total_amount"].between(0, 1000)]


    return prepared


def assign_ride_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add ride IDs if they are not present in the input dataset."""
    prepared = df.copy()
    if RIDE_ID_COLUMN not in prepared:
        prepared[RIDE_ID_COLUMN] = generate_ride_ids(len(prepared))
    return prepared


def build_prediction_frame(
    df: pd.DataFrame,
    predictions: Iterable[float],
    *,
    model_reference: str,
) -> pd.DataFrame:
    """Combine prediction output with the most useful input columns."""
    result = pd.DataFrame(
        {
            RIDE_ID_COLUMN: df[RIDE_ID_COLUMN],
            "trip_distance": df["trip_distance"],
            "total_amount": df["total_amount"],
            "predicted_duration": predictions,
            "model_reference": model_reference,
        }
    )

    if TARGET_COLUMN in df:
        result["actual_duration"] = df[TARGET_COLUMN]
        result["diff"] = result["actual_duration"] - result["predicted_duration"]

    return result
