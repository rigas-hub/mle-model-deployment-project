from typing import Any

import mlflow
from mlflow import MlflowClient


def configure_tracking(tracking_uri: str) -> None:
    """Point MLflow APIs at the configured tracking server."""
    mlflow.set_tracking_uri(tracking_uri)


def ensure_experiment(experiment_name: str) -> str:
    """Create the experiment on demand and return its identifier."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id

    return mlflow.create_experiment(experiment_name)


def _registered_model_version_scores(
    client: MlflowClient, model_name: str
) -> list[dict[str, Any]]:
    versions = client.search_model_versions(filter_string=f"name='{model_name}'")
    return [
        {
            "version": str(version.version),
            # Fetch the complete version because search_model_versions may
            # omit aliases even when they are assigned.
            "aliases": list(
                client.get_model_version(model_name, version.version).aliases or []
            ),
            "run_id": version.run_id,
            "metrics": client.get_run(version.run_id).data.metrics,
        }
        for version in sorted(versions, key=lambda item: int(item.version), reverse=True)
    ]


def get_registered_model_version_scores(
    *, tracking_uri: str, model_name: str
) -> list[dict[str, Any]]:
    """Return each registered version with its aliases and logged run metrics."""
    client = MlflowClient(tracking_uri=tracking_uri)
    return _registered_model_version_scores(client, model_name)


def promote_model_if_better(
    *,
    tracking_uri: str,
    model_name: str,
    candidate_version: str,
    metric_name: str = "rmse_minutes",
    greater_is_better: bool = False,
    alias: str = "production",
) -> bool:
    """Assign an alias when the candidate beats its current version.

    Returns True if the alias was assigned, including when no version currently
    has the alias. Metric comparison is strict; ties do not trigger promotion.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = _registered_model_version_scores(client, model_name)

    candidate = next(
        (version for version in versions if version["version"] == candidate_version),
        None,
    )
    if candidate is None:
        raise ValueError(
            f"Version '{candidate_version}' is not registered as '{model_name}'."
        )

    candidate_score = candidate["metrics"].get(metric_name)
    if candidate_score is None:
        raise ValueError(
            f"Metric '{metric_name}' is missing from candidate version "
            f"'{candidate_version}'."
        )

    current = next(
        (version for version in versions if alias in version["aliases"]),
        None,
    )
    if current is None:
        client.set_registered_model_alias(model_name, alias, candidate_version)
        return True

    current_score = current["metrics"].get(metric_name)
    if current_score is None:
        raise ValueError(
            f"Metric '{metric_name}' is missing from current '{alias}' version "
            f"'{current['version']}'."
        )

    is_better = (
        candidate_score > current_score
        if greater_is_better
        else candidate_score < current_score
    )
    if not is_better:
        return False

    client.set_registered_model_alias(model_name, alias, candidate_version)
    return True


def resolve_model_uri(
    *,
    tracking_uri: str,
    model_uri: str | None = None,
    model_name: str | None = None,
    model_version: str | None = None,
    model_alias: str | None = None,
    model_stage: str | None = None,
    run_id: str | None = None,
    artifact_path: str = "model",
) -> str:
    """Resolve a model URI from either a direct URI, a run ID, or the registry."""

    # The repository uses this one resolution path everywhere so the
    # training script, batch flow, and API all agree on which model version
    # should be active.

    configure_tracking(tracking_uri)

    # Keep the override order explicit: direct URI first, then a run artifact,
    # then progressively more specific registry selectors.
    if model_uri:
        return model_uri

    if run_id:
        return f"runs:/{run_id}/{artifact_path}"

    if not model_name:
        raise ValueError(
            "Model resolution requires either MODEL_URI, MODEL_RUN_ID, or MLFLOW_MODEL_NAME."
        )

    if model_alias:
        return f"models:/{model_name}@{model_alias}"

    if model_version:
        return f"models:/{model_name}/{model_version}"

    client = MlflowClient(tracking_uri=tracking_uri)
    versions = list(client.search_model_versions(f"name='{model_name}'"))
    if not versions:
        raise ValueError(
            f"No registered model versions found for '{model_name}'. "
            "Train and register a model first, or set MODEL_RUN_ID."
        )

    if model_stage:
        staged_versions = [
            version for version in versions if version.current_stage == model_stage
        ]
        if not staged_versions:
            raise ValueError(
                f"No versions for '{model_name}' found in stage '{model_stage}'."
            )
        latest = max(staged_versions, key=lambda version: int(version.version))
        return f"models:/{model_name}/{latest.version}"

    latest = max(versions, key=lambda version: int(version.version))
    return f"models:/{model_name}/{latest.version}"


def load_pyfunc_model(model_uri: str) -> Any:
    """Load a pyfunc model from MLflow."""
    return mlflow.pyfunc.load_model(model_uri)
