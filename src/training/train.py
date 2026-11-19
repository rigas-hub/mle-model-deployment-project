import click
import mlflow
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.common.config import get_settings
from src.common.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_dataframe,
    prepare_dataframe,
)
from src.common.model_registry import (
    configure_tracking,
    ensure_experiment,
    promote_model_if_better,
)

settings = get_settings()


@click.command(help="Train the taxi duration model.")
@click.option(
    "--data-uri",
    default=settings.train_data_uri,
    help="URI to training data.",
)
@click.option(
    "--tracking-uri",
    default=settings.mlflow_tracking_uri,
    help="MLflow tracking server URI.",
)
@click.option(
    "--experiment-name",
    default=settings.experiment_name,
    help="MLflow experiment name.",
)
@click.option(
    "--registered-model-name",
    default=settings.registered_model_name,
    help="MLflow model name.",
)
@click.option(
    "--model-alias",
    default=settings.model_alias,
    help="MLflow model alias.",
)
@click.option(
    "--test-size",
    type=float,
    default=0.3,
    help="Fraction of data to use for testing.",
)
@click.option(
    "--random-state",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
)
@click.option(
    "--run-name",
    default="src/training/train.py",
    help="MLflow run name.",
)
def main(
    data_uri: str,
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    model_alias: str,
    test_size: float,
    random_state: int,
    run_name: str,
) -> None:
    # Create the experiment on demand and log a single baseline pipeline.
    configure_tracking(tracking_uri)
    experiment_id = ensure_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    df = load_dataframe(data_uri)
    prepared = prepare_dataframe(df, include_target=True)

    prepared_features = prepared[FEATURE_COLUMNS]
    target = prepared[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        prepared_features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
    ) as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, predictions)

        mlflow.log_params(
            {
                "train_data_uri": data_uri,
                "test_size": test_size,
                "random_state": random_state,
                "model_type": "RandomForestRegressor",
                "feature_columns": ', '.join(FEATURE_COLUMNS),
            }
        )
        mlflow.log_metric("rmse_minutes", float(rmse))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model trained by script",
            registered_model_name=registered_model_name,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        client = MlflowClient(tracking_uri=tracking_uri)
        registered_versions = client.search_model_versions(
            filter_string=f"name='{registered_model_name}'"
        )
        candidate_versions = [
            version
            for version in registered_versions
            if version.run_id == run.info.run_id
        ]
        if len(candidate_versions) != 1:
            raise RuntimeError(
                f"Expected one registered version for run '{run.info.run_id}', "
                f"found {len(candidate_versions)}."
            )

        candidate_version = str(candidate_versions[0].version)
        promoted = promote_model_if_better(
            tracking_uri=tracking_uri,
            model_name=registered_model_name,
            candidate_version=candidate_version,
            alias=model_alias
        )

        print(f"Run ID: {run.info.run_id}")
        print(f"RMSE: {rmse:.4f}")
        if promoted:
            print(
                f"Promoted {registered_model_name} version {candidate_version} "
                f"to the '{model_alias}' alias."
            )
        else:
            print(
                f"Did not promote {registered_model_name} version {candidate_version}; "
                f"'{model_alias}' already has an equal or better rmse_minutes score."
            )


if __name__ == "__main__":
    main()
