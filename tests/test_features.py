import pandas as pd
import pytest

from src.common.features import TARGET_COLUMN, prepare_dataframe


def _trips(durations_minutes, distances, amounts):
    pickup = pd.Timestamp("2025-01-01")
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": [pickup] * len(durations_minutes),
            "tpep_dropoff_datetime": [
                pickup + pd.Timedelta(minutes=minutes)
                for minutes in durations_minutes
            ],
            "trip_distance": distances,
            "total_amount": amounts,
        },
        index=[f"trip-{index}" for index in range(len(durations_minutes))],
    )


def test_prepare_dataframe_derives_duration_and_keeps_inclusive_boundaries():
    trips = _trips(
        durations_minutes=[1, 360, 10, 10, 10, 10],
        distances=[1, 1, 0.1, 360, 1, 1],
        amounts=[10, 10, 10, 10, 0, 1000],
    )
    original = trips.copy(deep=True)

    prepared = prepare_dataframe(trips, include_target=True)

    assert prepared.index.tolist() == trips.index.tolist()
    assert prepared[TARGET_COLUMN].tolist() == [1, 360, 10, 10, 10, 10]
    pd.testing.assert_frame_equal(trips, original)


@pytest.mark.parametrize(
    ("durations", "distances", "amounts"),
    [
        ([0.99], [1], [10]),
        ([360.01], [1], [10]),
        ([10], [0.09], [10]),
        ([10], [360.01], [10]),
        ([10], [1], [-0.01]),
        ([10], [1], [1000.01]),
    ],
)
def test_prepare_dataframe_removes_rows_outside_cleaning_ranges(
    durations, distances, amounts
):
    trips = _trips(
        durations_minutes=[10, *durations],
        distances=[1, *distances],
        amounts=[10, *amounts],
    )

    prepared = prepare_dataframe(trips, include_target=True)

    assert prepared.index.tolist() == ["trip-0"]


@pytest.mark.parametrize(
    "columns",
    [
        {"tpep_pickup_datetime": pd.to_datetime(["2025-01-01"])},
        {"tpep_dropoff_datetime": pd.to_datetime(["2025-01-01"])},
    ],
)
def test_prepare_dataframe_requires_both_timestamps(columns):
    with pytest.raises(ValueError, match="Cannot compute trip duration"):
        prepare_dataframe(pd.DataFrame(columns), include_target=True)


def test_prepare_dataframe_does_not_clean_or_add_target_without_target():
    trips = _trips(
        durations_minutes=[0.5],
        distances=[0.01],
        amounts=[-1],
    )

    prepared = prepare_dataframe(trips, include_target=False)

    pd.testing.assert_frame_equal(prepared, trips)
    assert TARGET_COLUMN not in prepared
