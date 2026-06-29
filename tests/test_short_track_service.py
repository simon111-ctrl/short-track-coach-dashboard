from pathlib import Path

import pandas as pd
import pytest

from short_track_service import (
    DISTANCE_LAPS,
    FeatureAlignmentError,
    PredictionService,
    format_seconds_mmss,
    parse_time_to_seconds,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("00:42:318", 42.318),
        ("01:26:542", 86.542),
        ("2:15:076", 135.076),
        ("42.318", 42.318),
        ("42", 42.0),
        ("1:26", 86.0),
    ],
)
def test_parse_time_to_seconds_accepts_web_formats(text, expected):
    assert parse_time_to_seconds(text) == pytest.approx(expected)


def test_format_seconds_mmss_uses_milliseconds():
    assert format_seconds_mmss(86.542) == "01:26:542"
    assert format_seconds_mmss(42.318) == "00:42:318"


def test_gender_model_routing_uses_separate_assets():
    service = PredictionService(ROOT / "model_package")

    male = service.load_asset("500m", "male", "grade")
    female = service.load_asset("500m", "female", "grade")

    assert male.meta["gender"] == "male"
    assert female.meta["gender"] == "female"
    assert "models\\male\\500m\\grade" in male.meta["model_file"] or "models/male/500m/grade" in male.meta["model_file"]
    assert "models\\female\\500m\\grade" in female.meta["model_file"] or "models/female/500m/grade" in female.meta["model_file"]


def test_feature_alignment_reports_missing_and_extra_columns():
    err = FeatureAlignmentError(
        required=["a", "b"],
        current=["a", "c"],
        missing=["b"],
        extra=["c"],
    )

    assert "b" in str(err)
    assert "c" in str(err)


@pytest.mark.parametrize("gender", ["male", "female"])
@pytest.mark.parametrize("distance", ["500m", "1000m", "1500m"])
def test_predict_gender_distance_returns_all_coach_outputs(gender, distance):
    service = PredictionService(ROOT / "model_package")
    sample = pd.read_csv(ROOT / "model_package" / "examples" / f"example_input_{gender}_{distance}_advancement.csv").head(1)

    result = service.predict(distance, gender, sample).iloc[0].to_dict()

    expected = {
        "gender",
        "distance",
        "calculated_total_time",
        "calculated_total_time_display",
        "grade",
        "advancement_probability",
        "advancement_reference",
        "max_round",
        "final_entry_probability",
        "final_entry_reference",
        "rhythm_type",
        "tactical_style",
        "key_lap",
        "risk_score",
        "risk_label",
    }
    assert expected.issubset(result)
    assert result["gender"] == gender
    assert result["distance"] == distance
    assert 0 <= result["advancement_probability"] <= 1
    assert 0 <= result["final_entry_probability"] <= 1
    assert result["calculated_total_time"] == pytest.approx(
        sum(float(sample.iloc[0][f"lap{i}_time"]) for i in range(1, DISTANCE_LAPS[distance] + 1))
    )
    assert ":" in result["calculated_total_time_display"]
