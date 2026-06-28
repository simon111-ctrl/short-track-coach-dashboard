from pathlib import Path

import pandas as pd

from short_track_service import PredictionService


def test_predict_distance_returns_all_coach_outputs():
    root = Path(__file__).resolve().parents[1]
    service = PredictionService(root / "model_package")
    sample = pd.read_csv(root / "model_package" / "examples" / "example_input_500m_grade.csv").head(1)

    result = service.predict("500m", sample).iloc[0].to_dict()

    expected = {
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
    assert 0 <= result["advancement_probability"] <= 1
    assert 0 <= result["final_entry_probability"] <= 1
    assert result["advancement_reference"] in {"有晋级机会", "晋级压力大"}
    assert result["final_entry_reference"] in {"有机会进决赛", "暂时难进决赛"}
    assert not str(result["rhythm_type"]).startswith("Cluster")
