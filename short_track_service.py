from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


DISTANCE_LAPS = {"500m": 5, "1000m": 9, "1500m": 14}
TASKS = [
    "grade",
    "advancement",
    "max_round",
    "final_entry",
    "style_cluster",
    "tactical_style",
    "key_lap",
    "risk_detection",
]

GRADE_LABELS = {
    0: "Developing",
    1: "Competitive",
    2: "Strong",
    3: "Elite",
}

STYLE_LABELS = {
    "front_runner": "Front runner",
    "late_attacker": "Late attacker",
    "stable_controller": "Stable controller",
    "volatile_risk": "Volatile risk",
    "chaser": "Chaser",
}

FEATURE_NAMES = {
    "official_total_time_filled": "Official total time",
    "reconstructed_total_time": "Reconstructed total time",
    "mean_lap_time": "Average lap time",
    "lap_time_std": "Lap-time variability",
    "lap_time_cv": "Lap-time consistency",
    "lap_time_range": "Fastest-slowest lap gap",
    "start_ratio_to_mean": "Start speed ratio",
    "finish_ratio_to_mean": "Finish speed ratio",
    "finish_vs_mid_delta": "Finish vs middle section",
    "finish_vs_previous_delta": "Final lap change",
    "start_position": "Opening position",
    "final_position": "Final position",
    "mean_position": "Average position",
    "position_std": "Position volatility",
    "position_range": "Position range",
    "position_gain_total": "Total position gain",
    "overtake_events_count": "Overtaking events",
    "position_loss_events_count": "Position loss events",
    "positive_position_gain_sum": "Total gains",
    "negative_position_loss_sum": "Total losses",
    "position_stability_score": "Position stability",
    "lap_time_delta_std": "Pace-change volatility",
}


@dataclass(frozen=True)
class ModelAsset:
    model: Any
    features: list[str]
    meta: dict[str, Any]


class PredictionService:
    def __init__(self, package_dir: str | Path):
        self.package_dir = Path(package_dir)
        src_dir = self.package_dir / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        import feature_engineering as fe

        self.fe = fe
        self.manifest = json.loads((self.package_dir / "model_manifest.json").read_text(encoding="utf-8"))
        self.explanations = pd.read_csv(self.package_dir / "reports" / "model_explanations.csv")
        self.metrics = pd.read_csv(self.package_dir / "reports" / "model_metrics.csv")
        self._asset_cache: dict[str, ModelAsset] = {}

    @staticmethod
    def required_columns(distance: str) -> list[str]:
        laps = DISTANCE_LAPS[distance]
        return (
            ["athlete_name", "distance", "round", "qual_code", "official_total_time"]
            + [f"lap{i}_time" for i in range(1, laps + 1)]
            + [f"lap{i}_position" for i in range(1, laps + 1)]
        )

    def load_asset(self, distance: str, task: str) -> ModelAsset:
        model_id = f"{distance}_{task}"
        if model_id in self._asset_cache:
            return self._asset_cache[model_id]
        meta = self.manifest["models"][model_id]
        model = joblib.load(self.package_dir / meta["model_file"])
        features = json.loads((self.package_dir / meta["features_file"]).read_text(encoding="utf-8"))
        asset = ModelAsset(model=model, features=features, meta=meta)
        self._asset_cache[model_id] = asset
        return asset

    def prepare_features(self, distance: str, raw: pd.DataFrame) -> pd.DataFrame:
        laps = DISTANCE_LAPS[distance]
        data = raw.copy()
        data["distance"] = distance
        for col in ["athlete_name", "round", "qual_code"]:
            if col not in data:
                data[col] = ""
        for lap in range(1, laps + 1):
            data[f"lap{lap}_time"] = pd.to_numeric(data[f"lap{lap}_time"], errors="coerce")
            data[f"lap{lap}_position"] = pd.to_numeric(data[f"lap{lap}_position"], errors="coerce")
        lap_times = [f"lap{i}_time" for i in range(1, laps + 1)]
        data["reconstructed_total_time"] = data[lap_times].sum(axis=1)
        if "official_total_time" not in data:
            data["official_total_time"] = np.nan
        data["official_total_time"] = pd.to_numeric(data["official_total_time"], errors="coerce")
        data["official_total_time_filled"] = data["official_total_time"].fillna(data["reconstructed_total_time"])
        data["total_time_was_missing"] = data["official_total_time"].isna().astype(int)
        data["official_total_delta"] = data["official_total_time"] - data["reconstructed_total_time"]
        data["official_total_abs_delta"] = data["official_total_delta"].abs()
        data["timing_consistent"] = (
            data["official_total_time"].isna() | (data["official_total_abs_delta"] <= 0.10)
        ).astype(int)

        data = self.fe.add_common_features(data, laps)
        data, _ = self.fe.add_distance_features(data, distance, laps)
        return data

    def predict(self, distance: str, raw: pd.DataFrame) -> pd.DataFrame:
        engineered = self.prepare_features(distance, raw)
        result = raw.reset_index(drop=True).copy()

        for task in TASKS:
            asset = self.load_asset(distance, task)
            X = engineered[asset.features]
            pred = asset.model.predict(X)
            pred = np.asarray(pred).reshape(-1)

            if task == "risk_detection":
                scores = -asset.model.decision_function(X)
                result["risk_score"] = scores
                result["risk_label"] = np.where(pred == -1, "High risk", "Normal")
                continue

            if task == "style_cluster":
                result["rhythm_type"] = [f"Cluster {int(x)}" for x in pred]
                result["rhythm_cluster"] = pred.astype(int)
                continue

            probabilities = self._positive_or_winning_probability(asset.model, X, pred)
            if task == "grade":
                result["grade"] = [GRADE_LABELS.get(int(x), str(x)) for x in pred]
                result["grade_score"] = pred.astype(int)
                result["grade_probability"] = probabilities
            elif task == "advancement":
                result["advancement_probability"] = probabilities
                result["advancement_reference"] = np.where(probabilities >= 0.5, "Likely", "Needs review")
            elif task == "max_round":
                labels = asset.meta.get("label_map", {})
                result["max_round_score"] = pred.astype(int)
                result["max_round"] = [labels.get(str(int(x)), str(x)) for x in pred]
                result["max_round_probability"] = probabilities
            elif task == "final_entry":
                result["final_entry_probability"] = probabilities
                result["final_entry_reference"] = np.where(probabilities >= 0.5, "Final path likely", "Final path uncertain")
            elif task == "tactical_style":
                reverse = {int(k): v for k, v in asset.meta.get("label_map", {}).items()}
                raw_labels = [reverse.get(int(x), str(x)) for x in pred]
                result["tactical_style"] = [STYLE_LABELS.get(x, x.replace("_", " ").title()) for x in raw_labels]
                result["tactical_style_probability"] = probabilities
            elif task == "key_lap":
                result["key_lap"] = [f"Lap {int(x)}" for x in pred]
                result["key_lap_probability"] = probabilities

        return result

    def global_explanation(self, distance: str, task: str, limit: int = 8) -> pd.DataFrame:
        model_id = f"{distance}_{task}"
        table = self.explanations[self.explanations["model_id"] == model_id].copy()
        if table.empty:
            return pd.DataFrame(columns=["feature", "feature_label", "importance_mean"])
        table["feature_label"] = table["feature"].map(lambda x: FEATURE_NAMES.get(x, x.replace("_", " ")))
        return table.sort_values("importance_mean", ascending=False).head(limit)

    def local_explanation(self, distance: str, task: str, raw: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
        engineered = self.prepare_features(distance, raw).reset_index(drop=True)
        global_top = self.global_explanation(distance, task, limit)
        rows = []
        for feature in global_top["feature"]:
            if feature in engineered:
                rows.append(
                    {
                        "feature": feature,
                        "feature_label": FEATURE_NAMES.get(feature, feature.replace("_", " ")),
                        "value": float(engineered.loc[0, feature]),
                    }
                )
        return pd.DataFrame(rows)

    def coach_notes(self, distance: str, prediction: pd.Series) -> list[str]:
        notes = []
        adv = float(prediction.get("advancement_probability", 0))
        final = float(prediction.get("final_entry_probability", 0))
        risk = prediction.get("risk_label", "Normal")
        style = prediction.get("tactical_style", "")
        key_lap = prediction.get("key_lap", "")
        grade = prediction.get("grade", "")
        if adv >= 0.7:
            notes.append("Advancement reference is strong; protect the current race structure and focus review on details.")
        elif adv >= 0.45:
            notes.append("Advancement reference is borderline; video review should focus on position control and the key lap.")
        else:
            notes.append("Advancement reference is low; treat this as a warning for race position, timing, or tactical execution.")
        if final >= 0.5:
            notes.append("Final-entry probability supports an aggressive competition plan when lane and draw conditions allow it.")
        else:
            notes.append("Final-entry probability is uncertain; prioritize repeatable execution before high-risk passing plans.")
        if risk == "High risk":
            notes.append("The anomaly detector marks this profile as high risk; check contact, pacing disruption, penalties, or data quality.")
        notes.append(f"Current tactical read: {style}; key review point: {key_lap}; performance band: {grade}.")
        notes.append(f"This is a post-race {distance} reference and should be combined with video and coach judgement.")
        return notes

    @staticmethod
    def _positive_or_winning_probability(model: Any, X: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
        if not hasattr(model, "predict_proba"):
            return np.ones(len(X))
        proba = np.asarray(model.predict_proba(X))
        classes = np.asarray(getattr(model, "classes_", []))
        if proba.ndim != 2 or proba.shape[1] == 0:
            return np.ones(len(X))
        if len(classes) and 1 in classes:
            return proba[:, int(np.where(classes == 1)[0][0])]
        if len(classes):
            indexes = [int(np.where(classes == value)[0][0]) for value in pred]
            return proba[np.arange(len(X)), indexes]
        return proba.max(axis=1)
