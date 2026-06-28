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

GRADE_ZH = {0: "发展中", 1: "稳定达标", 2: "强势", 3: "顶尖"}
GRADE_EN = {0: "Developing", 1: "Competitive", 2: "Strong", 3: "Elite"}

STYLE_ZH = {
    "front_runner": "领跑控制型",
    "late_attacker": "后程突击型",
    "stable_controller": "稳定控节奏型",
    "volatile_risk": "高波动风险型",
    "chaser": "追赶反击型",
}
STYLE_EN = {
    "front_runner": "Front control",
    "late_attacker": "Late attacker",
    "stable_controller": "Stable control",
    "volatile_risk": "Volatile risk",
    "chaser": "Chasing counter",
}

RHYTHM_LABELS = {
    "500m": {
        0: {"zh": "起速控位型", "en": "Fast-start control"},
        1: {"zh": "起速反击型", "en": "Fast-start counter"},
        2: {"zh": "前快后掉型", "en": "Fast-start fade"},
        3: {"zh": "高波动风险型", "en": "High-volatility risk"},
    },
    "1000m": {
        0: {"zh": "慢起回落型", "en": "Slow-start fade"},
        1: {"zh": "后程追击型", "en": "Late chase"},
        2: {"zh": "稳控推进型", "en": "Steady control"},
        3: {"zh": "高波动风险型", "en": "High-volatility risk"},
    },
    "1500m": {
        0: {"zh": "均衡守位型", "en": "Balanced holding"},
        1: {"zh": "开局积极但后程吃亏型", "en": "Aggressive start, late loss"},
        2: {"zh": "节奏稳提型", "en": "Balanced push"},
        3: {"zh": "高波动风险型", "en": "High-volatility risk"},
    },
}

RHYTHM_HINTS = {
    "500m": {
        0: {"zh": "前段起速快，节奏偏控位。", "en": "Fast opening with controlled positioning."},
        1: {"zh": "起速积极，同时有一定反击能力。", "en": "Aggressive start with counter ability."},
        2: {"zh": "前快后掉，后半程掉速较明显。", "en": "Fast early, then a visible late fade."},
        3: {"zh": "节奏波动大，风险高。", "en": "Large pace swings and high risk."},
    },
    "1000m": {
        0: {"zh": "前段偏保守，末段有一定回落。", "en": "Conservative opening with late fade."},
        1: {"zh": "适合后程提速和追击。", "en": "Built for late acceleration and chase."},
        2: {"zh": "整体稳定，适合控节奏推进。", "en": "Stable overall and suited to controlled pace."},
        3: {"zh": "波动大，容易被节奏打乱。", "en": "Highly variable and easy to disrupt."},
    },
    "1500m": {
        0: {"zh": "整体均衡，守位能力较好。", "en": "Balanced overall with solid position holding."},
        1: {"zh": "开局更积极，但后程保持不足。", "en": "Aggressive early, but less stable late."},
        2: {"zh": "前中后节奏衔接更顺。", "en": "Better rhythm linkage across the race."},
        3: {"zh": "长距离内波动过大，风险很高。", "en": "Too much variance over the long race."},
    },
}

ADV_LABELS = {
    "zh": {0: "晋级压力大", 1: "有晋级机会"},
    "en": {0: "Needs review", 1: "Likely to advance"},
}

FINAL_LABELS = {
    "zh": {0: "暂时难进决赛", 1: "有机会进决赛"},
    "en": {0: "Final path uncertain", 1: "Final path likely"},
}

RISK_LABELS = {
    "zh": {"Normal": "风险正常", "High risk": "高风险"},
    "en": {"Normal": "Normal", "High risk": "High risk"},
}

ROUND_LABELS = {
    "zh": {
        1: "预赛",
        2: "复活赛",
        3: "预赛 / 初轮",
        4: "复活赛 / 四分之一",
        5: "四分之一决赛",
        6: "复活赛 / 半决赛",
        7: "半决赛",
        8: "名次赛 / B组决赛",
        9: "决赛 / A组决赛",
    },
    "en": {
        1: "Preliminaries",
        2: "Repechage heats",
        3: "Heats",
        4: "Repechage quarterfinals",
        5: "Quarterfinals",
        6: "Repechage semifinals",
        7: "Semifinals",
        8: "Ranking / Final B",
        9: "Final / Final A",
    },
}

FEATURE_LABELS = {
    "zh": {
        "official_total_time_filled": "官方总成绩",
        "reconstructed_total_time": "圈速重建总成绩",
        "mean_lap_time": "平均圈速",
        "lap_time_std": "圈速波动",
        "lap_time_cv": "圈速稳定度",
        "lap_time_range": "最快最慢圈差",
        "start_ratio_to_mean": "起速相对平均",
        "finish_ratio_to_mean": "末段相对平均",
        "finish_vs_mid_delta": "末段对中段变化",
        "finish_vs_previous_delta": "末圈对前一圈变化",
        "start_position": "起跑位置",
        "final_position": "终点位置",
        "mean_position": "平均位置",
        "position_std": "位置波动",
        "position_range": "位置区间",
        "position_gain_total": "总位置提升",
        "overtake_events_count": "超越次数",
        "position_loss_events_count": "被超越次数",
        "positive_position_gain_sum": "有效追位总和",
        "negative_position_loss_sum": "掉位总和",
        "position_stability_score": "位置稳定度",
        "lap_time_delta_std": "圈速变化波动",
        "first_turn_position": "首弯位置",
        "early_position_gain_lap1_2": "前两圈追位",
        "explosive_stability_lap1_3_std": "前段爆发稳定度",
        "last_lap_conversion_delta": "末圈转换",
        "late_position_conversion": "末段追位转换",
        "front_mid_pace_mean_lap2_5": "前中段平均圈速",
        "mid_position_control_mean_lap3_6": "中段控位均值",
        "acceleration_lap6_vs_lap5": "第六圈相对第五圈",
        "speed_hold_lap6_8_std": "中后段速度稳定度",
        "sprint_reserve_lap9_vs_lap7_8": "末段冲刺储备",
        "late_position_gain_lap7_9": "末段位置提升",
        "long_pace_early_mean_lap2_5": "前段平均圈速",
        "long_pace_mid_mean_lap6_10": "中段平均圈速",
        "long_pace_late_mean_lap11_14": "后段平均圈速",
        "mid_energy_saving_delta": "中段蓄力差",
        "pace_layering_range": "分段节奏跨度",
        "attack_gain_lap10_12": "中后段发起追击",
        "back_end_capacity_lap11_14_vs_mid": "后程储备",
        "late_position_gain_lap10_14": "末段位置提升",
    },
    "en": {
        "official_total_time_filled": "Official total time",
        "reconstructed_total_time": "Reconstructed total time",
        "mean_lap_time": "Average lap time",
        "lap_time_std": "Lap-time variability",
        "lap_time_cv": "Lap-time stability",
        "lap_time_range": "Fastest-slowest gap",
        "start_ratio_to_mean": "Start vs average",
        "finish_ratio_to_mean": "Finish vs average",
        "finish_vs_mid_delta": "Finish vs middle section",
        "finish_vs_previous_delta": "Final lap vs previous",
        "start_position": "Start position",
        "final_position": "Finish position",
        "mean_position": "Average position",
        "position_std": "Position volatility",
        "position_range": "Position range",
        "position_gain_total": "Total position gain",
        "overtake_events_count": "Overtakes",
        "position_loss_events_count": "Position losses",
        "positive_position_gain_sum": "Total gains",
        "negative_position_loss_sum": "Total losses",
        "position_stability_score": "Position stability",
        "lap_time_delta_std": "Lap-change volatility",
        "first_turn_position": "First-turn position",
        "early_position_gain_lap1_2": "Early position gain",
        "explosive_stability_lap1_3_std": "Early explosive stability",
        "last_lap_conversion_delta": "Final-lap conversion",
        "late_position_conversion": "Late position conversion",
        "front_mid_pace_mean_lap2_5": "Front-mid pace",
        "mid_position_control_mean_lap3_6": "Mid-race position control",
        "acceleration_lap6_vs_lap5": "Lap 6 vs lap 5",
        "speed_hold_lap6_8_std": "Mid-late pace stability",
        "sprint_reserve_lap9_vs_lap7_8": "Sprint reserve",
        "late_position_gain_lap7_9": "Late position gain",
        "long_pace_early_mean_lap2_5": "Early long-race pace",
        "long_pace_mid_mean_lap6_10": "Mid-race pace",
        "long_pace_late_mean_lap11_14": "Late-race pace",
        "mid_energy_saving_delta": "Mid-race saving delta",
        "pace_layering_range": "Pace layering range",
        "attack_gain_lap10_12": "Attack gain",
        "back_end_capacity_lap11_14_vs_mid": "Back-end capacity",
        "late_position_gain_lap10_14": "Late position gain",
    },
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
    def _lang(value: str) -> str:
        return "zh" if value not in {"zh", "en"} else value

    @staticmethod
    def required_columns(distance: str) -> list[str]:
        laps = DISTANCE_LAPS[distance]
        return (
            ["athlete_name", "distance", "round", "qual_code", "official_total_time"]
            + [f"lap{i}_time" for i in range(1, laps + 1)]
            + [f"lap{i}_position" for i in range(1, laps + 1)]
        )

    def required_columns_label_table(self, distance: str, value_lang: str) -> pd.DataFrame:
        value_lang = self._lang(value_lang)
        rows = []
        base = {
            "athlete_name": {"zh": "运动员 / 样本名", "en": "Athlete / sample name"},
            "distance": {"zh": "距离", "en": "Distance"},
            "round": {"zh": "轮次", "en": "Round"},
            "qual_code": {"zh": "晋级代码", "en": "Qualification code"},
            "official_total_time": {"zh": "官方总成绩", "en": "Official total time"},
        }
        for col in ["athlete_name", "distance", "round", "qual_code", "official_total_time"]:
            rows.append({"column": col, "meaning": base[col][value_lang]})
        for lap in range(1, DISTANCE_LAPS[distance] + 1):
            rows.append({"column": f"lap{lap}_time", "meaning": {"zh": f"第{lap}圈成绩", "en": f"Lap {lap} time"}[value_lang]})
            rows.append({"column": f"lap{lap}_position", "meaning": {"zh": f"第{lap}圈位置", "en": f"Lap {lap} position"}[value_lang]})
        return pd.DataFrame(rows)

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

    def _grade_label(self, value: int, value_lang: str) -> str:
        return (GRADE_ZH if self._lang(value_lang) == "zh" else GRADE_EN).get(int(value), str(value))

    def _style_label(self, code: str, value_lang: str) -> str:
        labels = STYLE_ZH if self._lang(value_lang) == "zh" else STYLE_EN
        return labels.get(code, code.replace("_", " ").title())

    def _rhythm_label(self, distance: str, cluster: int, value_lang: str) -> str:
        return RHYTHM_LABELS.get(distance, {}).get(int(cluster), {}).get(self._lang(value_lang), f"Cluster {cluster}")

    def _rhythm_hint(self, distance: str, cluster: int, value_lang: str) -> str:
        return RHYTHM_HINTS.get(distance, {}).get(int(cluster), {}).get(self._lang(value_lang), "")

    def _round_label(self, value: int, value_lang: str) -> str:
        return ROUND_LABELS.get(self._lang(value_lang), {}).get(int(value), str(value))

    def _risk_label(self, code: str, value_lang: str) -> str:
        return RISK_LABELS.get(self._lang(value_lang), {}).get(code, code)

    def _adv_label(self, probability: float, value_lang: str) -> str:
        return ADV_LABELS[self._lang(value_lang)][1 if probability >= 0.5 else 0]

    def _final_label(self, probability: float, value_lang: str) -> str:
        return FINAL_LABELS[self._lang(value_lang)][1 if probability >= 0.5 else 0]

    def _feature_label(self, feature: str, value_lang: str) -> str:
        return FEATURE_LABELS[self._lang(value_lang)].get(feature, feature.replace("_", " "))

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
        data["timing_consistent"] = (data["official_total_time"].isna() | (data["official_total_abs_delta"] <= 0.10)).astype(int)
        data = self.fe.add_common_features(data, laps)
        data, _ = self.fe.add_distance_features(data, distance, laps)
        return data

    def predict(self, distance: str, raw: pd.DataFrame, value_lang: str = "zh") -> pd.DataFrame:
        value_lang = self._lang(value_lang)
        engineered = self.prepare_features(distance, raw)
        result = raw.reset_index(drop=True).copy()

        for task in TASKS:
            asset = self.load_asset(distance, task)
            X = engineered[asset.features]
            pred = np.asarray(asset.model.predict(X)).reshape(-1)

            if task == "risk_detection":
                result["risk_score"] = -asset.model.decision_function(X)
                result["risk_label"] = np.where(pred == -1, self._risk_label("High risk", value_lang), self._risk_label("Normal", value_lang))
                continue

            if task == "style_cluster":
                result["rhythm_cluster"] = pred.astype(int)
                result["rhythm_type"] = [self._rhythm_label(distance, int(x), value_lang) for x in pred]
                result["rhythm_hint"] = [self._rhythm_hint(distance, int(x), value_lang) for x in pred]
                continue

            probabilities = self._positive_or_winning_probability(asset.model, X, pred)
            if task == "grade":
                result["grade_code"] = pred.astype(int)
                result["grade"] = [self._grade_label(int(x), value_lang) for x in pred]
                result["grade_probability"] = probabilities
            elif task == "advancement":
                result["advancement_probability"] = probabilities
                result["advancement_reference"] = [self._adv_label(float(p), value_lang) for p in probabilities]
            elif task == "max_round":
                result["max_round_score"] = pred.astype(int)
                result["max_round"] = [self._round_label(int(x), value_lang) for x in pred]
                result["max_round_probability"] = probabilities
            elif task == "final_entry":
                result["final_entry_probability"] = probabilities
                result["final_entry_reference"] = [self._final_label(float(p), value_lang) for p in probabilities]
            elif task == "tactical_style":
                reverse = {int(k): v for k, v in asset.meta.get("label_map", {}).items()}
                raw_labels = [reverse.get(int(x), str(x)) for x in pred]
                result["tactical_style_code"] = raw_labels
                result["tactical_style"] = [self._style_label(code, value_lang) for code in raw_labels]
                result["tactical_style_probability"] = probabilities
            elif task == "key_lap":
                result["key_lap"] = [f"第{int(x)}圈" if value_lang == "zh" else f"Lap {int(x)}" for x in pred]
                result["key_lap_probability"] = probabilities

        return result

    def global_explanation(self, distance: str, task: str, value_lang: str = "zh", limit: int = 8) -> pd.DataFrame:
        value_lang = self._lang(value_lang)
        model_id = f"{distance}_{task}"
        table = self.explanations[self.explanations["model_id"] == model_id].copy()
        if table.empty:
            return pd.DataFrame(columns=["feature", "feature_label", "importance_mean"])
        table["feature_label"] = table["feature"].map(lambda name: self._feature_label(name, value_lang))
        return table.sort_values("importance_mean", ascending=False).head(limit)

    def local_explanation(self, distance: str, task: str, raw: pd.DataFrame, value_lang: str = "zh", limit: int = 6) -> pd.DataFrame:
        value_lang = self._lang(value_lang)
        engineered = self.prepare_features(distance, raw).reset_index(drop=True)
        global_top = self.global_explanation(distance, task, value_lang, limit)
        rows = []
        for feature in global_top["feature"]:
            if feature in engineered:
                rows.append(
                    {
                        "feature": feature,
                        "feature_label": self._feature_label(feature, value_lang),
                        "value": float(engineered.loc[0, feature]),
                    }
                )
        return pd.DataFrame(rows)

    def coach_notes(self, distance: str, prediction: pd.Series, value_lang: str = "zh") -> list[str]:
        value_lang = self._lang(value_lang)
        notes: list[str] = []
        adv = float(prediction.get("advancement_probability", 0))
        final = float(prediction.get("final_entry_probability", 0))
        risk = str(prediction.get("risk_label", self._risk_label("Normal", value_lang)))
        style = str(prediction.get("tactical_style", ""))
        rhythm = str(prediction.get("rhythm_type", ""))
        key_lap = str(prediction.get("key_lap", ""))
        grade = str(prediction.get("grade", ""))

        if value_lang == "zh":
            notes.append(f"当前结果是{grade}，晋级参考{self._adv_label(adv, value_lang)}，决赛通道{self._final_label(final, value_lang)}。")
            if adv >= 0.7:
                notes.append("晋级信号偏强，当前结构可以继续保留，但需要盯紧细节。")
            elif adv >= 0.45:
                notes.append("晋级信号处在中间带，重点看位置控制和转折圈处理。")
            else:
                notes.append("晋级信号偏弱，优先检查起跑、卡位和中后段执行。")
            if final >= 0.5:
                notes.append("决赛通道偏强，可以考虑更积极的比赛计划。")
            else:
                notes.append("决赛通道不稳，先保证可重复的执行质量，再考虑激进超越。")
            if risk == "高风险":
                notes.append("异常风险偏高，建议回看碰撞、节奏断裂、换道和数据质量。")
            if "起速" in rhythm:
                notes.append("起速偏快的比赛，前两圈不要把体能一次性打满，保留中后段反应空间。")
            elif "后程" in rhythm:
                notes.append("后程型比赛，前段以占位为主，真正发力点放在转折圈前后。")
            elif "稳控" in rhythm or "均衡" in rhythm:
                notes.append("节奏较稳，适合用固定圈速推进，减少无谓的节奏切换。")
            elif "波动" in rhythm:
                notes.append("波动明显，先处理节奏稳定和线路干净，再谈进一步提速。")
            if "前快后掉" in rhythm:
                notes.append("前快后掉的特征比较明显，建议把前段冲刺改成更平滑的推进。")
            if "追赶" in rhythm:
                notes.append("追赶型比赛，关键是把差距留到最后可处理的范围，不要过早透支。")
            notes.append(f"当前战术风格：{style}。转折圈：{key_lap}。")
            notes.append(f"这是赛后的{distance}参考结果，建议结合录像和教练判断一起看。")
        else:
            notes.append(f"Current profile reads as {grade}, with {self._adv_label(adv, value_lang)} and {self._final_label(final, value_lang)}.")
            if adv >= 0.7:
                notes.append("Advancement signal is strong; keep the current structure and sharpen the details.")
            elif adv >= 0.45:
                notes.append("Advancement is in the middle band; focus on position control and the key lap.")
            else:
                notes.append("Advancement looks weak; check the start, lane control, and mid-late execution first.")
            if final >= 0.5:
                notes.append("Final-entry path is open enough for a more aggressive race plan.")
            else:
                notes.append("Final-entry path is still unstable; protect repeatable execution before risky moves.")
            if risk == "High risk":
                notes.append("Risk is elevated; review contact, rhythm breaks, lane changes, and data quality.")
            if "Fast-start" in rhythm:
                notes.append("The race opens quickly, so avoid emptying the tank in the first two laps.")
            elif "Late" in rhythm:
                notes.append("This profile suits a late push, so keep position first and press after the key lap.")
            elif "Steady" in rhythm or "Balanced" in rhythm:
                notes.append("The rhythm is stable, so a fixed-pace plan should work well here.")
            elif "risk" in rhythm.lower():
                notes.append("The pace is unstable, so fix rhythm and line cleanliness before adding speed.")
            if "fade" in rhythm.lower():
                notes.append("The opening is strong but the finish fades, so smooth the first half instead of forcing it.")
            if "chase" in rhythm.lower():
                notes.append("This is a chasing profile, so keep the gap manageable until the final decision point.")
            notes.append(f"Current tactical style: {style}. Turning lap: {key_lap}.")
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
