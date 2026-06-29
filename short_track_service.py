from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd


DISTANCE_LAPS = {"500m": 5, "1000m": 9, "1500m": 14}
GENDERS = {"male", "female"}
GENDER_ALIASES = {
    "男": "male",
    "男子": "male",
    "male": "male",
    "men": "male",
    "man": "male",
    "m": "male",
    "女": "female",
    "女子": "female",
    "female": "female",
    "women": "female",
    "woman": "female",
    "f": "female",
}
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

ADV_LABELS = {"zh": {0: "晋级压力大", 1: "有晋级机会"}, "en": {0: "Needs review", 1: "Likely to advance"}}
FINAL_LABELS = {"zh": {0: "暂时难进决赛", 1: "有机会进决赛"}, "en": {0: "Final path uncertain", 1: "Final path likely"}}
RISK_LABELS = {"zh": {"Normal": "风险正常", "High risk": "高风险"}, "en": {"Normal": "Normal", "High risk": "High risk"}}

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
    "official_total_time_filled": {"zh": "官方总成绩", "en": "Official total time"},
    "reconstructed_total_time": {"zh": "自动求和总成绩", "en": "Calculated total time"},
    "mean_lap_time": {"zh": "平均圈速", "en": "Average lap time"},
    "lap_time_std": {"zh": "圈速波动", "en": "Lap-time variability"},
    "lap_time_cv": {"zh": "圈速稳定性", "en": "Lap-time stability"},
    "lap_time_range": {"zh": "最快最慢圈差", "en": "Fastest-slowest gap"},
    "start_ratio_to_mean": {"zh": "起速相对平均", "en": "Start vs average"},
    "finish_ratio_to_mean": {"zh": "末段相对平均", "en": "Finish vs average"},
    "finish_vs_mid_delta": {"zh": "末段对中段变化", "en": "Finish vs middle"},
    "finish_vs_previous_delta": {"zh": "末圈对前一圈变化", "en": "Final lap vs previous"},
    "start_position": {"zh": "起始位置", "en": "Start position"},
    "final_position": {"zh": "终点位置", "en": "Finish position"},
    "mean_position": {"zh": "平均位置", "en": "Average position"},
    "position_std": {"zh": "位置波动", "en": "Position volatility"},
    "position_range": {"zh": "位置区间", "en": "Position range"},
    "position_gain_total": {"zh": "总位置提升", "en": "Total position gain"},
    "overtake_events_count": {"zh": "超越次数", "en": "Overtakes"},
    "position_loss_events_count": {"zh": "掉位次数", "en": "Position losses"},
    "lap_time_delta_std": {"zh": "圈速变化波动", "en": "Lap-change volatility"},
}


class FeatureAlignmentError(ValueError):
    def __init__(self, required: Iterable[str], current: Iterable[str], missing: Iterable[str], extra: Iterable[str]):
        self.required = list(required)
        self.current = list(current)
        self.missing = list(missing)
        self.extra = list(extra)
        super().__init__(
            "Feature fields do not match the selected model.\n"
            f"Required fields: {self.required}\n"
            f"Current fields: {self.current}\n"
            f"Missing fields: {self.missing}\n"
            f"Extra fields: {self.extra}"
        )


@dataclass(frozen=True)
class ModelAsset:
    model: Any
    features: list[str]
    meta: dict[str, Any]


def normalize_gender(value: str) -> str:
    text = str(value).strip().lower()
    gender = GENDER_ALIASES.get(text)
    if gender not in GENDERS:
        raise ValueError("请选择性别：男 / Male / Men 或 女 / Female / Women")
    return gender


def parse_time_to_seconds(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    text = text.replace(",", ".")
    if text.upper() in {"DNS", "DNF", "PEN", "YC", "DQ", "DSQ", "WDR"}:
        return np.nan
    parts = text.split(":")
    try:
        if len(parts) == 3:
            minutes = int(parts[0])
            seconds = int(parts[1])
            millis = int(parts[2].ljust(3, "0")[:3])
            if seconds >= 60:
                raise ValueError
            return minutes * 60 + seconds + millis / 1000
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            if seconds >= 60:
                raise ValueError
            return minutes * 60 + seconds
        if len(parts) == 1:
            return float(text)
    except ValueError as exc:
        raise ValueError(f"成绩格式错误：{value}。请使用 mm:ss:SSS、mm:ss、ss 或 ss.sss。") from exc
    raise ValueError(f"成绩格式错误：{value}。请使用 mm:ss:SSS、mm:ss、ss 或 ss.sss。")


def format_seconds_mmss(seconds: float) -> str:
    if seconds is None or np.isnan(seconds):
        return ""
    total_ms = int(round(float(seconds) * 1000))
    minutes, remainder = divmod(total_ms, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}:{millis:03d}"


class PredictionService:
    def __init__(self, package_dir: str | Path):
        self.package_dir = Path(package_dir)
        src_dir = self.package_dir / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        import feature_engineering as fe

        self.fe = fe
        manifest_path = self.package_dir / "web_model_manifest.json"
        if not manifest_path.exists():
            manifest_path = self.package_dir / "model_manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.explanations = self._read_csv_if_exists(self.package_dir / "reports" / "model_explanations.csv")
        self.metrics = self._read_csv_if_exists(self.package_dir / "metrics" / "gender_distance_model_comparison.csv")
        self._asset_cache: dict[str, ModelAsset] = {}

    @staticmethod
    def _read_csv_if_exists(path: Path) -> pd.DataFrame:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    @staticmethod
    def _lang(value: str) -> str:
        return "zh" if value not in {"zh", "en"} else value

    def model_id(self, distance: str, gender: str, task: str) -> str:
        gender = normalize_gender(gender)
        if distance not in DISTANCE_LAPS:
            raise ValueError(f"不支持的距离：{distance}")
        if task not in TASKS:
            raise ValueError(f"不支持的模型任务：{task}")
        return f"{gender}_{distance}_{task}"

    def _local_asset_path(self, raw_path: str) -> Path:
        normalized = str(raw_path).replace("\\", "/").strip()
        if not normalized:
            raise FileNotFoundError("模型文件路径为空")

        for marker in ("models/", "reports/", "examples/", "src/", "metrics/", "web/"):
            index = normalized.lower().find(marker)
            if index != -1:
                return self.package_dir / normalized[index:]

        path = Path(normalized)
        if path.is_absolute():
            return path
        return self.package_dir / path

    @staticmethod
    def required_columns(distance: str) -> list[str]:
        laps = DISTANCE_LAPS[distance]
        return (
            ["athlete_name", "gender", "distance", "round", "qual_code", "official_total_time"]
            + [f"lap{i}_time" for i in range(1, laps + 1)]
            + [f"lap{i}_position" for i in range(1, laps + 1)]
        )

    def required_columns_label_table(self, distance: str, value_lang: str) -> pd.DataFrame:
        value_lang = self._lang(value_lang)
        names = {
            "athlete_name": {"zh": "运动员 / 样本名", "en": "Athlete / sample name"},
            "gender": {"zh": "性别", "en": "Gender"},
            "distance": {"zh": "距离", "en": "Distance"},
            "round": {"zh": "轮次", "en": "Round"},
            "qual_code": {"zh": "晋级标记", "en": "Qualification code"},
            "official_total_time": {"zh": "官方总成绩，可留空", "en": "Official total time, optional"},
        }
        rows = [{"column": col, "meaning": names[col][value_lang]} for col in names]
        for lap in range(1, DISTANCE_LAPS[distance] + 1):
            rows.append({"column": f"lap{lap}_time", "meaning": {"zh": f"第{lap}圈成绩，秒数", "en": f"Lap {lap} time, seconds"}[value_lang]})
            rows.append({"column": f"lap{lap}_position", "meaning": {"zh": f"第{lap}圈位置", "en": f"Lap {lap} position"}[value_lang]})
        return pd.DataFrame(rows)

    def load_asset(self, distance: str, gender: str, task: str) -> ModelAsset:
        model_id = self.model_id(distance, gender, task)
        if model_id in self._asset_cache:
            return self._asset_cache[model_id]
        if model_id not in self.manifest.get("models", {}):
            raise FileNotFoundError(f"模型清单中找不到：{model_id}")

        meta = dict(self.manifest["models"][model_id])
        model_path = self._local_asset_path(meta["model_file"])
        features_path = self._local_asset_path(meta["features_file"])
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"特征文件不存在：{features_path}")
        model = joblib.load(model_path)
        features = json.loads(features_path.read_text(encoding="utf-8"))
        if meta.get("feature_columns") and list(meta["feature_columns"]) != list(features):
            raise FeatureAlignmentError(meta["feature_columns"], features, [], [])
        asset = ModelAsset(model=model, features=list(features), meta=meta)
        self._asset_cache[model_id] = asset
        return asset

    def prepare_features(self, distance: str, gender: str, raw: pd.DataFrame) -> pd.DataFrame:
        gender = normalize_gender(gender)
        laps = DISTANCE_LAPS[distance]
        data = raw.copy()
        data["gender"] = gender
        data["distance"] = distance
        for col in ["athlete_name", "round", "qual_code"]:
            if col not in data:
                data[col] = ""

        for lap in range(1, laps + 1):
            time_col = f"lap{lap}_time"
            pos_col = f"lap{lap}_position"
            if time_col not in data or pos_col not in data:
                raise ValueError(f"缺少必填字段：{time_col if time_col not in data else pos_col}")
            data[time_col] = data[time_col].map(parse_time_to_seconds)
            data[pos_col] = pd.to_numeric(data[pos_col], errors="coerce")

        lap_times = [f"lap{i}_time" for i in range(1, laps + 1)]
        lap_positions = [f"lap{i}_position" for i in range(1, laps + 1)]
        if data[lap_times + lap_positions].isna().any(axis=None):
            missing = data[lap_times + lap_positions].columns[data[lap_times + lap_positions].isna().any()].tolist()
            raise ValueError(f"请补全每圈成绩和每圈位置。缺失或格式错误字段：{', '.join(missing)}")

        data["reconstructed_total_time"] = data[lap_times].sum(axis=1)
        data["calculated_total_time"] = data["reconstructed_total_time"]
        data["calculated_total_time_display"] = data["calculated_total_time"].map(format_seconds_mmss)
        if "official_total_time" not in data:
            data["official_total_time"] = np.nan
        data["official_total_time"] = data["official_total_time"].map(parse_time_to_seconds)
        data["official_total_time_filled"] = data["official_total_time"].fillna(data["reconstructed_total_time"])
        data["total_time_was_missing"] = data["official_total_time"].isna().astype(int)
        data["official_total_delta"] = data["official_total_time"] - data["reconstructed_total_time"]
        data["official_total_abs_delta"] = data["official_total_delta"].abs()
        data["timing_consistent"] = (data["official_total_time"].isna() | (data["official_total_abs_delta"] <= 0.10)).astype(int)

        data = self.fe.add_common_features(data, laps)
        data, _ = self.fe.add_distance_features(data, distance, laps)
        return data

    def align_features(self, engineered: pd.DataFrame, asset: ModelAsset) -> pd.DataFrame:
        required = list(asset.features)
        current = list(engineered.columns)
        missing = [col for col in required if col not in engineered.columns]
        extra = [col for col in engineered.columns if col not in required]
        if missing:
            raise FeatureAlignmentError(required, current, missing, extra)
        return engineered.loc[:, required]

    def predict(self, distance: str, gender: str, raw: pd.DataFrame, value_lang: str = "zh") -> pd.DataFrame:
        value_lang = self._lang(value_lang)
        gender = normalize_gender(gender)
        engineered = self.prepare_features(distance, gender, raw)
        result = raw.reset_index(drop=True).copy()
        result["gender"] = gender
        result["distance"] = distance
        result["calculated_total_time"] = engineered["calculated_total_time"].to_numpy()
        result["calculated_total_time_display"] = engineered["calculated_total_time_display"].to_numpy()

        for task in TASKS:
            asset = self.load_asset(distance, gender, task)
            X = self.align_features(engineered, asset)
            pred = np.asarray(asset.model.predict(X)).reshape(-1)

            if task == "risk_detection":
                if hasattr(asset.model, "decision_function"):
                    result["risk_score"] = -asset.model.decision_function(X)
                else:
                    result["risk_score"] = np.where(pred == -1, 1.0, 0.0)
                result["risk_label"] = np.where(pred == -1, self._risk_label("High risk", value_lang), self._risk_label("Normal", value_lang))
                continue

            if task == "style_cluster":
                result["rhythm_cluster"] = pred.astype(int)
                result["rhythm_type"] = [self._rhythm_label(distance, int(x), value_lang) for x in pred]
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

    def reference_stats(self, distance: str, gender: str) -> dict[str, Any]:
        return self.manifest.get("reference_statistics", {}).get(f"{normalize_gender(gender)}_{distance}", {})

    def gender_distance_summary(self, distance: str, gender: str) -> dict[str, Any]:
        model = self.manifest.get("models", {}).get(self.model_id(distance, gender, "grade"), {})
        stats = self.reference_stats(distance, gender)
        return {"model": model, "stats": stats}

    def global_explanation(self, distance: str, gender: str, task: str, value_lang: str = "zh", limit: int = 8) -> pd.DataFrame:
        value_lang = self._lang(value_lang)
        model_id = self.model_id(distance, gender, task)
        if self.explanations.empty:
            return pd.DataFrame(columns=["feature", "feature_label", "importance_mean"])
        table = self.explanations[self.explanations["model_id"] == model_id].copy()
        if table.empty:
            return pd.DataFrame(columns=["feature", "feature_label", "importance_mean"])
        table["feature_label"] = table["feature"].map(lambda name: self._feature_label(name, value_lang))
        return table.sort_values("importance_mean", ascending=False).head(limit)

    def local_explanation(self, distance: str, gender: str, task: str, raw: pd.DataFrame, value_lang: str = "zh", limit: int = 6) -> pd.DataFrame:
        engineered = self.prepare_features(distance, gender, raw).reset_index(drop=True)
        global_top = self.global_explanation(distance, gender, task, value_lang, limit)
        rows = []
        for feature in global_top["feature"]:
            if feature in engineered:
                rows.append({"feature": feature, "feature_label": self._feature_label(feature, value_lang), "value": float(engineered.loc[0, feature])})
        return pd.DataFrame(rows)

    def generate_advice(self, row: pd.Series, distance: str, gender: str, value_lang: str = "zh") -> list[dict[str, str]]:
        value_lang = self._lang(value_lang)
        gender = normalize_gender(gender)
        stats = self.reference_stats(distance, gender)
        total = float(row.get("calculated_total_time", np.nan))
        total_display = row.get("calculated_total_time_display", format_seconds_mmss(total))
        fast_q25 = stats.get("elite_reference_fast_q25_seconds")
        median = stats.get("reconstructed_total_time_median")
        grade_code = int(row.get("grade_code", -1))
        gender_text = "男性" if gender == "male" else "女性"
        if value_lang == "en":
            gender_text = "male" if gender == "male" else "female"

        if value_lang == "zh":
            if fast_q25 is not None and total <= float(fast_q25):
                gap = f"自动总成绩 {total_display} 已进入当前{gender_text}{distance}样本的快速四分位参考区间。"
            elif median is not None and total <= float(median):
                gap = f"自动总成绩 {total_display} 快于当前{gender_text}{distance}样本中位数，接近精英参考区间。"
            else:
                gap = f"自动总成绩 {total_display} 仍慢于当前{gender_text}{distance}样本中位参考，需要结合圈速结构继续提升。"
            return [
                {
                    "title": "性别模型说明",
                    "body": f"本次只调用{gender_text}{distance}分组模型，预测只和对应性别样本的精英特征比较，不与另一性别模型互相比较。",
                },
                {
                    "title": "成绩位置",
                    "body": f"{gap} 模型给出的成绩等级为“{row.get('grade', '')}”，晋级参考为“{row.get('advancement_reference', '')}”。",
                },
                {
                    "title": "分段表现",
                    "body": f"关键圈为{row.get('key_lap', '')}，节奏类型为“{row.get('rhythm_type', '')}”，战术风格为“{row.get('tactical_style', '')}”。建议重点回看关键圈前后的控位、变速和线路选择。",
                },
                {
                    "title": "训练重点",
                    "body": self._distance_training_advice(distance, grade_code, value_lang),
                },
            ]
        if fast_q25 is not None and total <= float(fast_q25):
            gap = f"The calculated total {total_display} is inside this {gender_text} {distance} fast-quartile reference band."
        elif median is not None and total <= float(median):
            gap = f"The calculated total {total_display} is faster than this {gender_text} {distance} median reference."
        else:
            gap = f"The calculated total {total_display} is slower than this {gender_text} {distance} median reference."
        return [
            {"title": "Gender-specific model", "body": f"This run uses only the {gender_text} {distance} model and compares the athlete with the same-gender reference group."},
            {"title": "Performance gap", "body": f"{gap} Predicted grade: {row.get('grade', '')}; advancement reference: {row.get('advancement_reference', '')}."},
            {"title": "Segments", "body": f"Key lap: {row.get('key_lap', '')}; rhythm: {row.get('rhythm_type', '')}; tactical style: {row.get('tactical_style', '')}."},
            {"title": "Training focus", "body": self._distance_training_advice(distance, grade_code, value_lang)},
        ]

    @staticmethod
    def _distance_training_advice(distance: str, grade_code: int, value_lang: str) -> str:
        if value_lang == "zh":
            base = {
                "500m": "500m优先看起速、首弯位置、前两圈消耗和末圈转换。",
                "1000m": "1000m优先看中段控位、6-8圈速度保持和最后两圈冲刺储备。",
                "1500m": "1500m优先看前中后段节奏分层、中段省力和10圈后的后程能力。",
            }[distance]
            return base + (" 当前等级偏高，训练重点可放在稳定复现。" if grade_code >= 2 else " 当前仍有提升空间，先把节奏稳定和关键圈执行做扎实。")
        base = {
            "500m": "For 500m, watch the start, first-turn position, early cost, and final-lap conversion.",
            "1000m": "For 1000m, watch mid-race position control, lap 6-8 speed hold, and sprint reserve.",
            "1500m": "For 1500m, watch pace layering, mid-race energy saving, and late-race capacity.",
        }[distance]
        return base + (" The current grade is strong, so focus on repeatability." if grade_code >= 2 else " Build rhythm stability and key-lap execution first.")

    def _grade_label(self, value: int, value_lang: str) -> str:
        return (GRADE_ZH if self._lang(value_lang) == "zh" else GRADE_EN).get(int(value), str(value))

    def _style_label(self, code: str, value_lang: str) -> str:
        labels = STYLE_ZH if self._lang(value_lang) == "zh" else STYLE_EN
        return labels.get(code, code.replace("_", " ").title())

    def _rhythm_label(self, distance: str, cluster: int, value_lang: str) -> str:
        return RHYTHM_LABELS.get(distance, {}).get(int(cluster), {}).get(self._lang(value_lang), f"Cluster {cluster}")

    def _round_label(self, value: int, value_lang: str) -> str:
        return ROUND_LABELS.get(self._lang(value_lang), {}).get(int(value), str(value))

    def _risk_label(self, code: str, value_lang: str) -> str:
        return RISK_LABELS.get(self._lang(value_lang), {}).get(code, code)

    def _adv_label(self, probability: float, value_lang: str) -> str:
        return ADV_LABELS[self._lang(value_lang)][1 if probability >= 0.5 else 0]

    def _final_label(self, probability: float, value_lang: str) -> str:
        return FINAL_LABELS[self._lang(value_lang)][1 if probability >= 0.5 else 0]

    def _feature_label(self, feature: str, value_lang: str) -> str:
        return FEATURE_LABELS.get(feature, {}).get(self._lang(value_lang), feature.replace("_", " "))

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
