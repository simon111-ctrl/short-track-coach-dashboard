from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

ADVANCEMENT_CODES = {
    "Q", "QA", "QB", "QAA", "QAB", "QBA", "QBB", "q", "qA", "qB", "ADV", "ADA", "ADVA", "ADVB"
}
SPECIAL_CODES = {"ADV", "ADA", "ADVA", "ADVB"}

def parse_time_to_seconds(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    if text.upper() in {"DNS", "DNF", "PEN", "YC", "DQ", "DSQ", "WDR"}:
        return np.nan
    text = text.replace(",", ".")
    match = re.search(r"(?:(\d+):)?(\d+(?:\.\d+)?)", text)
    if not match:
        return np.nan
    minutes = float(match.group(1) or 0)
    seconds = float(match.group(2))
    return minutes * 60 + seconds


def season_start(value) -> float:
    text = "" if pd.isna(value) else str(value)
    match = re.search(r"(20\d{2}|19\d{2})", text)
    return float(match.group(1)) if match else np.nan


def normalize_round(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    low = text.lower()
    if "semi" in low:
        return "Semifinals"
    if "quarter" in low:
        return "Quarterfinals"
    if "ranking final" in low:
        return "Ranking Finals"
    if "final a" in low:
        return "Final A"
    if "final b" in low:
        return "Final B"
    if "final" in low:
        return "Finals"
    if "rep. semi" in low or "repechage semi" in low:
        return "Rep. Semifinals"
    if "rep. quarter" in low or "repechage quarter" in low:
        return "Rep. Quarterfinals"
    if "rep. heat" in low or "repechage heat" in low:
        return "Rep. Heats"
    if "heat" in low:
        return "Heats"
    if "prelim" in low:
        return "Preliminaries"
    return text or "Unknown"


def clean_qual(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def is_advancing(code: str) -> int:
    if code in ADVANCEMENT_CODES:
        return 1
    return 0


def load_distance(distance: str, config: dict) -> tuple[pd.DataFrame, dict]:
    path = DATA_DIR / config["file"]
    df = pd.read_excel(path, sheet_name=distance)
    laps = config["laps"]

    df = df.copy()
    df["distance"] = distance
    df["athlete_name"] = df["姓名"].astype(str).str.strip()
    df["country"] = df.get("国家/地区", "").astype(str).str.strip()
    df["round_raw"] = df["轮次"].astype(str).str.strip()
    df["round"] = df["轮次"].map(normalize_round)
    df["qual_code"] = df.get("qual", "").map(clean_qual)
    df["season_start"] = df.get("赛季", "").map(season_start)
    df["official_total_time"] = df["总成绩"].map(parse_time_to_seconds)
    df["source_url_clean"] = df.get("source_url", "").astype(str).str.replace(r"\?round=.*$", "", regex=True)

    for lap in range(1, laps + 1):
        df[f"lap{lap}_position"] = pd.to_numeric(df[f"第{lap}圈位置"], errors="coerce")
        df[f"lap{lap}_time"] = pd.to_numeric(df[f"第{lap}圈成绩"], errors="coerce")

    lap_time_cols = [f"lap{i}_time" for i in range(1, laps + 1)]
    lap_pos_cols = [f"lap{i}_position" for i in range(1, laps + 1)]
    before_rows = len(df)
    complete_mask = df[lap_time_cols + lap_pos_cols].notna().all(axis=1)
    df = df.loc[complete_mask].reset_index(drop=True)
    removed_missing_lap = int(before_rows - len(df))

    df["reconstructed_total_time"] = df[lap_time_cols].sum(axis=1)
    df["total_time_was_missing"] = df["official_total_time"].isna().astype(int)
    df["official_total_time_filled"] = df["official_total_time"].fillna(df["reconstructed_total_time"])
    df["official_total_delta"] = df["official_total_time"] - df["reconstructed_total_time"]
    df["official_total_abs_delta"] = df["official_total_delta"].abs()
    df["timing_consistent"] = (
        df["official_total_time"].isna() | (df["official_total_abs_delta"] <= 0.10)
    ).astype(int)
    df["advancement_label"] = df["qual_code"].map(is_advancing).astype(int)
    df["special_adv_code_flag"] = df["qual_code"].isin(SPECIAL_CODES).astype(int)
    df["is_final_round"] = df["round"].isin(["Final A", "Final B", "Finals", "Ranking Finals"]).astype(int)
    df["sample_id"] = [f"{distance}_{i:06d}" for i in range(len(df))]
    df["athlete_event_key"] = df["distance"] + "|" + df["athlete_name"] + "|" + df["source_url_clean"]

    audit = {
        "distance": distance,
        "raw_rows": before_rows,
        "rows_removed_missing_any_lap_time_or_position": removed_missing_lap,
        "complete_lap_rows": int(len(df)),
        "official_total_missing_but_reconstructed": int(df["total_time_was_missing"].sum()),
        "timing_inconsistent_gt_0_10s": int((df["timing_consistent"] == 0).sum()),
        "qual_distribution": df["qual_code"].replace("", "(blank)").value_counts().to_dict(),
        "special_adv_distribution": df.loc[df["qual_code"].isin(SPECIAL_CODES), "qual_code"].value_counts().to_dict(),
        "round_distribution": df["round"].value_counts().to_dict(),
    }
    return df, audit


def add_common_features(df: pd.DataFrame, laps: int) -> pd.DataFrame:
    out = df.copy()
    time_cols = [f"lap{i}_time" for i in range(1, laps + 1)]
    pos_cols = [f"lap{i}_position" for i in range(1, laps + 1)]
    times = out[time_cols]
    pos = out[pos_cols]

    out["mean_lap_time"] = times.mean(axis=1)
    out["median_lap_time"] = times.median(axis=1)
    out["lap_time_std"] = times.std(axis=1)
    out["lap_time_cv"] = out["lap_time_std"] / out["mean_lap_time"].replace(0, np.nan)
    out["fastest_lap_time"] = times.min(axis=1)
    out["slowest_lap_time"] = times.max(axis=1)
    out["lap_time_range"] = out["slowest_lap_time"] - out["fastest_lap_time"]

    early_end = max(2, math.ceil(laps * 0.33))
    mid_start = early_end + 1
    mid_end = max(mid_start, math.ceil(laps * 0.67))
    late_start = mid_end + 1
    early_cols = [f"lap{i}_time" for i in range(1, early_end + 1)]
    mid_cols = [f"lap{i}_time" for i in range(mid_start, mid_end + 1)]
    late_cols = [f"lap{i}_time" for i in range(late_start, laps + 1)]
    if not late_cols:
        late_cols = [f"lap{laps}_time"]

    out["early_lap_mean_time"] = out[early_cols].mean(axis=1)
    out["mid_lap_mean_time"] = out[mid_cols].mean(axis=1)
    out["late_lap_mean_time"] = out[late_cols].mean(axis=1)
    out["start_ratio_to_mean"] = out["lap1_time"] / out["mean_lap_time"].replace(0, np.nan)
    out["finish_ratio_to_mean"] = out[f"lap{laps}_time"] / out["mean_lap_time"].replace(0, np.nan)
    out["finish_vs_mid_delta"] = out[f"lap{laps}_time"] - out["mid_lap_mean_time"]
    out["finish_vs_previous_delta"] = out[f"lap{laps}_time"] - out[f"lap{laps-1}_time"]

    out["start_position"] = out["lap1_position"]
    out["final_position"] = out[f"lap{laps}_position"]
    out["mean_position"] = pos.mean(axis=1)
    out["position_std"] = pos.std(axis=1)
    out["best_position"] = pos.min(axis=1)
    out["worst_position"] = pos.max(axis=1)
    out["position_range"] = out["worst_position"] - out["best_position"]
    out["position_gain_total"] = out["start_position"] - out["final_position"]
    out["position_stability_score"] = 1 / (1 + out["position_std"].fillna(0))

    gain_cols = []
    delta_cols = []
    for lap in range(2, laps + 1):
        delta_col = f"lap{lap}_time_delta_from_previous"
        gain_col = f"lap{lap}_position_gain_from_previous"
        out[delta_col] = out[f"lap{lap}_time"] - out[f"lap{lap-1}_time"]
        out[gain_col] = out[f"lap{lap-1}_position"] - out[f"lap{lap}_position"]
        delta_cols.append(delta_col)
        gain_cols.append(gain_col)

    gains = out[gain_cols]
    out["overtake_events_count"] = (gains > 0).sum(axis=1)
    out["position_loss_events_count"] = (gains < 0).sum(axis=1)
    out["positive_position_gain_sum"] = gains.clip(lower=0).sum(axis=1)
    out["negative_position_loss_sum"] = gains.clip(upper=0).abs().sum(axis=1)
    out["lap_time_delta_std"] = out[delta_cols].std(axis=1)

    return out


def add_distance_features(df: pd.DataFrame, distance: str, laps: int) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    features = []
    if distance == "500m":
        out["start_explosion_time"] = out["lap1_time"]
        out["first_turn_position"] = out["lap1_position"]
        out["early_position_gain_lap1_2"] = out["lap1_position"] - out["lap2_position"]
        out["first_two_lap_mean_time"] = out[["lap1_time", "lap2_time"]].mean(axis=1)
        out["explosive_stability_lap1_3_std"] = out[["lap1_time", "lap2_time", "lap3_time"]].std(axis=1)
        out["last_lap_conversion_delta"] = out["lap5_time"] - out["lap4_time"]
        out["late_position_conversion"] = out["lap4_position"] - out["lap5_position"]
        features = [
            "start_explosion_time",
            "first_turn_position",
            "early_position_gain_lap1_2",
            "first_two_lap_mean_time",
            "explosive_stability_lap1_3_std",
            "last_lap_conversion_delta",
            "late_position_conversion",
        ]
    elif distance == "1000m":
        out["front_mid_pace_mean_lap2_5"] = out[[f"lap{i}_time" for i in range(2, 6)]].mean(axis=1)
        out["mid_position_control_mean_lap3_6"] = out[[f"lap{i}_position" for i in range(3, 7)]].mean(axis=1)
        out["acceleration_lap6_vs_lap5"] = out["lap6_time"] - out["lap5_time"]
        out["speed_hold_lap6_8_std"] = out[["lap6_time", "lap7_time", "lap8_time"]].std(axis=1)
        out["sprint_reserve_lap9_vs_lap7_8"] = out["lap9_time"] - out[["lap7_time", "lap8_time"]].mean(axis=1)
        out["late_position_gain_lap7_9"] = out["lap7_position"] - out["lap9_position"]
        features = [
            "front_mid_pace_mean_lap2_5",
            "mid_position_control_mean_lap3_6",
            "acceleration_lap6_vs_lap5",
            "speed_hold_lap6_8_std",
            "sprint_reserve_lap9_vs_lap7_8",
            "late_position_gain_lap7_9",
        ]
    elif distance == "1500m":
        out["long_pace_early_mean_lap2_5"] = out[[f"lap{i}_time" for i in range(2, 6)]].mean(axis=1)
        out["long_pace_mid_mean_lap6_10"] = out[[f"lap{i}_time" for i in range(6, 11)]].mean(axis=1)
        out["long_pace_late_mean_lap11_14"] = out[[f"lap{i}_time" for i in range(11, 15)]].mean(axis=1)
        out["mid_energy_saving_delta"] = out["long_pace_mid_mean_lap6_10"] - out["long_pace_early_mean_lap2_5"]
        out["pace_layering_range"] = out[
            ["long_pace_early_mean_lap2_5", "long_pace_mid_mean_lap6_10", "long_pace_late_mean_lap11_14"]
        ].max(axis=1) - out[
            ["long_pace_early_mean_lap2_5", "long_pace_mid_mean_lap6_10", "long_pace_late_mean_lap11_14"]
        ].min(axis=1)
        out["attack_gain_lap10_12"] = out["lap10_position"] - out["lap12_position"]
        out["back_end_capacity_lap11_14_vs_mid"] = out["long_pace_late_mean_lap11_14"] - out["long_pace_mid_mean_lap6_10"]
        out["late_position_gain_lap10_14"] = out["lap10_position"] - out["lap14_position"]
        features = [
            "long_pace_early_mean_lap2_5",
            "long_pace_mid_mean_lap6_10",
            "long_pace_late_mean_lap11_14",
            "mid_energy_saving_delta",
            "pace_layering_range",
            "attack_gain_lap10_12",
            "back_end_capacity_lap11_14_vs_mid",
            "late_position_gain_lap10_14",
        ]
    return out, features


def feature_sets(distance: str, laps: int, distance_specific: list[str]) -> dict[str, list[str]]:
    lap_time_cols = [f"lap{i}_time" for i in range(1, laps + 1)]
    lap_pos_cols = [f"lap{i}_position" for i in range(1, laps + 1)]
    deltas = [f"lap{i}_time_delta_from_previous" for i in range(2, laps + 1)]
    gains = [f"lap{i}_position_gain_from_previous" for i in range(2, laps + 1)]
    base = [
        "median_lap_time",
        "lap_time_std",
        "lap_time_cv",
        "fastest_lap_time",
        "slowest_lap_time",
        "lap_time_range",
        "early_lap_mean_time",
        "mid_lap_mean_time",
        "late_lap_mean_time",
        "start_ratio_to_mean",
        "finish_ratio_to_mean",
        "finish_vs_mid_delta",
        "finish_vs_previous_delta",
        "start_position",
        "final_position",
        "mean_position",
        "position_std",
        "best_position",
        "worst_position",
        "position_range",
        "position_gain_total",
        "position_stability_score",
        "overtake_events_count",
        "position_loss_events_count",
        "positive_position_gain_sum",
        "negative_position_loss_sum",
        "lap_time_delta_std",
    ]
    advancement = (
        ["reconstructed_total_time", "official_total_time_filled", "mean_lap_time"]
        + lap_time_cols
        + lap_pos_cols
        + base
        + deltas
        + gains
        + distance_specific
    )
    grade_distance_specific = [
        c
        for c in distance_specific
        if any(
            token in c
            for token in [
                "position",
                "gain",
                "delta",
                "std",
                "range",
                "ratio",
                "stability",
                "conversion",
                "control",
                "attack",
                "capacity",
                "saving",
                "layering",
            ]
        )
        and "mean_time" not in c
        and not c.endswith("_time")
    ]
    grade_no_total = (
        lap_pos_cols
        + [
            "lap_time_std",
            "lap_time_cv",
            "lap_time_range",
            "start_ratio_to_mean",
            "finish_ratio_to_mean",
            "finish_vs_mid_delta",
            "finish_vs_previous_delta",
            "position_std",
            "position_range",
            "position_gain_total",
            "position_stability_score",
            "overtake_events_count",
            "position_loss_events_count",
            "positive_position_gain_sum",
            "negative_position_loss_sum",
            "lap_time_delta_std",
        ]
        + deltas
        + gains
        + grade_distance_specific
    )
    return {
        "advancement_postrace_features": list(dict.fromkeys(advancement)),
        "grade_no_total_features": list(dict.fromkeys(grade_no_total)),
        "round_depth_features": list(dict.fromkeys(advancement)),
        "final_entry_features": list(dict.fromkeys(advancement)),
        "tactical_style_features": list(dict.fromkeys(grade_no_total)),
        "key_lap_features": list(dict.fromkeys(advancement)),
        "style_cluster_features": style_cluster_features(list(dict.fromkeys(advancement))),
        "risk_detection_features": risk_features(list(dict.fromkeys(advancement))),
    }


def assign_grade_label(df: pd.DataFrame) -> pd.Series:
    labels = pd.qcut(
        df["reconstructed_total_time"].rank(method="first"),
        q=4,
        labels=[3, 2, 1, 0],
    )
    return labels.astype(int)


def add_extended_labels(df: pd.DataFrame, laps: int) -> pd.DataFrame:
    out = df.copy()
    out["round_score"] = out["round"].map(ROUND_SCORES).fillna(0).astype(int)
    out["max_round_score"] = out.groupby("athlete_event_key")["round_score"].transform("max").astype(int)
    out["final_entry_label"] = (out["max_round_score"] >= 8).astype(int)
    out["tactical_style_label"] = assign_tactical_style(out)
    out["key_lap_label"] = assign_key_lap(out, laps)
    return out


def assign_tactical_style(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(4, index=df.index, dtype=int)
    stable_cut = df["position_std"].quantile(0.33)
    mean_pos_cut = df["mean_position"].quantile(0.50)
    loss_cut = df["negative_position_loss_sum"].quantile(0.80)
    range_cut = df["position_range"].quantile(0.80)

    out.loc[(df["start_position"] <= 2) & (df["mean_position"] <= mean_pos_cut)] = 0
    out.loc[(df["position_gain_total"] >= 2) | (df["positive_position_gain_sum"] >= 3)] = 1
    out.loc[(df["position_std"] <= stable_cut) & (df["mean_position"] <= mean_pos_cut)] = 2
    out.loc[(df["negative_position_loss_sum"] >= loss_cut) | (df["position_range"] >= range_cut)] = 3
    return out


def assign_key_lap(df: pd.DataFrame, laps: int) -> pd.Series:
    scores = []
    for lap in range(1, laps + 1):
        time_col = f"lap{lap}_time"
        time_z = (df[time_col] - df[time_col].median()) / df[time_col].std(ddof=0)
        score = time_z.abs().fillna(0)
        if lap > 1:
            gain_col = f"lap{lap}_position_gain_from_previous"
            score = score + df[gain_col].abs().fillna(0) * 0.35
        if lap == laps:
            score = score + df["finish_vs_previous_delta"].abs().fillna(0) * 0.50
        scores.append(score.rename(lap))
    score_df = pd.concat(scores, axis=1)
    return score_df.idxmax(axis=1).astype(int)


def style_cluster_features(features: list[str]) -> list[str]:
    preferred = [
        "start_ratio_to_mean",
        "finish_ratio_to_mean",
        "finish_vs_mid_delta",
        "finish_vs_previous_delta",
        "lap_time_std",
        "lap_time_cv",
        "lap_time_delta_std",
        "start_position",
        "final_position",
        "mean_position",
        "position_std",
        "position_range",
        "position_gain_total",
        "overtake_events_count",
        "position_loss_events_count",
        "positive_position_gain_sum",
        "negative_position_loss_sum",
        "position_stability_score",
    ]
    return [f for f in preferred if f in features]


def risk_features(features: list[str]) -> list[str]:
    preferred = [
        "reconstructed_total_time",
        "official_total_time_filled",
        "mean_lap_time",
        "lap_time_std",
        "lap_time_cv",
        "lap_time_range",
        "lap_time_delta_std",
        "position_std",
        "position_range",
        "position_loss_events_count",
        "negative_position_loss_sum",
        "finish_vs_mid_delta",
        "finish_vs_previous_delta",
        "final_position",
        "position_gain_total",
    ]
    return [f for f in preferred if f in features]


