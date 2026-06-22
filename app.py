from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostClassifier, Pool


APP_DIR = Path(__file__).resolve().parent
GRADE_DIR = APP_DIR / "models" / "grade_main_no_total_mean"
ADV_DIR = APP_DIR / "models" / "advancement_postrace_no_adv"


FRIENDLY_NAMES = {
    "total_time": "Official total time",
    "lap1_pos": "Lap 1 position",
    "lap1_time": "Lap 1 time",
    "lap2_pos": "Lap 2 position",
    "lap2_time": "Lap 2 time",
    "lap3_pos": "Lap 3 position",
    "lap3_time": "Lap 3 time",
    "lap4_pos": "Lap 4 position",
    "lap4_time": "Lap 4 time",
    "lap5_pos": "Lap 5 position",
    "lap5_time": "Lap 5 time",
    "speed_12": "Lap 1-2 speed change",
    "speed_23": "Lap 2-3 speed change",
    "speed_34": "Lap 3-4 speed change",
    "speed_45": "Lap 4-5 speed change",
    "pos_12": "Lap 1-2 position change",
    "pos_23": "Lap 2-3 position change",
    "pos_34": "Lap 3-4 position change",
    "pos_45": "Lap 4-5 position change",
    "mean_lap": "Mean lap time",
    "speed_var": "Speed variability",
    "pos_stability": "Position stability",
    "start_ratio": "Start-speed ratio",
    "sprint_ratio": "Sprint-speed ratio",
    "pos_improvement": "Position improvement",
}


@st.cache_resource
def load_assets():
    grade_meta = json.loads((GRADE_DIR / "metadata.json").read_text(encoding="utf-8"))
    adv_meta = json.loads((ADV_DIR / "metadata.json").read_text(encoding="utf-8"))
    grade_model = CatBoostClassifier()
    grade_model.load_model(str(GRADE_DIR / "CatBoost.cbm"))
    adv_model = CatBoostClassifier()
    adv_model.load_model(str(ADV_DIR / "CatBoost.cbm"))
    return grade_model, adv_model, grade_meta, adv_meta


def compute_features(total_time: float, lap_times: list[float], lap_positions: list[int]) -> dict[str, float]:
    lap_times_arr = np.array(lap_times, dtype=float)
    lap_pos_arr = np.array(lap_positions, dtype=float)
    mean_lap = float(np.mean(lap_times_arr))
    return {
        "total_time": float(total_time),
        "lap1_pos": float(lap_positions[0]),
        "lap1_time": float(lap_times[0]),
        "lap2_pos": float(lap_positions[1]),
        "lap2_time": float(lap_times[1]),
        "lap3_pos": float(lap_positions[2]),
        "lap3_time": float(lap_times[2]),
        "lap4_pos": float(lap_positions[3]),
        "lap4_time": float(lap_times[3]),
        "lap5_pos": float(lap_positions[4]),
        "lap5_time": float(lap_times[4]),
        "speed_12": float(lap_times[1] - lap_times[0]),
        "speed_23": float(lap_times[2] - lap_times[1]),
        "speed_34": float(lap_times[3] - lap_times[2]),
        "speed_45": float(lap_times[4] - lap_times[3]),
        "pos_12": float(lap_positions[1] - lap_positions[0]),
        "pos_23": float(lap_positions[2] - lap_positions[1]),
        "pos_34": float(lap_positions[3] - lap_positions[2]),
        "pos_45": float(lap_positions[4] - lap_positions[3]),
        "mean_lap": mean_lap,
        "speed_var": float(np.std(lap_times_arr, ddof=0)),
        "pos_stability": float(np.std(lap_pos_arr, ddof=0)),
        "start_ratio": float(lap_times[0] / mean_lap) if mean_lap else 0.0,
        "sprint_ratio": float(lap_times[4] / mean_lap) if mean_lap else 0.0,
        "pos_improvement": float(lap_positions[0] - lap_positions[4]),
    }


def feature_frame(meta: dict, features: dict[str, float]) -> pd.DataFrame:
    all_order = [
        "total_time",
        "lap1_pos",
        "lap1_time",
        "lap2_pos",
        "lap2_time",
        "lap3_pos",
        "lap3_time",
        "lap4_pos",
        "lap4_time",
        "lap5_pos",
        "lap5_time",
        "speed_12",
        "speed_23",
        "speed_34",
        "speed_45",
        "pos_12",
        "pos_23",
        "pos_34",
        "pos_45",
        "mean_lap",
        "speed_var",
        "pos_stability",
        "start_ratio",
        "sprint_ratio",
        "pos_improvement",
    ]
    selected = all_order[-23:] if len(meta["feature_columns"]) == 23 else all_order
    values = [features[k] for k in selected]
    return pd.DataFrame([values], columns=meta["feature_columns"])


def class_probability(model: CatBoostClassifier, meta: dict, frame: pd.DataFrame) -> tuple[str, np.ndarray]:
    proba = np.asarray(model.predict_proba(frame))[0]
    labels = [str(x) for x in meta["class_labels"]]
    return labels[int(np.argmax(proba))], proba


def shap_table(model: CatBoostClassifier, meta: dict, frame: pd.DataFrame, friendly_order: list[str], target_index: int = 0) -> pd.DataFrame:
    pool = Pool(frame, feature_names=meta["feature_columns"])
    contrib = model.get_feature_importance(pool, type="ShapValues")
    arr = np.asarray(contrib)
    if arr.ndim == 3:
        values = arr[0, target_index, :-1]
    else:
        values = arr[0, :-1]
    names = friendly_order[-len(values):]
    table = pd.DataFrame(
        {
            "Feature": [FRIENDLY_NAMES.get(name, name) for name in names],
            "Contribution": values,
            "AbsContribution": np.abs(values),
        }
    ).sort_values("AbsContribution", ascending=False)
    return table.head(8)


def gauge(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number={"suffix": "%", "font": {"size": 28}},
            title={"text": title, "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#d8dee9",
                "steps": [
                    {"range": [0, 40], "color": "#f8d7da"},
                    {"range": [40, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#d1e7dd"},
                ],
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def lap_chart(lap_times: list[float], positions: list[int]) -> go.Figure:
    laps = [1, 2, 3, 4, 5]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=laps,
            y=lap_times,
            mode="lines+markers",
            name="Lap time",
            line=dict(color="#2f5d8c", width=3),
            marker=dict(size=9),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=laps,
            y=positions,
            mode="lines+markers",
            name="Position",
            line=dict(color="#c14953", width=3, dash="dot"),
            marker=dict(size=9),
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=310,
        margin=dict(l=40, r=45, t=20, b=35),
        xaxis=dict(title="Lap", dtick=1),
        yaxis=dict(title="Lap time (s)", rangemode="tozero"),
        yaxis2=dict(title="Position", overlaying="y", side="right", autorange="reversed", dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fbfcfd",
    )
    return fig


def contribution_bar(table: pd.DataFrame, title: str) -> go.Figure:
    ordered = table.iloc[::-1]
    colors = ["#2e7d62" if v >= 0 else "#b94a48" for v in ordered["Contribution"]]
    fig = go.Figure(go.Bar(x=ordered["Contribution"], y=ordered["Feature"], orientation="h", marker_color=colors))
    fig.update_layout(
        title=title,
        height=310,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Model contribution",
        yaxis_title=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fbfcfd",
    )
    return fig


def coach_notes(features: dict[str, float], adv_prob: float, grade_label: str) -> list[str]:
    notes: list[str] = []
    if features["lap5_pos"] <= 2:
        notes.append("Final-lap position is competition-ready: the athlete entered the decisive phase in a realistic advancement lane.")
    else:
        notes.append("Late-race position is the main review point: video should check first-corner entry, lane protection, and missed passing windows.")
    if features["pos_improvement"] > 0:
        notes.append("Position improved across the race, suggesting useful tactical recovery or overtaking effectiveness.")
    elif features["pos_improvement"] < 0:
        notes.append("Position deteriorated across the race; review whether speed loss, contact, or lane choice caused the drop.")
    if features["speed_var"] > 1.2:
        notes.append("Lap-time variability is high for a 500 m profile; check for technical disruption, contact, or pacing instability.")
    else:
        notes.append("Lap-time variability is controlled; the next review should focus on whether stable speed was converted into position.")
    if adv_prob >= 0.7 and grade_label in {"1", "2"}:
        notes.append("Overall profile is strong: maintain the current start-to-finish structure and refine race-specific details.")
    elif adv_prob < 0.45:
        notes.append("Advancement reference is low; this should be treated as a tactical-position warning rather than a final verdict.")
    return notes


def app() -> None:
    st.set_page_config(
        page_title="500 m Short-Track Coach Dashboard",
        page_icon="ST",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {background: #f8fafc; border: 1px solid #dce3ea; padding: 12px 14px; border-radius: 8px;}
        .coach-card {border: 1px solid #dce3ea; border-radius: 8px; padding: 14px 16px; background: #ffffff;}
        .small-note {color: #52616f; font-size: 0.9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    grade_model, adv_model, grade_meta, adv_meta = load_assets()
    screenshot_mode = st.query_params.get("screenshot", "")

    st.title("500 m World Cup Short-Track Coach Dashboard")
    st.caption("Explainable post-race reference system based on the final 2024-2025 lap-level machine-learning models.")

    preset = st.sidebar.selectbox(
        "Sample profile",
        ["Balanced finalist profile", "Fast but trapped profile", "Developing recovery profile", "Manual input"],
    )
    presets = {
        "Balanced finalist profile": ([7.30, 8.92, 8.78, 8.84, 8.71], [2, 2, 2, 1, 1], 42.55),
        "Fast but trapped profile": ([7.18, 8.72, 8.66, 8.80, 8.75], [3, 3, 4, 4, 4], 42.21),
        "Developing recovery profile": ([7.65, 9.35, 9.10, 8.98, 8.82], [5, 5, 4, 3, 3], 43.90),
        "Manual input": ([7.35, 8.95, 8.85, 8.90, 8.80], [3, 3, 2, 2, 2], 42.85),
    }
    default_times, default_pos, default_total = presets[preset]

    st.sidebar.header("Race input")
    athlete = st.sidebar.text_input("Athlete / race label", value="Athlete A - 500 m")
    adv_special = st.sidebar.checkbox("ADV special outcome", value=False)
    total_time = st.sidebar.number_input("Official total time (s)", min_value=30.0, max_value=120.0, value=float(default_total), step=0.01)

    st.sidebar.subheader("Lap times and positions")
    lap_times: list[float] = []
    lap_positions: list[int] = []
    for i in range(5):
        col_t, col_p = st.sidebar.columns([1.1, 0.9])
        with col_t:
            lap_times.append(st.number_input(f"Lap {i + 1} time", min_value=1.0, max_value=40.0, value=float(default_times[i]), step=0.01, key=f"time_{i}"))
        with col_p:
            lap_positions.append(st.number_input(f"Lap {i + 1} pos", min_value=1, max_value=8, value=int(default_pos[i]), step=1, key=f"pos_{i}"))

    features = compute_features(total_time, lap_times, lap_positions)
    friendly_order = list(FRIENDLY_NAMES.keys())
    grade_frame = feature_frame(grade_meta, features)
    adv_frame = feature_frame(adv_meta, features)

    grade_label, grade_proba = class_probability(grade_model, grade_meta, grade_frame)
    adv_label, adv_proba = class_probability(adv_model, adv_meta, adv_frame)
    adv_positive_prob = float(adv_proba[1]) if len(adv_proba) > 1 else float(adv_proba[0])

    grade_idx = int(np.argmax(grade_proba))
    grade_shap = shap_table(grade_model, grade_meta, grade_frame, friendly_order, target_index=grade_idx)
    adv_shap = shap_table(adv_model, adv_meta, adv_frame, friendly_order, target_index=1)

    top = st.columns([1.1, 1.0, 1.0, 1.0])
    top[0].metric("Race label", athlete)
    top[1].metric("Performance grade", f"Grade {grade_label}", "1 = fastest group")
    top[2].metric("Advancement reference", f"{adv_positive_prob:.1%}", "ordinary ADV excluded")
    top[3].metric("Final-lap position", f"{features['lap5_pos']:.0f}", f"improvement {features['pos_improvement']:+.0f}")

    if adv_special:
        st.warning("ADV was marked as a special outcome. The advancement model was trained for ordinary advancement reference after excluding ADV cases; interpret this race separately.")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Lap profile")
        st.plotly_chart(lap_chart(lap_times, lap_positions), use_container_width=True)
    with right:
        st.subheader("Model confidence")
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(gauge(float(np.max(grade_proba)), "Grade confidence", "#2f5d8c"), use_container_width=True)
        with g2:
            st.plotly_chart(gauge(adv_positive_prob, "Advancement reference", "#2e7d62"), use_container_width=True)

    if screenshot_mode == "explain":
        st.subheader("Model explanations")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(contribution_bar(grade_shap, "Grade model: main local contributions"), use_container_width=True)
        with c2:
            st.plotly_chart(contribution_bar(adv_shap, "Advancement model: main local contributions"), use_container_width=True)
        st.markdown(
            '<p class="small-note">Positive and negative bars show local model contributions for the current race profile. They support review, not causal claims.</p>',
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3 = st.tabs(["Coach interpretation", "Model explanations", "Feature table"])
    with tab1:
        st.subheader("Coach-facing notes")
        for note in coach_notes(features, adv_positive_prob, grade_label):
            st.markdown(f'<div class="coach-card">{note}</div>', unsafe_allow_html=True)
            st.write("")
        st.markdown(
            '<p class="small-note">This dashboard is a post-race decision-support tool. It summarizes model evidence for coach review and does not replace video analysis or expert judgement.</p>',
            unsafe_allow_html=True,
        )
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(contribution_bar(grade_shap, "Grade model: main local contributions"), use_container_width=True)
        with c2:
            st.plotly_chart(contribution_bar(adv_shap, "Advancement model: main local contributions"), use_container_width=True)
    with tab3:
        display = pd.DataFrame(
            {
                "Feature": [FRIENDLY_NAMES[k] for k in friendly_order],
                "Value": [features[k] for k in friendly_order],
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption("Derived features are computed automatically from official total time, five lap times, and five lap positions.")

    st.divider()
    st.caption(
        "Final grouped-validation metrics: grade model weighted F1 = 0.920, macro AUC = 0.982; advancement model weighted F1 = 0.823, AUC = 0.894."
    )


if __name__ == "__main__":
    app()
