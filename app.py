from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from short_track_service import (
    DISTANCE_LAPS,
    FeatureAlignmentError,
    PredictionService,
    format_seconds_mmss,
    normalize_gender,
    parse_time_to_seconds,
)


APP_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = APP_DIR / "model_package"

TEXT = {
    "zh": {
        "title": "短道速滑精英特征预测模型",
        "subtitle": "按性别分别训练，支持 500m / 1000m / 1500m 赛后特征分析",
        "language": "界面语言",
        "distance": "项目",
        "gender": "性别",
        "mode": "分析方式",
        "single": "单场分析",
        "batch": "批量分析",
        "sample": "使用样例数据",
        "athlete": "运动员 / 样本名",
        "round": "轮次",
        "qual": "晋级标记",
        "official_total": "官方总成绩，可留空",
        "calculated_total": "自动计算总成绩",
        "run": "开始分析",
        "model_info": "模型信息",
        "rows": "训练样本",
        "version": "模型版本",
        "result": "预测结果",
        "grade": "成绩等级",
        "advancement": "晋级参考",
        "final": "决赛参考",
        "round_pred": "最高轮次",
        "style": "战术风格",
        "rhythm": "节奏类型",
        "key": "关键圈",
        "risk": "风险状态",
        "summary": "分段趋势",
        "advice": "训练建议",
        "explain": "模型解释",
        "global": "模型最关注的指标",
        "local": "本场关键指标值",
        "download": "下载结果",
        "template": "下载模板",
        "upload": "上传 CSV / Excel",
        "batch_help": "批量上传说明",
        "missing": "缺少标准列",
        "note": "本模型按男、女分开训练。选择性别后只调用对应性别模型，预测结果仅作为训练和科研辅助参考，不等同于最终选材结论。",
        "time_hint": "时间可输入 00:42:318、01:26:542、m:ss、ss 或 42.318；最后三位为毫秒。",
        "lap_time": "第{lap}圈成绩",
        "lap_position": "第{lap}圈位置",
        "project": "项目",
        "selected_gender": "性别",
        "model_rule": "当前预测只和对应性别样本的精英特征进行比较，不与另一性别模型互相比较。",
        "report": "下载单场报告",
        "task": "解释任务",
        "display": "结果表",
    },
    "en": {
        "title": "Short-Track Elite Feature Predictor",
        "subtitle": "Gender-specific models for 500m / 1000m / 1500m post-race analysis",
        "language": "Language",
        "distance": "Event",
        "gender": "Gender",
        "mode": "Analysis mode",
        "single": "Single race",
        "batch": "Batch analysis",
        "sample": "Use sample data",
        "athlete": "Athlete / sample",
        "round": "Round",
        "qual": "Qualification code",
        "official_total": "Official total time, optional",
        "calculated_total": "Calculated total time",
        "run": "Analyze",
        "model_info": "Model info",
        "rows": "Training rows",
        "version": "Model version",
        "result": "Prediction result",
        "grade": "Performance grade",
        "advancement": "Advancement reference",
        "final": "Final reference",
        "round_pred": "Highest round",
        "style": "Tactical style",
        "rhythm": "Rhythm type",
        "key": "Key lap",
        "risk": "Risk status",
        "summary": "Segment trend",
        "advice": "Training advice",
        "explain": "Model explanation",
        "global": "Top model features",
        "local": "Current race values",
        "download": "Download results",
        "template": "Download template",
        "upload": "Upload CSV / Excel",
        "batch_help": "Batch upload guide",
        "missing": "Missing standard columns",
        "note": "The model is trained separately by gender. After selecting gender, the app calls only the matching gender model. Results are training and research references, not final selection decisions.",
        "time_hint": "Accepted formats: 00:42:318, 01:26:542, m:ss, ss, or 42.318. The final three digits are milliseconds.",
        "lap_time": "Lap {lap} time",
        "lap_position": "Lap {lap} position",
        "project": "Event",
        "selected_gender": "Gender",
        "model_rule": "This prediction is compared only with the selected gender reference group.",
        "report": "Download single-race report",
        "task": "Explanation task",
        "display": "Result table",
    },
}

GENDER_OPTIONS = {
    "zh": {"请选择性别": "", "男 / Male / Men": "male", "女 / Female / Women": "female"},
    "en": {"Select gender": "", "Male / Men / 男": "male", "Female / Women / 女": "female"},
}

TASK_OPTIONS = {
    "grade": {"zh": "成绩等级", "en": "Performance grade"},
    "advancement": {"zh": "晋级参考", "en": "Advancement"},
    "final_entry": {"zh": "决赛参考", "en": "Final entry"},
    "max_round": {"zh": "最高轮次", "en": "Highest round"},
    "tactical_style": {"zh": "战术风格", "en": "Tactical style"},
    "key_lap": {"zh": "关键圈", "en": "Key lap"},
}


@st.cache_resource
def get_service() -> PredictionService:
    return PredictionService(PACKAGE_DIR)


def lang() -> str:
    return st.session_state.lang


def t(key: str) -> str:
    return TEXT[lang()][key]


def sample_frame(distance: str, gender: str) -> pd.DataFrame:
    return pd.read_csv(PACKAGE_DIR / "examples" / f"example_input_{gender}_{distance}_advancement.csv")


def template_frame(distance: str, gender: str) -> pd.DataFrame:
    frame = sample_frame(distance, gender).head(1).copy()
    cols = [col for col in get_service().required_columns(distance) if col in frame.columns]
    return frame[cols]


def metric(label: str, value: object, delta: str | None = None, help_text: str | None = None) -> None:
    st.metric(label, value, delta=delta, help=help_text)


def safe_error(error: Exception) -> None:
    if isinstance(error, FeatureAlignmentError):
        st.error("模型特征字段不匹配。")
        st.code(str(error), language="text")
    else:
        st.error(str(error))


def read_total_from_inputs(values: list[str]) -> tuple[float | None, str]:
    seconds = []
    for value in values:
        if not str(value).strip():
            return None, ""
        seconds.append(parse_time_to_seconds(value))
    total = float(sum(seconds))
    return total, format_seconds_mmss(total)


def build_single_raw(distance: str, gender: str, defaults: dict[str, object], use_sample: bool) -> pd.DataFrame | None:
    laps = DISTANCE_LAPS[distance]
    st.markdown(f"### {t('single')}")
    st.caption(t("time_hint"))

    with st.container(border=True):
        top = st.columns([1.2, 0.8, 0.8, 1.0])
        athlete = top[0].text_input(t("athlete"), value=str(defaults.get("athlete_name", "")) if use_sample else "")
        round_name = top[1].text_input(t("round"), value=str(defaults.get("round", "")) if use_sample else "")
        qual = top[2].text_input(t("qual"), value=str(defaults.get("qual_code", "")) if use_sample else "")
        official_default = defaults.get("official_total_time", "") if use_sample else ""
        official_total = top[3].text_input(t("official_total"), value="" if pd.isna(official_default) else str(official_default))

        lap_values: list[str] = []
        position_values: list[int] = []
        for start in range(1, laps + 1, 3):
            cols = st.columns(min(3, laps - start + 1))
            for offset, col in enumerate(cols):
                lap = start + offset
                with col:
                    st.markdown(f"**Lap {lap}**")
                    default_time = defaults.get(f"lap{lap}_time", "")
                    if use_sample and default_time != "":
                        default_time = format_seconds_mmss(float(default_time))
                    lap_values.append(
                        st.text_input(
                            t("lap_time").format(lap=lap),
                            value=str(default_time),
                            key=f"time_{distance}_{gender}_{lap}",
                        )
                    )
                    position_values.append(
                        st.number_input(
                            t("lap_position").format(lap=lap),
                            min_value=0,
                            max_value=12,
                            value=int(float(defaults.get(f"lap{lap}_position", 1) or 1)) if use_sample else 1,
                            step=1,
                            key=f"pos_{distance}_{gender}_{lap}",
                        )
                    )

        try:
            total_seconds, total_display = read_total_from_inputs(lap_values)
        except ValueError as exc:
            total_seconds, total_display = None, ""
            st.warning(str(exc))
        st.metric(t("calculated_total"), total_display or "-")

    submitted = st.button(t("run"), type="primary", use_container_width=True)
    if not submitted:
        return None

    if total_seconds is None:
        st.error(t("time_hint"))
        return None

    row = {
        "athlete_name": athlete,
        "gender": gender,
        "distance": distance,
        "round": round_name,
        "qual_code": qual,
        "official_total_time": official_total,
    }
    for lap, value in enumerate(lap_values, start=1):
        row[f"lap{lap}_time"] = value
    for lap, value in enumerate(position_values, start=1):
        row[f"lap{lap}_position"] = value
    return pd.DataFrame([row])


def lap_chart(row: pd.Series, distance: str) -> go.Figure:
    laps = list(range(1, DISTANCE_LAPS[distance] + 1))
    times = [parse_time_to_seconds(row[f"lap{i}_time"]) for i in laps]
    pos = [float(row[f"lap{i}_position"]) for i in laps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=laps, y=times, mode="lines+markers", name="Lap time", line={"color": "#1f5f8b", "width": 3}))
    fig.add_trace(go.Scatter(x=laps, y=pos, mode="lines+markers", name="Position", yaxis="y2", line={"color": "#b24a4a", "width": 3, "dash": "dot"}))
    fig.update_layout(
        height=320,
        margin={"l": 30, "r": 35, "t": 20, "b": 30},
        xaxis={"dtick": 1},
        yaxis={"title": "Seconds"},
        yaxis2={"title": "Position", "overlaying": "y", "side": "right", "autorange": "reversed", "dtick": 1},
        legend={"orientation": "h", "y": 1.05},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fbfcfe",
    )
    return fig


def make_report(distance: str, gender: str, row: pd.Series, advice: list[dict[str, str]]) -> str:
    lines = [
        f"# Short-Track Elite Model Report - {gender} {distance}",
        "",
        f"- Event: {distance}",
        f"- Gender: {gender}",
        f"- Calculated total time: {row.get('calculated_total_time_display', '')}",
        f"- Grade: {row.get('grade', '')} ({float(row.get('grade_probability', 0)):.1%})",
        f"- Advancement: {row.get('advancement_reference', '')} ({float(row.get('advancement_probability', 0)):.1%})",
        f"- Final entry: {row.get('final_entry_reference', '')} ({float(row.get('final_entry_probability', 0)):.1%})",
        f"- Key lap: {row.get('key_lap', '')}",
        f"- Tactical style: {row.get('tactical_style', '')}",
        "",
        "## Advice",
    ]
    for item in advice:
        lines.append(f"- {item['title']}: {item['body']}")
    return "\n".join(lines)


def output_view(output: pd.DataFrame, raw: pd.DataFrame, distance: str, gender: str) -> None:
    row = output.iloc[0]
    st.subheader(t("result"))
    c0, c1, c2 = st.columns(3)
    c0.metric(t("project"), distance)
    c1.metric(t("selected_gender"), "男 / Male" if gender == "male" else "女 / Female")
    c2.metric(t("calculated_total"), row["calculated_total_time_display"])
    st.caption(t("model_rule"))

    cols = st.columns(4)
    cols[0].metric(t("grade"), row["grade"], f"{float(row['grade_probability']):.0%}")
    cols[1].metric(t("advancement"), row["advancement_reference"], f"{float(row['advancement_probability']):.0%}")
    cols[2].metric(t("final"), row["final_entry_reference"], f"{float(row['final_entry_probability']):.0%}")
    cols[3].metric(t("round_pred"), row["max_round"], f"{float(row['max_round_probability']):.0%}")

    cols = st.columns(4)
    cols[0].metric(t("style"), row["tactical_style"])
    cols[1].metric(t("rhythm"), row["rhythm_type"])
    cols[2].metric(t("key"), row["key_lap"])
    cols[3].metric(t("risk"), row["risk_label"], f"{float(row['risk_score']):.3f}")

    left, right = st.columns([1.15, 0.85], vertical_alignment="top")
    with left:
        st.subheader(t("summary"))
        st.plotly_chart(lap_chart(raw.iloc[0], distance), use_container_width=True)
    with right:
        st.subheader(t("advice"))
        advice = get_service().generate_advice(row, distance, gender, lang())
        for item in advice:
            st.markdown(
                f"""
                <div style="border:1px solid #d7dee8;border-radius:8px;padding:0.75rem 0.85rem;margin-bottom:0.6rem;background:#fff;">
                  <div style="font-size:0.85rem;font-weight:700;margin-bottom:0.25rem">{item['title']}</div>
                  <div style="font-size:0.96rem;line-height:1.55">{item['body']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    tab1, tab2 = st.tabs([t("explain"), t("display")])
    with tab1:
        labels = {TASK_OPTIONS[k][lang()]: k for k in TASK_OPTIONS}
        selected = st.selectbox(t("task"), list(labels), index=0)
        task = labels[selected]
        global_df = get_service().global_explanation(distance, gender, task, lang())
        local_df = get_service().local_explanation(distance, gender, task, raw.head(1), lang())
        g1, g2 = st.columns(2)
        with g1:
            st.markdown(f"#### {t('global')}")
            if global_df.empty:
                st.info("No explanation data.")
            else:
                bar = go.Figure(go.Bar(x=global_df["importance_mean"], y=global_df["feature_label"], orientation="h", marker={"color": "#1f5f8b"}))
                bar.update_layout(height=330, margin={"l": 20, "r": 20, "t": 10, "b": 20})
                st.plotly_chart(bar, use_container_width=True)
        with g2:
            st.markdown(f"#### {t('local')}")
            if local_df.empty:
                st.info("No local values.")
            else:
                st.dataframe(local_df[["feature_label", "value"]], hide_index=True, use_container_width=True)
    with tab2:
        display_cols = [
            "athlete_name",
            "gender",
            "distance",
            "calculated_total_time_display",
            "grade",
            "advancement_reference",
            "final_entry_reference",
            "max_round",
            "tactical_style",
            "rhythm_type",
            "key_lap",
            "risk_label",
        ]
        st.dataframe(output[[c for c in display_cols if c in output]], hide_index=True, use_container_width=True)

    st.download_button(
        t("report"),
        make_report(distance, gender, row, get_service().generate_advice(row, distance, gender, lang())),
        file_name=f"short_track_{gender}_{distance}_report.md",
        mime="text/markdown",
    )


def batch_upload(distance: str, gender: str) -> None:
    st.subheader(t("batch_help"))
    st.caption(t("time_hint"))
    st.dataframe(get_service().required_columns_label_table(distance, lang()), hide_index=True, use_container_width=True)
    st.download_button(
        t("template"),
        template_frame(distance, gender).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"template_{gender}_{distance}.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader(t("upload"), type=["csv", "xlsx", "xls"])
    if not uploaded:
        return
    raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
    required = get_service().required_columns(distance)
    missing = [c for c in required if c not in raw.columns and c != "gender"]
    if missing:
        st.error(f"{t('missing')}: {', '.join(missing)}")
        return
    raw["gender"] = gender
    raw["distance"] = distance
    try:
        output = get_service().predict(distance, gender, raw, lang())
    except Exception as exc:
        safe_error(exc)
        return
    display_cols = [c for c in ["athlete_name", "gender", "distance", "calculated_total_time_display", "grade", "advancement_reference", "final_entry_reference", "max_round", "tactical_style", "rhythm_type", "key_lap", "risk_label"] if c in output]
    st.dataframe(output[display_cols], hide_index=True, use_container_width=True)
    buffer = BytesIO()
    output.to_excel(buffer, index=False)
    st.download_button(t("download"), buffer.getvalue(), file_name=f"short_track_{gender}_{distance}_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def render_header(distance: str, gender: str) -> None:
    summary = get_service().gender_distance_summary(distance, gender)
    model = summary["model"]
    stats = summary["stats"]
    left, right = st.columns([1.35, 0.65], vertical_alignment="center")
    with left:
        st.title(t("title"))
        st.caption(t("subtitle"))
    with right:
        st.markdown(
            f"""
            <div style="text-align:right;padding-top:0.3rem;">
              <div style="font-size:0.8rem;opacity:0.72">{t('version')}</div>
              <div style="font-size:1rem;font-weight:700;line-height:1.35">{get_service().manifest.get('created_at', '')}</div>
              <div style="font-size:0.8rem;opacity:0.72;margin-top:0.25rem">{t('rows')}: {int(model.get('n_training_rows', stats.get('sample_size', 0))):,}</div>
              <div style="font-size:0.8rem;opacity:0.72">{distance} / {gender}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="短道速滑精英特征预测模型", page_icon="ST", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1380px;}
        h1 {line-height: 1.25; padding-top: 0.2rem; padding-bottom: 0.15rem;}
        div[data-testid="stMetric"] {
            background: #f8fbfe;
            border: 1px solid #d8e1ea;
            border-radius: 8px;
            padding: 0.85rem 0.9rem;
        }
        div[data-testid="stMetricLabel"] {font-size: 0.78rem; line-height: 1.2;}
        div[data-testid="stMetricValue"] {font-size: 1.2rem; line-height: 1.2;}
        section[data-testid="stSidebar"] {border-right: 1px solid #e6ebf0;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "lang" not in st.session_state:
        st.session_state.lang = "zh"
    if "single_output" not in st.session_state:
        st.session_state.single_output = None
        st.session_state.single_raw = None
        st.session_state.single_key = None

    with st.sidebar:
        language_choice = st.selectbox(t("language"), ["中文", "English"], index=0 if st.session_state.lang == "zh" else 1)
        st.session_state.lang = "zh" if language_choice == "中文" else "en"
        distance = st.radio(t("distance"), ["500m", "1000m", "1500m"], horizontal=True)
        gender_label = st.selectbox(t("gender"), list(GENDER_OPTIONS[lang()].keys()), index=0)
        gender_value = GENDER_OPTIONS[lang()][gender_label]
        mode = st.radio(t("mode"), [t("single"), t("batch")])
        use_sample = st.checkbox(t("sample"), value=True)
        if not gender_value:
            st.warning("请先选择性别。" if lang() == "zh" else "Please select gender first.")
            st.stop()
        gender = normalize_gender(gender_value)
        st.markdown(f"**{t('model_info')}**")
        meta = get_service().gender_distance_summary(distance, gender)
        st.caption(f"{t('version')}: {get_service().manifest.get('created_at', '')}")
        st.caption(f"{t('rows')}: {int(meta['model'].get('n_training_rows', meta['stats'].get('sample_size', 0))):,}")
        st.caption(f"{t('project')}: {distance}")
        st.caption(f"{t('selected_gender')}: {gender}")
        st.markdown("---")
        st.download_button(
            t("template"),
            template_frame(distance, gender).to_csv(index=False).encode("utf-8-sig"),
            file_name=f"template_{gender}_{distance}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    render_header(distance, gender)
    st.info(t("note"))

    if mode == t("single"):
        defaults = sample_frame(distance, gender).iloc[0].to_dict() if use_sample else {}
        raw = build_single_raw(distance, gender, defaults, use_sample)
        if raw is not None:
            try:
                st.session_state.single_output = get_service().predict(distance, gender, raw, lang())
                st.session_state.single_raw = raw
                st.session_state.single_key = (distance, gender)
            except Exception as exc:
                safe_error(exc)
                st.stop()
        if st.session_state.single_output is None or st.session_state.single_key != (distance, gender):
            st.stop()
        output_view(st.session_state.single_output, st.session_state.single_raw, distance, gender)
    else:
        batch_upload(distance, gender)


if __name__ == "__main__":
    main()
