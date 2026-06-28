from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from short_track_service import DISTANCE_LAPS, PredictionService


APP_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = APP_DIR / "model_package"

TEXT = {
    "zh": {
        "title": "短道速滑多距离教练工作台",
        "subtitle": "500m / 1000m / 1500m 赛后模型参考系统",
        "language": "语言",
        "distance": "距离",
        "mode": "分析方式",
        "single": "单场输入",
        "batch": "批量上传",
        "model": "模型版本",
        "training": "训练数据行数",
        "manual": "手动输入单场数据",
        "sample": "使用示例数据",
        "athlete": "运动员 / 样本名",
        "round": "轮次",
        "qual": "晋级代码",
        "total": "官方总成绩（秒，可留空后按圈速求和）",
        "lap_time": "第 {lap} 圈成绩",
        "lap_pos": "第 {lap} 圈位置",
        "run": "开始分析",
        "conclusion": "比赛结论",
        "grade": "成绩等级",
        "adv": "晋级参考",
        "final": "决赛入围概率",
        "round_pred": "最高轮次预测",
        "style": "战术风格",
        "rhythm": "节奏类型",
        "key": "关键圈",
        "risk": "异常风险",
        "explain": "模型解释",
        "global": "全局重要因素",
        "local": "当前样本重点",
        "notes": "教练建议",
        "upload": "上传 CSV 或 Excel",
        "download": "下载分析结果",
        "report": "下载单场报告",
        "template": "下载输入模板",
        "warning": "本工具为赛后决策支持，不替代录像分析和教练判断。",
        "missing": "上传文件缺少以下字段：",
    },
    "en": {
        "title": "Short-Track Multi-Distance Coach Workspace",
        "subtitle": "Post-race model reference for 500m / 1000m / 1500m",
        "language": "Language",
        "distance": "Distance",
        "mode": "Analysis mode",
        "single": "Single race",
        "batch": "Batch upload",
        "model": "Model version",
        "training": "Training rows",
        "manual": "Manual single-race input",
        "sample": "Use sample data",
        "athlete": "Athlete / sample name",
        "round": "Round",
        "qual": "Qualification code",
        "total": "Official total time (seconds, optional)",
        "lap_time": "Lap {lap} time",
        "lap_pos": "Lap {lap} position",
        "run": "Analyze",
        "conclusion": "Race conclusion",
        "grade": "Performance grade",
        "adv": "Advancement reference",
        "final": "Final-entry probability",
        "round_pred": "Highest round prediction",
        "style": "Tactical style",
        "rhythm": "Rhythm type",
        "key": "Key lap",
        "risk": "Anomaly risk",
        "explain": "Model explanation",
        "global": "Global drivers",
        "local": "Current sample focus",
        "notes": "Coach notes",
        "upload": "Upload CSV or Excel",
        "download": "Download results",
        "report": "Download single-race report",
        "template": "Download input template",
        "warning": "This is post-race decision support, not a replacement for video review or coach judgement.",
        "missing": "Uploaded file is missing these columns:",
    },
}


@st.cache_resource
def service() -> PredictionService:
    return PredictionService(PACKAGE_DIR)


def t(key: str) -> str:
    return TEXT[st.session_state.lang][key]


def sample_frame(distance: str) -> pd.DataFrame:
    path = PACKAGE_DIR / "examples" / f"example_input_{distance}_grade.csv"
    return pd.read_csv(path)


def template_frame(distance: str) -> pd.DataFrame:
    sample = sample_frame(distance).head(1).copy()
    return sample[service().required_columns(distance)]


def manual_frame(distance: str, use_sample: bool) -> pd.DataFrame:
    laps = DISTANCE_LAPS[distance]
    defaults = sample_frame(distance).iloc[0].to_dict() if use_sample else {}
    with st.form("manual_form"):
        st.subheader(t("manual"))
        c1, c2, c3, c4 = st.columns([1.2, 0.8, 0.8, 1.0])
        athlete = c1.text_input(t("athlete"), value=str(defaults.get("athlete_name", "Athlete A")))
        round_name = c2.text_input(t("round"), value=str(defaults.get("round", "Heats")))
        qual = c3.text_input(t("qual"), value=str(defaults.get("qual_code", "")))
        total_default = float(defaults.get("official_total_time", 0) or 0)
        total = c4.number_input(t("total"), min_value=0.0, max_value=300.0, value=total_default, step=0.001)

        lap_times = []
        lap_positions = []
        for start in range(1, laps + 1, 4):
            cols = st.columns(min(4, laps - start + 1))
            for offset, col in enumerate(cols):
                lap = start + offset
                with col:
                    lap_times.append(
                        st.number_input(
                            t("lap_time").format(lap=lap),
                            min_value=0.0,
                            max_value=60.0,
                            value=float(defaults.get(f"lap{lap}_time", 0) or 0),
                            step=0.001,
                            key=f"time_{distance}_{lap}",
                        )
                    )
                    lap_positions.append(
                        st.number_input(
                            t("lap_pos").format(lap=lap),
                            min_value=1,
                            max_value=12,
                            value=int(float(defaults.get(f"lap{lap}_position", 1) or 1)),
                            step=1,
                            key=f"pos_{distance}_{lap}",
                        )
                    )
        submitted = st.form_submit_button(t("run"), type="primary")
    if not submitted:
        return pd.DataFrame()
    row = {
        "athlete_name": athlete,
        "distance": distance,
        "round": round_name,
        "qual_code": qual,
        "official_total_time": total if total > 0 else None,
    }
    for i, value in enumerate(lap_times, 1):
        row[f"lap{i}_time"] = value
    for i, value in enumerate(lap_positions, 1):
        row[f"lap{i}_position"] = value
    row["reconstructed_total_time"] = sum(lap_times)
    return pd.DataFrame([row])


def line_chart(row: pd.Series, distance: str) -> go.Figure:
    laps = list(range(1, DISTANCE_LAPS[distance] + 1))
    times = [row[f"lap{i}_time"] for i in laps]
    pos = [row[f"lap{i}_position"] for i in laps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=laps, y=times, name="Lap time", mode="lines+markers", line={"color": "#245b8f"}))
    fig.add_trace(
        go.Scatter(
            x=laps,
            y=pos,
            name="Position",
            mode="lines+markers",
            yaxis="y2",
            line={"color": "#b84a4a", "dash": "dot"},
        )
    )
    fig.update_layout(
        height=310,
        margin={"l": 30, "r": 35, "t": 20, "b": 30},
        xaxis={"dtick": 1, "title": "Lap"},
        yaxis={"title": "Seconds"},
        yaxis2={"title": "Position", "overlaying": "y", "side": "right", "autorange": "reversed", "dtick": 1},
        legend={"orientation": "h", "y": 1.08},
    )
    return fig


def show_results(raw: pd.DataFrame, distance: str) -> pd.DataFrame:
    svc = service()
    output = svc.predict(distance, raw)
    first = output.iloc[0]

    st.subheader(t("conclusion"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("grade"), str(first["grade"]), f"{first['grade_probability']:.0%}")
    c2.metric(t("adv"), str(first["advancement_reference"]), f"{first['advancement_probability']:.0%}")
    c3.metric(t("final"), f"{first['final_entry_probability']:.0%}", str(first["final_entry_reference"]))
    c4.metric(t("round_pred"), str(first["max_round"]), f"{first['max_round_probability']:.0%}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(t("style"), str(first["tactical_style"]))
    c6.metric(t("rhythm"), str(first["rhythm_type"]))
    c7.metric(t("key"), str(first["key_lap"]))
    c8.metric(t("risk"), str(first["risk_label"]), f"{first['risk_score']:.3f}")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.plotly_chart(line_chart(raw.iloc[0], distance), use_container_width=True)
    with right:
        st.subheader(t("notes"))
        for note in svc.coach_notes(distance, first):
            st.info(note)

    st.subheader(t("explain"))
    tabs = st.tabs([t("global"), t("local")])
    with tabs[0]:
        task = st.selectbox("Task", ["advancement", "grade", "final_entry", "max_round", "tactical_style", "key_lap"])
        st.dataframe(svc.global_explanation(distance, task), hide_index=True, use_container_width=True)
    with tabs[1]:
        st.dataframe(svc.local_explanation(distance, "advancement", raw), hide_index=True, use_container_width=True)

    report = build_report(distance, first, svc.coach_notes(distance, first))
    st.download_button(t("report"), report, file_name=f"short_track_{distance}_report.md", mime="text/markdown")
    return output


def build_report(distance: str, row: pd.Series, notes: list[str]) -> str:
    lines = [
        f"# Short-Track Coach Report - {distance}",
        "",
        f"- Athlete/sample: {row.get('athlete_name', '')}",
        f"- Performance grade: {row['grade']} ({row['grade_probability']:.1%})",
        f"- Advancement reference: {row['advancement_reference']} ({row['advancement_probability']:.1%})",
        f"- Final-entry probability: {row['final_entry_probability']:.1%}",
        f"- Highest round prediction: {row['max_round']}",
        f"- Tactical style: {row['tactical_style']}",
        f"- Rhythm type: {row['rhythm_type']}",
        f"- Key lap: {row['key_lap']}",
        f"- Risk: {row['risk_label']} ({row['risk_score']:.3f})",
        "",
        "## Coach Notes",
    ]
    lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines)


def read_upload(uploaded) -> pd.DataFrame:
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


def output_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="Short-Track Coach Workspace", page_icon="ST", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {background:#f8fafc;border:1px solid #d7dee8;border-radius:8px;padding:12px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if "lang" not in st.session_state:
        st.session_state.lang = "zh"

    with st.sidebar:
        lang_label = st.selectbox("Language / 语言", ["中文", "English"])
        st.session_state.lang = "zh" if lang_label == "中文" else "en"
        distance = st.radio(t("distance"), ["500m", "1000m", "1500m"], horizontal=True)
        mode = st.radio(t("mode"), [t("single"), t("batch")])
        use_sample = st.checkbox(t("sample"), value=True)
        meta = service().manifest["models"][f"{distance}_advancement"]
        st.caption(f"{t('model')}: {service().manifest['created_at']}")
        st.caption(f"{t('training')}: {meta['n_training_rows']:,}")
        st.download_button(
            t("template"),
            template_frame(distance).to_csv(index=False).encode("utf-8-sig"),
            file_name=f"template_{distance}.csv",
            mime="text/csv",
        )

    st.title(t("title"))
    st.caption(t("subtitle"))
    st.warning(t("warning"))

    if mode == t("single"):
        raw = manual_frame(distance, use_sample)
        if not raw.empty:
            show_results(raw, distance)
    else:
        uploaded = st.file_uploader(t("upload"), type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            raw = read_upload(uploaded)
            required = service().required_columns(distance)
            missing = [col for col in required if col not in raw.columns]
            if missing:
                st.error(f"{t('missing')} {', '.join(missing)}")
            else:
                output = service().predict(distance, raw)
                st.dataframe(output, hide_index=True, use_container_width=True)
                st.download_button(
                    t("download"),
                    output_bytes(output),
                    file_name=f"short_track_{distance}_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


if __name__ == "__main__":
    main()
