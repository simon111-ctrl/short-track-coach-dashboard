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
        "subtitle": "500m / 1000m / 1500m 赛后分析、解释和建议",
        "language": "语言",
        "distance": "距离",
        "mode": "分析方式",
        "single": "单场分析",
        "batch": "批量分析",
        "manual": "单场输入",
        "sample": "使用示例数据",
        "athlete": "运动员 / 样本名",
        "round": "轮次",
        "qual": "晋级代码",
        "total": "官方总成绩（可留空）",
        "run": "开始分析",
        "model_info": "模型信息",
        "version": "模型版本",
        "rows": "训练样本",
        "scope": "适用距离",
        "note": "本工具用于赛后参考，不替代录像回看和教练判断。",
        "result": "比赛结论",
        "grade": "成绩等级",
        "adv": "晋级参考",
        "final": "决赛入围",
        "round_pred": "最高轮次",
        "style": "战术风格",
        "rhythm": "节奏类型",
        "key": "关键圈",
        "risk": "异常风险",
        "explain": "模型解释",
        "global": "全局重要因素",
        "local": "本场样本重点",
        "advice": "教练建议",
        "batch_help": "批量上传说明",
        "template": "下载模板",
        "download": "下载结果",
        "upload": "上传 CSV / Excel",
        "missing": "缺少标准列：",
        "input_hint": "标准列名以 `athlete_name`、`round`、`qual_code`、`official_total_time`、`lap1_time` 开头。",
        "summary": "核心结论",
        "pace_chart": "圈速与位置走势",
        "display_columns": "结果展示列",
        "guide": "列名对照",
        "coach_block": "建议区",
        "report": "下载单场报告",
    },
    "en": {
        "title": "Short-Track Multi-Distance Coach Workspace",
        "subtitle": "Post-race analysis, explanation, and recommendations for 500m / 1000m / 1500m",
        "language": "Language",
        "distance": "Distance",
        "mode": "Analysis mode",
        "single": "Single race",
        "batch": "Batch analysis",
        "manual": "Single-race input",
        "sample": "Use sample data",
        "athlete": "Athlete / sample name",
        "round": "Round",
        "qual": "Qualification code",
        "total": "Official total time (optional)",
        "run": "Analyze",
        "model_info": "Model info",
        "version": "Model version",
        "rows": "Training rows",
        "scope": "Applicable distance",
        "note": "This tool is for post-race reference and does not replace video review or coach judgement.",
        "result": "Race conclusion",
        "grade": "Performance grade",
        "adv": "Advancement reference",
        "final": "Final-entry path",
        "round_pred": "Highest round",
        "style": "Tactical style",
        "rhythm": "Rhythm type",
        "key": "Key lap",
        "risk": "Risk check",
        "explain": "Model explanation",
        "global": "Global drivers",
        "local": "Current sample focus",
        "advice": "Coach advice",
        "batch_help": "Batch upload guide",
        "template": "Download template",
        "download": "Download results",
        "upload": "Upload CSV / Excel",
        "missing": "Missing standard columns:",
        "input_hint": "Standard columns start with `athlete_name`, `round`, `qual_code`, `official_total_time`, `lap1_time`.",
        "summary": "Key summary",
        "pace_chart": "Lap time and position trend",
        "display_columns": "Display columns",
        "guide": "Column guide",
        "coach_block": "Advice",
        "report": "Download single-race report",
    },
}


@st.cache_resource
def get_service() -> PredictionService:
    return PredictionService(PACKAGE_DIR)


def t(key: str) -> str:
    return TEXT[st.session_state.lang][key]


def lang() -> str:
    return st.session_state.lang


def sample_frame(distance: str) -> pd.DataFrame:
    return pd.read_csv(PACKAGE_DIR / "examples" / f"example_input_{distance}_grade.csv")


def template_frame(distance: str) -> pd.DataFrame:
    frame = sample_frame(distance).head(1).copy()
    cols = get_service().required_columns(distance)
    return frame[cols]


def make_report(distance: str, row: pd.Series, notes: list[str]) -> str:
    if lang() == "zh":
        lines = [
            f"# 短道速滑教练报告 - {distance}",
            "",
            f"- 运动员 / 样本：{row.get('athlete_name', '')}",
            f"- 成绩等级：{row['grade']}（{row['grade_probability']:.1%}）",
            f"- 晋级参考：{row['advancement_reference']}（{row['advancement_probability']:.1%}）",
            f"- 决赛入围：{row['final_entry_reference']}（{row['final_entry_probability']:.1%}）",
            f"- 最高轮次：{row['max_round']}",
            f"- 战术风格：{row['tactical_style']}",
            f"- 节奏类型：{row['rhythm_type']}",
            f"- 关键圈：{row['key_lap']}",
            f"- 异常风险：{row['risk_label']}（{row['risk_score']:.3f}）",
            "",
            "## 教练建议",
        ]
    else:
        lines = [
            f"# Short-Track Coach Report - {distance}",
            "",
            f"- Athlete / sample: {row.get('athlete_name', '')}",
            f"- Performance grade: {row['grade']} ({row['grade_probability']:.1%})",
            f"- Advancement reference: {row['advancement_reference']} ({row['advancement_probability']:.1%})",
            f"- Final-entry path: {row['final_entry_reference']} ({row['final_entry_probability']:.1%})",
            f"- Highest round: {row['max_round']}",
            f"- Tactical style: {row['tactical_style']}",
            f"- Rhythm type: {row['rhythm_type']}",
            f"- Key lap: {row['key_lap']}",
            f"- Risk check: {row['risk_label']} ({row['risk_score']:.3f})",
            "",
            "## Coach Advice",
        ]
    return "\n".join(lines + [f"- {x}" for x in notes])


def metric_block(label: str, value: str, delta: str = "", help_text: str = "") -> None:
    st.metric(label=label, value=value, delta=delta, help=help_text)


def lap_chart(row: pd.Series, distance: str) -> go.Figure:
    laps = list(range(1, DISTANCE_LAPS[distance] + 1))
    times = [float(row[f"lap{i}_time"]) for i in laps]
    pos = [float(row[f"lap{i}_position"]) for i in laps]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=laps, y=times, mode="lines+markers", name="Lap time", line={"color": "#1f5f8b", "width": 3}))
    fig.add_trace(
        go.Scatter(
            x=laps,
            y=pos,
            mode="lines+markers",
            name="Position",
            yaxis="y2",
            line={"color": "#b24a4a", "width": 3, "dash": "dot"},
        )
    )
    fig.update_layout(
        height=320,
        margin={"l": 30, "r": 35, "t": 20, "b": 30},
        xaxis={"title": "Lap", "dtick": 1},
        yaxis={"title": "Seconds"},
        yaxis2={"title": "Position", "overlaying": "y", "side": "right", "autorange": "reversed", "dtick": 1},
        legend={"orientation": "h", "y": 1.05},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fbfcfe",
    )
    return fig


def advice_items(row: pd.Series, distance: str) -> list[str]:
    notes = get_service().coach_notes(distance, row, lang())
    if lang() == "zh":
        notes.insert(0, "先看结论，再看过程。")
    else:
        notes.insert(0, "Start with the conclusion, then review the process.")
    return notes


def render_header(distance: str) -> None:
    model_meta = get_service().manifest["models"][f"{distance}_advancement"]
    col1, col2 = st.columns([1.3, 0.7], vertical_alignment="center")
    with col1:
        st.title(t("title"))
        st.caption(t("subtitle"))
    with col2:
        st.markdown(
            f"""
            <div style="text-align:right;padding-top:0.3rem;">
              <div style="font-size:0.8rem;opacity:0.72">{t("version")}</div>
              <div style="font-size:1rem;font-weight:700;line-height:1.35">{get_service().manifest["created_at"]}</div>
              <div style="font-size:0.8rem;opacity:0.72;margin-top:0.25rem">{t("rows")}: {model_meta["n_training_rows"]:,}</div>
              <div style="font-size:0.8rem;opacity:0.72">{t("scope")}: {distance}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def manual_input(distance: str, use_sample: bool) -> pd.DataFrame:
    laps = DISTANCE_LAPS[distance]
    defaults = sample_frame(distance).iloc[0].to_dict() if use_sample else {}

    with st.form("manual_form", clear_on_submit=False):
        st.markdown(f"### {t('manual')}")
        head = st.columns([1.2, 0.8, 0.8, 1.0])
        athlete = head[0].text_input(t("athlete"), value=str(defaults.get("athlete_name", "Athlete A")))
        round_name = head[1].text_input(t("round"), value=str(defaults.get("round", "Heats")))
        qual = head[2].text_input(t("qual"), value=str(defaults.get("qual_code", "")))
        total = head[3].number_input(t("total"), min_value=0.0, max_value=300.0, value=float(defaults.get("official_total_time", 0) or 0), step=0.001)

        st.caption("圈速 / 位置输入")
        rows = []
        for start in range(1, laps + 1, 3):
            cols = st.columns(min(3, laps - start + 1))
            for offset, col in enumerate(cols):
                lap = start + offset
                with col:
                    st.markdown(f"**L{lap}**")
                    lap_time = st.number_input(
                        f"L{lap} time",
                        label_visibility="collapsed",
                        min_value=0.0,
                        max_value=60.0,
                        value=float(defaults.get(f"lap{lap}_time", 0) or 0),
                        step=0.001,
                        key=f"time_{distance}_{lap}",
                    )
                    lap_pos = st.number_input(
                        f"L{lap} pos",
                        label_visibility="collapsed",
                        min_value=1,
                        max_value=12,
                        value=int(float(defaults.get(f"lap{lap}_position", 1) or 1)),
                        step=1,
                        key=f"pos_{distance}_{lap}",
                    )
                    rows.append((lap, lap_time, lap_pos))

        submitted = st.form_submit_button(t("run"), type="primary", use_container_width=True)

    if not submitted:
        return pd.DataFrame()

    row = {
        "athlete_name": athlete,
        "distance": distance,
        "round": round_name,
        "qual_code": qual,
        "official_total_time": total if total > 0 else None,
    }
    for lap, lap_time, lap_pos in rows:
        row[f"lap{lap}_time"] = lap_time
        row[f"lap{lap}_position"] = lap_pos
    row["reconstructed_total_time"] = sum(v for _, v, _ in rows)
    return pd.DataFrame([row])


def output_view(output: pd.DataFrame, distance: str) -> None:
    row = output.iloc[0]
    st.subheader(t("result"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("grade"), row["grade"], f"{row['grade_probability']:.0%}")
    c2.metric(t("adv"), row["advancement_reference"], f"{row['advancement_probability']:.0%}")
    c3.metric(t("final"), row["final_entry_reference"], f"{row['final_entry_probability']:.0%}")
    c4.metric(t("round_pred"), row["max_round"], f"{row['max_round_probability']:.0%}")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric(t("style"), row["tactical_style"])
    c6.metric(t("rhythm"), row["rhythm_type"])
    c7.metric(t("key"), row["key_lap"])
    c8.metric(t("risk"), row["risk_label"], f"{row['risk_score']:.3f}")

    left, right = st.columns([1.15, 0.85], vertical_alignment="top")
    with left:
        st.subheader(t("summary"))
        st.plotly_chart(lap_chart(row, distance), use_container_width=True)
    with right:
        st.subheader(t("coach_block"))
        for item in advice_items(row, distance):
            st.markdown(
                f"""
                <div style="border:1px solid #d7dee8;border-radius:8px;padding:0.75rem 0.85rem;margin-bottom:0.6rem;background:#fff;">
                  <div style="font-size:0.96rem;line-height:1.5">{item}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    tab1, tab2, tab3 = st.tabs([t("explain"), t("advice"), t("display_columns")])
    with tab1:
        task = st.selectbox("Task", ["grade", "advancement", "final_entry", "max_round", "tactical_style", "key_lap"], index=0)
        st.markdown(f"#### {t('global')}")
        st.dataframe(get_service().global_explanation(distance, task, lang()), hide_index=True, use_container_width=True)
        st.markdown(f"#### {t('local')}")
        st.dataframe(get_service().local_explanation(distance, task, output.head(1), lang()), hide_index=True, use_container_width=True)
    with tab2:
        notes = advice_items(row, distance)
        for idx, item in enumerate(notes, 1):
            st.markdown(
                f"""
                <div style="border:1px solid #d7dee8;border-radius:8px;padding:0.8rem 0.9rem;margin-bottom:0.7rem;background:#fff;">
                  <div style="font-size:0.9rem;opacity:0.65;margin-bottom:0.2rem">{idx}</div>
                  <div style="font-size:1rem;line-height:1.55">{item}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with tab3:
        show_cols = [
            "athlete_name",
            "grade",
            "advancement_reference",
            "final_entry_reference",
            "max_round",
            "tactical_style",
            "rhythm_type",
            "key_lap",
            "risk_label",
        ]
        st.dataframe(output[[c for c in show_cols if c in output.columns]], hide_index=True, use_container_width=True)

    st.download_button(t("report"), make_report(distance, row, notes), file_name=f"short_track_{distance}_report.md", mime="text/markdown")


def batch_upload(distance: str) -> None:
    st.subheader(t("batch_help"))
    st.caption(t("input_hint"))
    cols = get_service().required_columns_label_table(distance, lang())
    st.dataframe(cols, hide_index=True, use_container_width=True)
    st.download_button(
        t("template"),
        template_frame(distance).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"template_{distance}.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader(t("upload"), type=["csv", "xlsx", "xls"])
    if not uploaded:
        return

    raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
    required = get_service().required_columns(distance)
    missing = [c for c in required if c not in raw.columns]
    if missing:
        st.error(f"{t('missing')} {', '.join(missing)}")
        return

    output = get_service().predict(distance, raw, lang())
    display_cols = [c for c in ["athlete_name", "grade", "advancement_reference", "final_entry_reference", "max_round", "tactical_style", "rhythm_type", "key_lap", "risk_label"] if c in output.columns]
    st.markdown(f"#### {t('display_columns')}")
    st.dataframe(output[display_cols], hide_index=True, use_container_width=True)
    buffer = BytesIO()
    output.to_excel(buffer, index=False)
    st.download_button(
        t("download"),
        buffer.getvalue(),
        file_name=f"short_track_{distance}_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def main() -> None:
    st.set_page_config(page_title="Short-Track Coach Workspace", page_icon="ST", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1380px;}
        div[data-testid="stMetric"] {
            background: #f8fbfe;
            border: 1px solid #d8e1ea;
            border-radius: 8px;
            padding: 0.85rem 0.9rem;
        }
        div[data-testid="stMetricLabel"] {font-size: 0.78rem; line-height: 1.2;}
        div[data-testid="stMetricValue"] {font-size: 1.25rem; line-height: 1.2;}
        div[data-testid="stMetricDelta"] {font-size: 0.78rem;}
        section[data-testid="stSidebar"] {border-right: 1px solid #e6ebf0;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "lang" not in st.session_state:
        st.session_state.lang = "zh"

    with st.sidebar:
        choice = st.selectbox("Language / 语言", ["中文", "English"], index=0 if st.session_state.lang == "zh" else 1)
        st.session_state.lang = "zh" if choice == "中文" else "en"
        distance = st.radio(t("distance"), ["500m", "1000m", "1500m"], horizontal=True)
        mode = st.radio(t("mode"), [t("single"), t("batch")], horizontal=False)
        use_sample = st.checkbox(t("sample"), value=True)
        meta = get_service().manifest["models"][f"{distance}_advancement"]
        st.markdown(f"**{t('model_info')}**")
        st.caption(f"{t('version')}: {get_service().manifest['created_at']}")
        st.caption(f"{t('rows')}: {meta['n_training_rows']:,}")
        st.caption(f"{t('scope')}: {distance}")
        st.markdown("---")
        st.download_button(
            t("template"),
            template_frame(distance).to_csv(index=False).encode("utf-8-sig"),
            file_name=f"template_{distance}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    render_header(distance)
    st.info(t("note"))

    if mode == t("single"):
        raw = manual_input(distance, use_sample)
        if raw.empty:
            st.stop()
        output = get_service().predict(distance, raw, lang())
        output_view(output, distance)
    else:
        batch_upload(distance)


if __name__ == "__main__":
    main()
