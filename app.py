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
        "key": "转折圈",
        "risk": "异常风险",
        "explain": "模型解释",
        "global": "最该看的 5 个点",
        "local": "这场对应值",
        "advice": "教练怎么打",
        "batch_help": "批量上传说明",
        "template": "下载模板",
        "download": "下载结果",
        "upload": "上传 CSV / Excel",
        "missing": "缺少标准列：",
        "input_hint": "下面这些信息都可以留空，系统会尽量只用你填的圈速和位置来分析。",
        "optional_info": "可选信息",
        "summary": "核心结论",
        "pace_chart": "圈速与位置走势",
        "display_columns": "结果展示列",
        "guide": "列名对照",
        "fill_hint": "都可留空",
        "analysis_hint": "先填圈速和位置就能跑，其他信息只是帮助系统识别这场比赛。",
        "report": "下载单场报告",
        "adv_label": "晋级判断",
        "final_label": "决赛判断",
        "entry_point": "比赛怎么打",
        "risk_point": "风险提醒",
        "next_point": "下一步怎么改",
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
        "key": "Turning lap",
        "risk": "Risk check",
        "explain": "Model explanation",
        "global": "Top 5 things to watch",
        "local": "What this race shows",
        "advice": "How to coach it",
        "batch_help": "Batch upload guide",
        "template": "Download template",
        "download": "Download results",
        "upload": "Upload CSV / Excel",
        "missing": "Missing standard columns:",
        "input_hint": "You can leave the metadata blank. The app can still analyze the lap times and positions you provide.",
        "optional_info": "Optional info",
        "summary": "Key summary",
        "pace_chart": "Lap time and position trend",
        "display_columns": "Display columns",
        "guide": "Column guide",
        "fill_hint": "Optional",
        "analysis_hint": "Fill lap times and positions first. The rest only helps identify the race context.",
        "report": "Download single-race report",
        "adv_label": "Advance call",
        "final_label": "Final call",
        "entry_point": "Race plan",
        "risk_point": "Risk note",
        "next_point": "What to change next",
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
            f"- 晋级判断：{row['advancement_reference']}（{row['advancement_probability']:.1%}）",
            f"- 决赛判断：{row['final_entry_reference']}（{row['final_entry_probability']:.1%}）",
            f"- 最高轮次：{row['max_round']}",
            f"- 战术风格：{row['tactical_style']}",
            f"- 节奏类型：{row['rhythm_type']}",
            f"- 转折圈：{row['key_lap']}",
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
            f"- Advancement call: {row['advancement_reference']} ({row['advancement_probability']:.1%})",
            f"- Final call: {row['final_entry_reference']} ({row['final_entry_probability']:.1%})",
            f"- Highest round: {row['max_round']}",
            f"- Tactical style: {row['tactical_style']}",
            f"- Rhythm type: {row['rhythm_type']}",
            f"- Turning lap: {row['key_lap']}",
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


def status_adv(probability: float) -> str:
    if lang() == "zh":
        if probability >= 0.7:
            return "晋级把握高"
        if probability >= 0.45:
            return "有晋级机会"
        return "晋级压力大"
    if probability >= 0.7:
        return "Likely to advance"
    if probability >= 0.45:
        return "Possible to advance"
    return "Needs a push"


def status_final(probability: float) -> str:
    if lang() == "zh":
        if probability >= 0.7:
            return "很有机会进决赛"
        if probability >= 0.45:
            return "有机会进决赛"
        return "暂时难进决赛"
    if probability >= 0.7:
        return "Very likely to make the final"
    if probability >= 0.45:
        return "Possible to make the final"
    return "Still a tough final path"


def advice_cards(row: pd.Series, distance: str) -> list[dict[str, str]]:
    adv_prob = float(row.get("advancement_probability", 0))
    final_prob = float(row.get("final_entry_probability", 0))
    style = str(row.get("tactical_style", ""))
    rhythm = str(row.get("rhythm_type", ""))
    key_lap = str(row.get("key_lap", ""))
    risk = str(row.get("risk_label", ""))
    grade = str(row.get("grade", ""))

    if lang() == "zh":
        cards = [
            {
                "title": "先看结论",
                "body": f"这场是“{grade}”。晋级判断是“{status_adv(adv_prob)}”，决赛判断是“{status_final(final_prob)}”。",
            },
            {
                "title": t("entry_point"),
                "body": f"风格是“{style}”，节奏是“{rhythm}”。转折圈是“{key_lap}”，意思是这圈最值得回看，因为它最容易决定后面的提速、掉速、超越或被压住。",
            },
            {
                "title": t("risk_point"),
                "body": "风险正常" if "正常" in risk else "这场风险偏高，建议重点回看碰撞、掉速和换道。",
            },
            {
                "title": t("next_point"),
                "body": "起速型就别把前两圈烧太满；后程型就把力量留到转折圈后；稳定型就用固定圈速推进。",
            },
        ]
    else:
        cards = [
            {
                "title": "Start with the result",
                "body": f"This race reads as “{grade}”. The advance call is “{status_adv(adv_prob)}”, and the final call is “{status_final(final_prob)}”.",
            },
            {
                "title": t("entry_point"),
                "body": f"The style is “{style}”, the rhythm is “{rhythm}”. The turning lap is “{key_lap}”, which is the lap worth rewatching because it often decides whether the race speeds up, fades, or gets boxed in.",
            },
            {
                "title": t("risk_point"),
                "body": "Risk looks normal." if "Normal" in risk else "Risk is elevated, so review contacts, fades, and lane changes.",
            },
            {
                "title": t("next_point"),
                "body": "Fast-start profiles should avoid overcooking the first two laps; late-chase profiles should save power for after the key lap.",
            },
        ]
    return cards


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
    st.markdown(f"### {t('manual')}")
    st.caption(t("analysis_hint"))
    rows = []
    with st.container(border=True):
        st.markdown(f"**{t('optional_info')}**")
        head = st.columns([1.15, 0.85, 0.85, 1.05])
        athlete = head[0].text_input(
            t("athlete"),
            value=str(defaults.get("athlete_name", "")) if use_sample else "",
            placeholder=t("fill_hint"),
            help="可留空",
        )
        round_name = head[1].text_input(
            t("round"),
            value=str(defaults.get("round", "")) if use_sample else "",
            placeholder=t("fill_hint"),
            help="可留空",
        )
        qual = head[2].text_input(
            t("qual"),
            value=str(defaults.get("qual_code", "")) if use_sample else "",
            placeholder=t("fill_hint"),
            help="可留空",
        )
        total_text = head[3].text_input(
            t("total"),
            value=str(defaults.get("official_total_time", "")) if use_sample else "",
            placeholder=t("fill_hint"),
            help="可留空",
        )

        st.markdown("**圈速 / 位置**")
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
                        value=float(defaults.get(f"lap{lap}_time", 0) or 0) if use_sample else 0.0,
                        step=0.001,
                        key=f"time_{distance}_{lap}",
                    )
                    lap_pos = st.number_input(
                        f"L{lap} pos",
                        label_visibility="collapsed",
                        min_value=1,
                        max_value=12,
                        value=int(float(defaults.get(f"lap{lap}_position", 1) or 1)) if use_sample else 1,
                        step=1,
                        key=f"pos_{distance}_{lap}",
                    )
                    rows.append((lap, lap_time, lap_pos))

    submitted = st.button(t("run"), type="primary", use_container_width=True)
    if not submitted:
        return pd.DataFrame()

    total_value = None
    if total_text.strip():
        try:
            total_value = float(total_text)
        except ValueError:
            total_value = None

    row = {
        "athlete_name": athlete,
        "distance": distance,
        "round": round_name,
        "qual_code": qual,
        "official_total_time": total_value,
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
    c2.metric(t("adv_label"), status_adv(float(row["advancement_probability"])), f"{row['advancement_probability']:.0%}")
    c3.metric(t("final_label"), status_final(float(row["final_entry_probability"])), f"{row['final_entry_probability']:.0%}")
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
        st.subheader(t("advice"))
        for item in advice_cards(row, distance):
            st.markdown(
                f"""
                <div style="border:1px solid #d7dee8;border-radius:8px;padding:0.75rem 0.85rem;margin-bottom:0.6rem;background:#fff;">
                  <div style="font-size:0.8rem;opacity:0.68;margin-bottom:0.2rem">{item['title']}</div>
                  <div style="font-size:0.96rem;line-height:1.55">{item['body']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    tab1, tab2 = st.tabs([t("explain"), t("display_columns")])
    with tab1:
        task = st.selectbox("Task", ["grade", "advancement", "final_entry", "max_round", "tactical_style", "key_lap"], index=0)
        global_df = get_service().global_explanation(distance, task, lang())
        local_df = get_service().local_explanation(distance, task, output.head(1), lang())
        g1, g2 = st.columns([1.05, 0.95], vertical_alignment="top")
        with g1:
            st.markdown(f"#### {t('global')}")
            if not global_df.empty:
                bar = go.Figure(go.Bar(
                    x=global_df["importance_mean"],
                    y=global_df["feature_label"],
                    orientation="h",
                    marker={"color": ["#1f5f8b"] * len(global_df)},
                ))
                bar.update_layout(height=330, margin={"l": 20, "r": 20, "t": 10, "b": 20}, xaxis_title="Importance")
                st.plotly_chart(bar, use_container_width=True)
            else:
                st.info("No explanation data found.")
        with g2:
            st.markdown(f"#### {t('local')}")
            for _, item in local_df.head(5).iterrows():
                st.markdown(
                    f"""
                    <div style="border:1px solid #d7dee8;border-radius:8px;padding:0.7rem 0.85rem;margin-bottom:0.55rem;background:#fff;">
                      <div style="font-size:0.9rem;font-weight:700;margin-bottom:0.15rem">{item['feature_label']}</div>
                      <div style="font-size:0.88rem;line-height:1.45">本场值：{item['value']:.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    with tab2:
        st.markdown(f"#### {t('display_columns')}")
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
        st.caption("这些是给教练看的结果列，不是模型内部字段。")

    st.download_button(t("report"), make_report(distance, row, [x["body"] for x in advice_cards(row, distance)]), file_name=f"short_track_{distance}_report.md", mime="text/markdown")


def batch_upload(distance: str) -> None:
    st.subheader(t("batch_help"))
    st.caption(t("input_hint"))
    st.caption("运动员名、轮次、晋级代码、官方总成绩都可以留空。")
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
