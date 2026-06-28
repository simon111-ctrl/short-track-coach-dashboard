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
        "subtitle": "500米 / 1000米 / 1500米赛后分析、模型解释和训练建议",
        "language": "界面语言",
        "distance": "距离",
        "mode": "分析方式",
        "single": "单场分析",
        "batch": "批量分析",
        "manual": "单场输入",
        "sample": "使用示例数据",
        "athlete": "运动员 / 样本名",
        "round": "轮次",
        "qual": "晋级标记",
        "total": "官方总成绩",
        "run": "开始分析",
        "model_info": "模型信息",
        "version": "模型版本",
        "rows": "训练样本",
        "scope": "适用距离",
        "note": "本工具用于赛后参考，不替代录像回看和教练判断。",
        "result": "比赛结论",
        "grade": "成绩等级",
        "adv": "晋级参考",
        "final": "决赛参考",
        "round_pred": "最高轮次",
        "style": "战术风格",
        "rhythm": "节奏类型",
        "key": "转折圈",
        "risk": "风险状态",
        "risk_help": "风险正常表示这场的圈速、位置变化和节奏波动大体在模型见过的常见范围内；高风险表示波动明显偏大，建议重点回看碰撞、被卡位、掉速、换道和原始数据是否录错。",
        "explain": "模型解释",
        "global": "模型最看重的指标",
        "local": "本场关键指标值",
        "advice": "建议区",
        "batch_help": "批量上传说明",
        "template": "下载模板",
        "download": "下载结果",
        "upload": "上传 CSV / Excel",
        "missing": "缺少标准列：",
        "input_hint": "系统会优先根据圈速和位置分析比赛；运动员名、轮次、晋级标记和官方总成绩只是辅助信息。",
        "optional_info": "可选信息",
        "summary": "核心结论",
        "pace_chart": "圈速与位置走势",
        "display_columns": "结果展示列",
        "guide": "列名对照",
        "analysis_hint": "先填每圈成绩和位置即可分析；其他信息只用于标记比赛背景。",
        "report": "下载单场报告",
        "adv_label": "晋级判断",
        "final_label": "决赛判断",
        "entry_point": "比赛解读",
        "risk_point": "风险说明",
        "next_point": "下一步怎么改",
        "explain_task": "解释任务",
        "importance_axis": "重要度",
        "no_explain": "没有找到该任务的解释数据。",
        "race_value": "本场指标值",
        "lap_time": "第{lap}圈成绩",
        "lap_position": "第{lap}圈位置",
        "chart_lap": "圈数",
        "chart_seconds": "秒",
        "chart_position": "位置",
        "chart_lap_time": "圈速",
        "chart_position_trace": "位置",
        "column_name": "标准列名",
        "column_meaning": "含义",
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
        "risk_help": "Normal means the lap-time, position, and rhythm changes are close to patterns the model has seen before. High risk means the race looks more unstable than usual, so review contact, fading, lane changes, and data quality.",
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
        "explain_task": "Explanation task",
        "importance_axis": "Importance",
        "no_explain": "No explanation data found.",
        "race_value": "Race value",
        "lap_time": "Lap {lap} time",
        "lap_position": "Lap {lap} position",
        "chart_lap": "Lap",
        "chart_seconds": "Seconds",
        "chart_position": "Position",
        "chart_lap_time": "Lap time",
        "chart_position_trace": "Position",
        "column_name": "Column",
        "column_meaning": "Meaning",
    },
}

TASK_LABELS = {
    "zh": {
        "grade": "成绩等级",
        "advancement": "晋级判断",
        "final_entry": "决赛判断",
        "max_round": "最高轮次",
        "tactical_style": "战术风格",
        "key_lap": "转折圈",
    },
    "en": {
        "grade": "Performance grade",
        "advancement": "Advancement call",
        "final_entry": "Final call",
        "max_round": "Highest round",
        "tactical_style": "Tactical style",
        "key_lap": "Turning lap",
    },
}

DISPLAY_COL_LABELS = {
    "zh": {
        "athlete_name": "运动员 / 样本名",
        "grade": "成绩等级",
        "advancement_reference": "晋级判断",
        "final_entry_reference": "决赛判断",
        "max_round": "最高轮次",
        "tactical_style": "战术风格",
        "rhythm_type": "节奏类型",
        "key_lap": "转折圈",
        "risk_label": "风险状态",
    },
    "en": {
        "athlete_name": "Athlete / sample",
        "grade": "Performance grade",
        "advancement_reference": "Advancement call",
        "final_entry_reference": "Final call",
        "max_round": "Highest round",
        "tactical_style": "Tactical style",
        "rhythm_type": "Rhythm type",
        "key_lap": "Turning lap",
        "risk_label": "Risk check",
    },
}


@st.cache_resource
def get_service() -> PredictionService:
    return PredictionService(PACKAGE_DIR)


def t(key: str) -> str:
    return TEXT[st.session_state.lang][key]


def lang() -> str:
    return st.session_state.lang


def current_task_labels() -> dict[str, str]:
    return TASK_LABELS[lang()]


def localize_display_columns(frame: pd.DataFrame) -> pd.DataFrame:
    labels = DISPLAY_COL_LABELS[lang()]
    return frame.rename(columns={col: labels[col] for col in frame.columns if col in labels})


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
            f"- 风险状态：{row['risk_label']}（{row['risk_score']:.3f}）",
            "",
            "## 建议区",
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
    fig.add_trace(go.Scatter(x=laps, y=times, mode="lines+markers", name=t("chart_lap_time"), line={"color": "#1f5f8b", "width": 3}))
    fig.add_trace(
        go.Scatter(
            x=laps,
            y=pos,
            mode="lines+markers",
            name=t("chart_position_trace"),
            yaxis="y2",
            line={"color": "#b24a4a", "width": 3, "dash": "dot"},
        )
    )
    fig.update_layout(
        height=320,
        margin={"l": 30, "r": 35, "t": 20, "b": 30},
        xaxis={"title": t("chart_lap"), "dtick": 1},
        yaxis={"title": t("chart_seconds")},
        yaxis2={"title": t("chart_position"), "overlaying": "y", "side": "right", "autorange": "reversed", "dtick": 1},
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
        if "正常" in risk:
            risk_text = "风险状态是“风险正常”。这不是说比赛没有问题，而是说圈速、位置和节奏波动没有明显偏离常见样本。"
        else:
            risk_text = "风险状态是“高风险”。模型看到的波动偏大，建议把碰撞、被卡位、突然掉速、换道选择和数据记录都重新核一遍。"

        if adv_prob >= 0.7 and final_prob >= 0.5:
            decision_text = "这场可以保留当前大框架，训练上重点抠细节。"
        elif adv_prob >= 0.45:
            decision_text = "这场有机会，但还不稳，转折圈前后的控位和提速要优先处理。"
        else:
            decision_text = "这场晋级压力偏大，先别急着加激进动作，先把起跑、卡位和中后段执行做扎实。"

        cards = [
            {
                "title": "综合判断",
                "body": f"这场综合看是“{grade}”，晋级判断为“{status_adv(adv_prob)}”，决赛判断为“{status_final(final_prob)}”。{decision_text}",
            },
            {
                "title": t("entry_point"),
                "body": f"模型给出的风格是“{style}”，节奏是“{rhythm}”。转折圈在“{key_lap}”，这圈最值得回看：很多时候，后面能不能提速、能不能超、会不会被压住，就从这里开始变清楚。",
            },
            {
                "title": t("risk_point"),
                "body": risk_text,
            },
            {
                "title": t("next_point"),
                "body": "训练建议先按一个点收：起速型控制前两圈消耗，后程型把发力留到转折圈后，稳定型减少无谓变速。如果风险偏高，先把线路干净和节奏稳定做好。",
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
        )
        round_name = head[1].text_input(
            t("round"),
            value=str(defaults.get("round", "")) if use_sample else "",
        )
        qual = head[2].text_input(
            t("qual"),
            value=str(defaults.get("qual_code", "")) if use_sample else "",
        )
        total_text = head[3].text_input(
            t("total"),
            value=str(defaults.get("official_total_time", "")) if use_sample else "",
        )

        st.markdown("**圈速 / 位置**")
        for start in range(1, laps + 1, 3):
            cols = st.columns(min(3, laps - start + 1))
            for offset, col in enumerate(cols):
                lap = start + offset
                with col:
                    st.markdown(f"**第{lap}圈**" if lang() == "zh" else f"**Lap {lap}**")
                    lap_time = st.number_input(
                        t("lap_time").format(lap=lap),
                        min_value=0.0,
                        max_value=60.0,
                        value=float(defaults.get(f"lap{lap}_time", 0) or 0) if use_sample else 0.0,
                        step=0.001,
                        key=f"time_{distance}_{lap}",
                    )
                    lap_pos = st.number_input(
                        t("lap_position").format(lap=lap),
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
    c8.metric(t("risk"), row["risk_label"], f"{row['risk_score']:.3f}", help=t("risk_help"))

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
        task_labels = current_task_labels()
        task_options = ["grade", "advancement", "final_entry", "max_round", "tactical_style", "key_lap"]
        task_label = st.selectbox(
            t("explain_task"),
            [task_labels[item] for item in task_options],
            index=0,
            key=f"explain_task_{distance}",
        )
        task = next(item for item in task_options if task_labels[item] == task_label)
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
                bar.update_layout(height=330, margin={"l": 20, "r": 20, "t": 10, "b": 20}, xaxis_title=t("importance_axis"))
                st.plotly_chart(bar, use_container_width=True)
            else:
                st.info(t("no_explain"))
        with g2:
            st.markdown(f"#### {t('local')}")
            for _, item in local_df.head(5).iterrows():
                st.markdown(
                    f"""
                    <div style="border:1px solid #d7dee8;border-radius:8px;padding:0.7rem 0.85rem;margin-bottom:0.55rem;background:#fff;">
                      <div style="font-size:0.9rem;font-weight:700;margin-bottom:0.15rem">{item['feature_label']}</div>
                      <div style="font-size:0.88rem;line-height:1.45">{t('race_value')}：{item['value']:.3f}</div>
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
        display = output[[c for c in show_cols if c in output.columns]]
        st.dataframe(localize_display_columns(display), hide_index=True, use_container_width=True)
        st.caption("这些是给教练看的结果列，不是模型内部字段。")

    st.download_button(t("report"), make_report(distance, row, [x["body"] for x in advice_cards(row, distance)]), file_name=f"short_track_{distance}_report.md", mime="text/markdown")


def batch_upload(distance: str) -> None:
    st.subheader(t("batch_help"))
    st.caption(t("input_hint"))
    cols = get_service().required_columns_label_table(distance, lang())
    cols = cols.rename(columns={"column": t("column_name"), "meaning": t("column_meaning")})
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
    st.dataframe(localize_display_columns(output[display_cols]), hide_index=True, use_container_width=True)
    buffer = BytesIO()
    output.to_excel(buffer, index=False)
    st.download_button(
        t("download"),
        buffer.getvalue(),
        file_name=f"short_track_{distance}_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def main() -> None:
    st.set_page_config(page_title="短道速滑教练工作台", page_icon="速", layout="wide")
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
        div[data-testid="stMetricValue"] {font-size: 1.25rem; line-height: 1.2;}
        div[data-testid="stMetricDelta"] {font-size: 0.78rem;}
        section[data-testid="stSidebar"] {border-right: 1px solid #e6ebf0;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "lang" not in st.session_state:
        st.session_state.lang = "zh"
    if "single_output" not in st.session_state:
        st.session_state.single_output = None
    if "single_output_distance" not in st.session_state:
        st.session_state.single_output_distance = None

    with st.sidebar:
        choice = st.selectbox(t("language"), ["中文", "英文"], index=0 if st.session_state.lang == "zh" else 1)
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
        if not raw.empty:
            st.session_state.single_output = get_service().predict(distance, raw, lang())
            st.session_state.single_output_distance = distance
        if st.session_state.single_output is None or st.session_state.single_output_distance != distance:
            st.stop()
        output_view(st.session_state.single_output, distance)
    else:
        batch_upload(distance)


if __name__ == "__main__":
    main()
