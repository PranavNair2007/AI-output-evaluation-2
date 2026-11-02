"""AI Output Quality Evaluation Dashboard (Streamlit). Run: streamlit run app.py"""
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

st.set_page_config(page_title="AI Output Quality Evaluation Dashboard", page_icon="✅", layout="wide")
core.init_db()

# Custom CSS for modern grid elevation shadows and cards
st.markdown("""
<style>
/* Grid Card Containers with modern elevation shadow */
div[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.05);
    border-radius: 12px !important;
    border: 1px solid rgba(128, 128, 128, 0.18) !important;
    transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.14), 0 3px 6px rgba(0, 0, 0, 0.08);
    transform: translateY(-2px);
}

/* Data Grid (st.dataframe) modern shadow */
div[data-testid="stDataFrame"] {
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.05);
    border-radius: 10px;
    border: 1px solid rgba(128, 128, 128, 0.16);
    overflow: hidden;
}

/* Metric KPI card shadows */
div[data-testid="stMetric"] {
    background: rgba(128, 128, 128, 0.04);
    padding: 14px 18px;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
    border: 1px solid rgba(128, 128, 128, 0.15);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.1);
    transform: translateY(-1px);
}

/* Modern Sidebar Navigation Buttons */
section[data-testid="stSidebar"] div.stButton > button {
    text-align: left;
    justify-content: flex-start;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    padding: 8px 12px;
    margin-bottom: 2px;
    transition: all 0.15s ease-in-out;
}

section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
    background: transparent;
    border: 1px solid transparent;
}

section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
    background: rgba(128, 128, 128, 0.12);
    border: 1px solid rgba(128, 128, 128, 0.2);
}

section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
}
</style>
""", unsafe_allow_html=True)

PAGES = ["Overview", "New Evaluation", "Evaluation History", "Analytics",
         "Compare Responses", "Reports", "Settings"]
RANGES = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "All time": 0}
DL = [core.LABELS[k] for k in core.DIMS]
ss = st.session_state
ss.setdefault("page", "Overview")


def go_to(page, **kw):
    ss.page = page
    ss.update(kw)


def badge(status):
    c = core.STATUS_COLORS[status]
    return (f"<span style='background:{c};color:#fff;padding:2px 10px;border-radius:99px;"
            f"font-size:13px;font-weight:600'>{status}</span>")


def in_range(evals, days):
    if not days:
        return evals
    cut = datetime.now() - timedelta(days=days)
    return [e for e in evals if e["created_at"] >= cut]


def apply_grid(fig):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)"),
    )
    return fig


GRID_COLUMN_CONFIG = {
    "Overall": st.column_config.ProgressColumn(
        "Overall",
        help="Overall quality score (0-100)",
        format="%d%%",
        min_value=0,
        max_value=100,
    ),
    "Relevance": st.column_config.ProgressColumn(
        "Relevance",
        help="Relevance score (0-10)",
        format="%d",
        min_value=0,
        max_value=10,
    ),
    "Completeness": st.column_config.ProgressColumn(
        "Completeness",
        help="Completeness score (0-10)",
        format="%d",
        min_value=0,
        max_value=10,
    ),
    "Clarity": st.column_config.ProgressColumn(
        "Clarity",
        help="Clarity score (0-10)",
        format="%d",
        min_value=0,
        max_value=10,
    ),
    "Factual Support": st.column_config.ProgressColumn(
        "Factual",
        help="Factual Support score (0-10)",
        format="%d",
        min_value=0,
        max_value=10,
    ),
    "Status": st.column_config.TextColumn(
        "Status",
        help="Quality status band",
    ),
    "Date": st.column_config.DateColumn(
        "Date",
        format="YYYY-MM-DD",
    ),
}


def radar(series):
    fig = go.Figure()
    for name, e in series.items():
        v = [e["scores"][k] for k in core.DIMS] if isinstance(e, dict) else e
        fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=DL + DL[:1], fill="toself", name=name))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 10], showgrid=True, gridcolor="rgba(128,128,128,0.25)"),
            angularaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.25)")
        ),
        margin=dict(t=20, b=20, l=40, r=40),
        height=320,
        showlegend=len(series) > 1
    )
    return fig


def render_eval_grid(eval_list, key_prefix="grid", cols_count=3):
    """Renders a responsive card grid for evaluations."""
    cols = st.columns(cols_count)
    for i, e in enumerate(eval_list):
        with cols[i % cols_count]:
            with st.container(border=True):
                st.markdown(f"**{e['name']}**")
                c_badge, c_score = st.columns([1, 1])
                c_badge.markdown(badge(e["status"]), unsafe_allow_html=True)
                c_score.markdown(f"**{e['overall']}/100**")
                st.caption(f"`{e['id']}` · {e['created_at']:%d %b %Y}")
                d_grid = st.columns(2)
                for idx, dim in enumerate(core.DIMS):
                    with d_grid[idx % 2]:
                        score = e["scores"][dim]
                        st.caption(f"{core.LABELS[dim]}: **{score}/10**")
                if st.button("Inspect Details", key=f"{key_prefix}_btn_{e['id']}", use_container_width=True):
                    ss[f"{key_prefix}_selected"] = e["id"]
                    st.rerun()

    sel_id = ss.get(f"{key_prefix}_selected")
    if sel_id:
        selected_eval = next((e for e in eval_list if e["id"] == sel_id), None)
        if selected_eval:
            st.divider()
            detail(selected_eval, f"{key_prefix}_detail")


def detail(e, key=""):
    st.subheader(e["name"])
    mode = {"sample": "Sample data", "demo": "Demo-mode scoring (heuristic, not a real AI judge)",
            "ai": "AI-judged"}.get(e["mode"], e["mode"])
    st.caption(f"{e['id']} · {e['created_at']:%d %b %Y} · {mode}")
    a, b = st.columns(2)
    with a:
        st.metric("Overall Quality Score", f"{e['overall']}/100")
        st.markdown(badge(e["status"]), unsafe_allow_html=True)
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        st.markdown("**Dimension Breakdown**")
        d_cols = st.columns(2)
        for idx, k in enumerate(core.DIMS):
            with d_cols[idx % 2]:
                with st.container(border=True):
                    st.markdown(f"**{core.LABELS[k]}**")
                    st.progress(e["scores"][k] / 10, text=f"{e['scores'][k]}/10")
                    st.caption(e["feedback"]["why"][k])
    with b:
        st.plotly_chart(radar({"Scores": e}), key=f"radar{key}{e['id']}")
    c, d = st.columns(2)
    fb = e["feedback"]
    with c:
        with st.container(border=True):
            st.markdown("**Strengths**\n" + "\n".join(f"- {s}" for s in fb["strengths"]))
            st.markdown("**Weaknesses**\n" + "\n".join(f"- {s}" for s in fb["weaknesses"]))
    with d:
        with st.container(border=True):
            st.markdown("**Recommendations**\n" + "\n".join(f"- {s}" for s in fb["recs"]))
            if not e["sources"]:
                st.info("Factual Support is unverified: no reference material was provided.")
    with st.expander("Prompt and response"):
        st.markdown("**Prompt**"); st.text(e["prompt"])
        st.markdown("**Response**"); st.text(e["response"])


def to_df(evals):
    return pd.DataFrame([{"ID": e["id"], "Name": e["name"], "Date": e["created_at"].date(),
                          "Relevance": e["scores"]["relevance"], "Completeness": e["scores"]["completeness"],
                          "Clarity": e["scores"]["clarity"], "Factual Support": e["scores"]["factual"],
                          "Overall": e["overall"], "Status": e["status"]} for e in evals])


def run_eval(name, prompt, response, expected, sources, use_ai, eid=None, created=None):
    sc, fb, mode = core.evaluate(prompt, response, expected, sources, use_ai)
    e = dict(id=eid or core.next_id(), name=name, created_at=created or datetime.now(), prompt=prompt,
             response=response, expected=expected, sources=sources, scores=sc, feedback=fb, mode=mode)
    core.save_eval(e)
    return core.get_eval(e["id"])


# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("## ✅ EvalDash\n<span style='color:gray'>AI Quality</span>", unsafe_allow_html=True)

PAGE_ICONS = {
    "Overview": "📊",
    "New Evaluation": "➕",
    "Evaluation History": "📜",
    "Analytics": "📈",
    "Compare Responses": "⚖️",
    "Reports": "📑",
    "Settings": "⚙️",
}

for p in PAGES:
    icon = PAGE_ICONS.get(p, "•")
    is_active = (ss.page == p)
    if st.sidebar.button(
        f"{icon}  {p}",
        key=f"nav_btn_{p}",
        use_container_width=True,
        type="primary" if is_active else "secondary"
    ):
        if ss.page != p:
            ss.page = p
            st.rerun()

st.sidebar.divider()
st.sidebar.caption(f"Mode: **AI judge ({core.active_ai_provider()})**" if core.ai_available() else "Mode: **Demo** (heuristic scoring, not a real AI judge)")
evals = core.all_evals()
page = ss.page

# ------------------------------------------------------------------ pages
if page == "Overview":
    st.title("Overview")
    now = datetime.now()
    cur = [e for e in evals if e["created_at"] >= now - timedelta(days=30)]
    prv = [e for e in evals if now - timedelta(days=60) <= e["created_at"] < now - timedelta(days=30)]

    def kpis(l):
        n = len(l)
        return (n, round(sum(e["overall"] for e in l) / n) if n else 0,
                round(100 * sum(e["overall"] >= 90 for e in l) / n) if n else 0,
                round(100 * sum(e["overall"] < 60 for e in l) / n) if n else 0)

    c, p = kpis(cur), kpis(prv)
    delta = lambda a, b: f"{round((a - b) / b * 100)}%" if b else "n/a"
    cols = st.columns(4)
    cols[0].metric("Total Evaluations", c[0], delta(c[0], p[0]))
    cols[1].metric("Average Quality Score", c[1], delta(c[1], p[1]))
    cols[2].metric("Excellent Responses", f"{c[2]}%", delta(c[2], p[2]))
    cols[3].metric("Needs Improvement", f"{c[3]}%", delta(c[3], p[3]), delta_color="inverse")
    st.caption("Last 30 days, change versus the previous 30 days.")
    if evals:
        a, b = st.columns([2, 1])
        df = pd.DataFrame({"t": [e["created_at"] for e in evals], "s": [e["overall"] for e in evals]})
        wk = df.set_index("t").resample("W")["s"].mean().dropna().reset_index()
        a.plotly_chart(apply_grid(px.line(wk, x="t", y="s", markers=True, title="Weekly average quality (%)",
                               labels={"t": "", "s": "Score"})), key="ov_trend")
        cnt = pd.Series([e["status"] for e in evals]).value_counts().reindex([n for n, *_ in core.BANDS], fill_value=0)
        b.plotly_chart(px.pie(values=cnt.values, names=cnt.index, hole=.55, title="Status mix",
                              color=cnt.index, color_discrete_map=core.STATUS_COLORS), key="ov_donut")
        st.subheader("Recent evaluations")
        ss.setdefault("ov_view_mode", "Card Grid")
        ov_c1, ov_c2, _ = st.columns([1, 1, 3])
        if ov_c1.button("🃏 Card Grid", key="btn_ov_grid", type="primary" if ss.ov_view_mode == "Card Grid" else "secondary", use_container_width=True):
            ss.ov_view_mode = "Card Grid"
            st.rerun()
        if ov_c2.button("📋 Selector", key="btn_ov_sel", type="primary" if ss.ov_view_mode == "Single Selector" else "secondary", use_container_width=True):
            ss.ov_view_mode = "Single Selector"
            st.rerun()
        if ss.ov_view_mode == "Card Grid":
            render_eval_grid(evals[:6], key_prefix="ov_grid", cols_count=3)
        else:
            recent = evals[:10]
            pick = st.selectbox("Select an evaluation to see its results", recent,
                                format_func=lambda e: f"{e['id']} · {e['name']} · {e['overall']}/100 · {e['created_at']:%d %b %Y}")
            detail(pick, "ov")
    else:
        st.info("No evaluations yet. Start with New Evaluation.")

elif page == "New Evaluation":
    eid = ss.get("edit_id")
    old = core.get_eval(eid) if eid else None
    st.title(f"Edit {eid}" if old else "New Evaluation")
    if core.ai_available():
        st.success("API key detected: real AI judging is available.")
    else:
        st.warning("**Demo evaluation mode.** No AI API key is configured, so scores come from a deterministic "
                   "text heuristic, not a real AI judge. Factual Support stays low unless sources or an expected "
                   "answer are supplied, and is never marked verified.")
    with st.form("eval_form"):
        name = st.text_input("Evaluation Name *", old["name"] if old else "")
        prompt = st.text_area("User Prompt / Query *", old["prompt"] if old else "")
        response = st.text_area("AI-Generated Response *", old["response"] if old else "", height=180)
        expected = st.text_area("Expected Answer (optional)", old["expected"] if old else "")
        sources = st.text_area("Reference Sources (optional)", old["sources"] if old else "",
                               help="Paste source text or URLs, one per line")
        use_ai = st.checkbox("Use real AI judge", value=core.ai_available(), disabled=not core.ai_available())
        go_btn = st.form_submit_button("Evaluate Response", type="primary")
    if go_btn:
        errs = [m for ok, m in [(name.strip(), "Name is required."), (len(prompt.strip()) >= 5, "Prompt needs at least 5 characters."),
                                (len(response.strip()) >= 20, "Response needs at least 20 characters.")] if not ok]
        for m in errs:
            st.error(m)
        if not errs:
            try:
                with st.spinner("Evaluating…"):
                    e = run_eval(name.strip(), prompt.strip(), response.strip(), expected.strip(), sources.strip(),
                                 use_ai, old["id"] if old else None, old["created_at"] if old else None)
                ss.pop("edit_id", None)
                st.success(f"Saved as {e['id']}")
                detail(e, "new")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Evaluation failed: {exc}")

elif page == "Evaluation History":
    st.title("Evaluation History")
    c1, c2, c3 = st.columns([2, 1, 1])
    q = c1.text_input("Search name or ID")
    stt = c2.selectbox("Status", ["All"] + [n for n, *_ in core.BANDS])
    rng = c3.selectbox("Date range", list(RANGES), index=3)
    rows = [e for e in in_range(evals, RANGES[rng]) if (stt == "All" or e["status"] == stt)
            and q.lower() in (e["name"] + e["id"]).lower()]
    st.caption(f"{len(rows)} of {len(evals)} evaluations. Select a row to open it.")
    if rows:
        ss.setdefault("hist_view_mode", "Data Grid (Table)")
        h_c1, h_c2, _ = st.columns([1.2, 1.2, 3])
        if h_c1.button("📊 Data Grid (Table)", key="btn_hist_table", type="primary" if ss.hist_view_mode == "Data Grid (Table)" else "secondary", use_container_width=True):
            ss.hist_view_mode = "Data Grid (Table)"
            st.rerun()
        if h_c2.button("🃏 Card Grid", key="btn_hist_grid", type="primary" if ss.hist_view_mode == "Card Grid" else "secondary", use_container_width=True):
            ss.hist_view_mode = "Card Grid"
            st.rerun()
        if ss.hist_view_mode == "Data Grid (Table)":
            ev = st.dataframe(
                to_df(rows),
                column_config=GRID_COLUMN_CONFIG,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="hist"
            )
            sel = ev.selection.rows
            if sel and sel[0] < len(rows):
                e = rows[sel[0]]
                a, b, c, d = st.columns(4)
                a.button("Edit evaluation", on_click=go_to, args=("New Evaluation",), kwargs={"edit_id": e["id"]})
                b.download_button("Export CSV", core.to_csv([e]), f"{e['id']}.csv", "text/csv")
                c.download_button("Download PDF", core.to_pdf([e], e["name"]), f"{e['id']}.pdf", "application/pdf")
                if d.button("Delete evaluation"):
                    ss.confirm_del = e["id"]
                if ss.get("confirm_del") == e["id"]:
                    st.warning(f"Delete {e['id']}? This cannot be undone.")
                    y, n = st.columns(2)
                    if y.button("Yes, delete"):
                        core.delete_eval(e["id"]); ss.pop("confirm_del"); st.rerun()
                    if n.button("Cancel"):
                        ss.pop("confirm_del"); st.rerun()
                detail(e, "hist")
        else:
            render_eval_grid(rows, key_prefix="hist_grid", cols_count=3)
        st.download_button("Export filtered results (CSV)", core.to_csv(rows), "evaluations.csv", "text/csv")
    else:
        st.info("No evaluations match.")

elif page == "Analytics":
    st.title("Analytics")
    rng = st.selectbox("Date range", list(RANGES), index=1)
    l = in_range(evals, RANGES[rng])
    st.caption(f"{len(l)} evaluations in range.")
    if not l:
        st.info("No data in this range.")
    else:
        df = pd.DataFrame({"t": [e["created_at"] for e in l], "s": [e["overall"] for e in l]})
        a, b = st.columns(2)
        bins = pd.cut(df.s, [-1, 59, 69, 79, 89, 100], labels=["0–59", "60–69", "70–79", "80–89", "90–100"])
        a.plotly_chart(apply_grid(px.bar(bins.value_counts().sort_index(), title="Score distribution",
                              labels={"value": "Evaluations", "s": "Score"})), key="an1")
        avg = [sum(e["scores"][k] for e in l) / len(l) for k in core.DIMS]
        b.plotly_chart(apply_grid(px.bar(x=DL, y=avg, title="Average by dimension (out of 10)", labels={"x": "", "y": "Avg"})), key="an2")
        c, d = st.columns(2)
        c.plotly_chart(radar({"Average": [round(x, 1) for x in avg]}), key="an3")
        cnt = pd.Series([e["status"] for e in l]).value_counts().reindex([n for n, *_ in core.BANDS], fill_value=0)
        d.plotly_chart(px.pie(values=cnt.values, names=cnt.index, hole=.55, title="Status breakdown",
                              color=cnt.index, color_discrete_map=core.STATUS_COLORS), key="an4")
        freq = "D" if RANGES[rng] in (7, 30) else "W" if RANGES[rng] == 90 else "M"
        g = df.set_index("t").resample(freq)["s"].agg(["mean", "count"]).reset_index()
        e1, e2 = st.columns(2)
        e1.plotly_chart(apply_grid(px.line(g.dropna(), x="t", y="mean", markers=True, title="Quality trend (avg %)",
                                labels={"t": "", "mean": "Score"})), key="an5")
        e2.plotly_chart(apply_grid(px.bar(g, x="t", y="count", title="Evaluation volume", labels={"t": "", "count": "Evaluations"})), key="an6")

elif page == "Compare Responses":
    st.title("Compare Responses")

    def compare_view(a, b):
        x, y = st.columns(2)
        for col, e, o in ((x, a, b), (y, b, a)):
            with col:
                st.markdown(f"### {e['name']}")
                st.markdown(badge(e["status"]), unsafe_allow_html=True)
                st.metric("Overall", f"{e['overall']}/100", e["overall"] - o["overall"])
                for k in core.DIMS:
                    st.write(f"{core.LABELS[k]}: **{e['scores'][k]}/10** ({e['scores'][k] - o['scores'][k]:+d})")
                st.text_area("Response", e["response"], height=160, disabled=True, key=f"cmp{e['id']}{id(col)}")
        st.plotly_chart(radar({a["name"][:30]: a, b["name"][:30]: b}), key="cmp_radar")
        st.success("Tie" if a["overall"] == b["overall"] else
                   f"{(a if a['overall'] > b['overall'] else b)['name']} scores higher")

    t1, t2 = st.tabs(["Enter new responses", "Compare saved evaluations"])
    with t1:
        with st.form("cmp_form"):
            p = st.text_area("Your prompt *")
            r1, r2 = st.columns(2)
            ra = r1.text_area("Response A *", height=180)
            rb = r2.text_area("Response B *", height=180)
            exp = st.text_input("Expected answer (optional)")
            src = st.text_area("Reference sources (optional)", height=80)
            ok = st.form_submit_button("Evaluate & compare", type="primary")
        if ok:
            if len(p.strip()) < 5 or len(ra.strip()) < 20 or len(rb.strip()) < 20:
                st.error("Enter a prompt (5+ characters) and both responses (20+ characters each).")
            else:
                with st.spinner("Evaluating…"):
                    a = run_eval(f"Compare A: {p[:30]}", p, ra, exp, src, core.ai_available())
                    b = run_eval(f"Compare B: {p[:30]}", p, rb, exp, src, core.ai_available())
                st.caption("Both responses were saved to History.")
                compare_view(a, b)
    with t2:
        if len(evals) < 2:
            st.info("Need at least two evaluations.")
        else:
            fmt = lambda e: f"{e['id']} · {e['name']} · {e['overall']}/100"
            a = st.selectbox("Response A", evals, format_func=fmt, index=None, key="sa")
            b = st.selectbox("Response B", evals, format_func=fmt, index=None, key="sb")
            if a and b:
                compare_view(a, b) if a["id"] != b["id"] else st.warning("Pick two different evaluations.")

elif page == "Reports":
    st.title("Reports")
    n = len(evals)
    avg = round(sum(e["overall"] for e in evals) / n) if n else 0
    c = st.columns(4)
    c[0].metric("Evaluations", n)
    c[1].metric("Average score", f"{avg}/100")
    c[2].metric("Avg Relevance", round(sum(e["scores"]["relevance"] for e in evals) / n, 1) if n else 0)
    c[3].metric("Avg Factual Support", round(sum(e["scores"]["factual"] for e in evals) / n, 1) if n else 0)
    if n:
        st.write(" · ".join(f"{s}: **{sum(e['status'] == s for e in evals)}**" for s, *_ in core.BANDS))
        a, b = st.columns(2)
        a.download_button("Download CSV", core.to_csv(evals), "evaluations.csv", "text/csv")
        b.download_button("Download detailed PDF report", core.to_pdf(evals), "evaluation-report.pdf", "application/pdf")
        st.caption("The PDF lists up to 60 evaluations and gives full detail for the 25 most recent non-sample ones.")
        st.dataframe(to_df(evals[:15]), column_config=GRID_COLUMN_CONFIG, use_container_width=True, hide_index=True)

elif page == "Settings":
    st.title("Settings")
    provider = core.active_ai_provider()
    st.write("**Evaluation mode:** " + (f"AI judge (**{provider.upper()}** configured)" if provider else
             "Demo (deterministic heuristic). Set an API key below or in `.env` for real AI judging."))
    st.write(f"**Database:** `{core.DB_PATH}`")
    st.subheader("API key & Provider")
    st.caption("Paste your API key to turn on the real AI judge. Supports Google Gemini (`AQ.` or `AIza...`), Anthropic Claude (`sk-ant-...`), or FreeLLMAPI / OpenAI (`freellmapi-...` or `sk-...`). For permanent setup, configure `.env`.")
    key = st.text_input("API key", type="password", placeholder="Paste Gemini, Anthropic, or FreeLLMAPI key...")
    k1, k2 = st.columns(2)
    if k1.button("Use this key"):
        val = key.strip()
        if val.startswith("sk-ant-"):
            os.environ["ANTHROPIC_API_KEY"] = val
            st.success("Configured Anthropic Claude API key.")
            st.rerun()
        elif val.startswith("AQ.") or val.startswith("AIza"):
            os.environ["GEMINI_API_KEY"] = val
            st.success("Configured Google Gemini API key.")
            st.rerun()
        elif val.startswith("freellmapi-"):
            os.environ["FREELLMAPI_API_KEY"] = val
            st.success("Configured FreeLLMAPI key.")
            st.rerun()
        elif val:
            # Default to Gemini if key is provided
            os.environ["GEMINI_API_KEY"] = val
            st.success("Configured API key.")
            st.rerun()
        else:
            st.error("Please enter a valid API key.")
    if k2.button("Remove active keys", disabled=not core.ai_available()):
        for var in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "FREELLMAPI_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(var, None)
        st.rerun()
    st.divider()
    if st.button("Reset to sample data"):
        core.clear_all(); core.seed_samples(); st.rerun()
    sure = st.checkbox("I understand this deletes every evaluation")
    if st.button("Clear all evaluations", disabled=not sure):
        core.clear_all(); st.rerun()
