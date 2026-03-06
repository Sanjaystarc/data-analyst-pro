"""
app.py
======
Streamlit UI — Data Analyst Agent (LangChain + Gemini)
Run: streamlit run app.py
"""

import os
import io
import streamlit as st
import pandas as pd
import plotly.express as px

from core_agent import (
    get_llm, load_file, profile_dataframe, profile_to_text,
    ask_agent, auto_suggest_charts, make_plotly_chart, ai_recommend_chart
)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataMind Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a12;
    color: #e8e8ff;
}

.main { background-color: #0a0a12; }

/* Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e8e8ff 0%, #6C63FF 50%, #43E97B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: #6a6a9a;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* Cards */
.stat-card {
    background: #1a1a2e;
    border: 1px solid #2a2a45;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #6C63FF;
}
.stat-label { color: #6a6a9a; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; }

/* Chat bubbles */
.user-bubble {
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 18px 18px 4px 18px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
}
.agent-bubble {
    background: #1a1a2e;
    border: 1px solid #2a2a45;
    border-radius: 18px 18px 18px 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #10101e;
    border-right: 1px solid #2a2a45;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF, #43E97B);
    color: white;
    border: none;
    border-radius: 12px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    padding: 0.6rem 1.5rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; color: white; }

.stTextInput > div > div > input {
    background: #1a1a2e;
    border: 1px solid #2a2a45;
    border-radius: 12px;
    color: #e8e8ff;
}
.stSelectbox > div > div {
    background: #1a1a2e;
    border: 1px solid #2a2a45;
    border-radius: 12px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #10101e;
    border-radius: 12px;
    gap: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6a6a9a;
    border-radius: 10px;
    font-family: 'Syne', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: rgba(108,99,255,0.2) !important;
    color: #6C63FF !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
for key, default in {
    "df": None,
    "profile": None,
    "file_type": None,
    "chat_history": [],
    "llm": None,
    "api_key_set": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 DataMind Agent")
    st.markdown("---")

    # API Key
    st.markdown("**🔑 Gemini API Key**")
    api_key = st.text_input(
        "Enter your key", type="password",
        placeholder="AIza...",
        help="Get free key at aistudio.google.com",
        label_visibility="collapsed"
    )
    if api_key:
        if not st.session_state.api_key_set or st.session_state.get("_last_key") != api_key:
            try:
                st.session_state.llm = get_llm(api_key)
                st.session_state.api_key_set = True
                st.session_state["_last_key"] = api_key
                st.success("✅ Connected to Gemini!")
            except Exception as e:
                st.error(f"❌ Invalid key: {e}")

    st.markdown("---")

    # File Upload
    st.markdown("**📁 Upload Data File**")
    uploaded = st.file_uploader(
        "Upload", type=["csv", "xlsx", "xls", "json"],
        label_visibility="collapsed"
    )

    if uploaded and st.session_state.api_key_set:
        with st.spinner("📊 Analyzing your data..."):
            try:
                df, ftype = load_file(uploaded)
                st.session_state.df = df
                st.session_state.file_type = ftype
                st.session_state.profile = profile_dataframe(df)
                st.session_state.chat_history = []
                st.success(f"✅ Loaded {ftype} file!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    elif uploaded and not st.session_state.api_key_set:
        st.warning("⚠️ Enter your Gemini API key first")

    st.markdown("---")
    st.markdown("""
**How to use:**
1. Paste your Gemini API key above
2. Upload CSV, Excel, or JSON file
3. Explore the Dashboard tab
4. Ask questions in Chat tab
5. Generate visuals in Charts tab

---
**Get free Gemini API key:**
[aistudio.google.com](https://aistudio.google.com/app/apikey)
""")


# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 DataMind Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered data analysis using LangChain + Gemini · Upload any data file and start exploring</div>', unsafe_allow_html=True)

if st.session_state.df is None:
    # Landing state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-num">📂</div>
            <div class="stat-label">CSV, Excel, JSON</div>
            <br><p style="color:#6a6a9a; font-size:0.85rem">Upload any tabular data file — we handle the parsing automatically</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-num">💬</div>
            <div class="stat-label">Natural Language Q&A</div>
            <br><p style="color:#6a6a9a; font-size:0.85rem">Ask anything about your data in plain English — no SQL needed</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-num">📊</div>
            <div class="stat-label">Smart Visualizations</div>
            <br><p style="color:#6a6a9a; font-size:0.85rem">AI picks the right chart for your question automatically</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Enter your Gemini API key and upload a data file in the sidebar to get started!")

else:
    df      = st.session_state.df
    profile = st.session_state.profile
    llm     = st.session_state.llm

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "💬 Chat", "🎨 Charts", "🔍 Raw Data"])

    # ════════════════════════════════════════════════════════════════
    # TAB 1 — Dashboard
    # ════════════════════════════════════════════════════════════════
    with tab1:
        rows, cols = profile["shape"]
        nulls  = sum(profile["null_counts"].values())
        num_c  = len(profile["numeric_columns"])
        cat_c  = len(profile["categorical_columns"])

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="stat-card"><div class="stat-num">{rows:,}</div><div class="stat-label">Rows</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card"><div class="stat-num">{cols}</div><div class="stat-label">Columns</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-card"><div class="stat-num">{num_c}</div><div class="stat-label">Numeric Cols</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-card"><div class="stat-num">{nulls}</div><div class="stat-label">Missing Values</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Column overview
        st.markdown("#### 📋 Column Overview")
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Null %": (df.isnull().mean() * 100).round(1).values,
            "Unique": df.nunique().values,
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)

        # Auto charts
        st.markdown("#### 🤖 Auto-Generated Insights")
        suggested = auto_suggest_charts(profile)[:3]

        chart_cols = st.columns(min(len(suggested), 2))
        for i, ctype in enumerate(suggested[:2]):
            with chart_cols[i]:
                try:
                    fig = make_plotly_chart(ctype, df, profile)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render {ctype}: {e}")

        if len(suggested) > 2:
            try:
                fig = make_plotly_chart(suggested[2], df, profile)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        # AI summary
        st.markdown("#### 🧠 AI Dataset Summary")
        if st.button("✨ Generate AI Summary"):
            with st.spinner("Gemini is analyzing your dataset..."):
                summary = ask_agent(
                    "Give me a concise executive summary of this dataset. "
                    "Highlight key patterns, anomalies, and 3 actionable insights.",
                    df, profile, llm
                )
                st.markdown(f'<div class="agent-bubble">{summary}</div>', unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════
    # TAB 2 — Chat
    # ════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### 💬 Ask Anything About Your Data")
        st.markdown("*The AI has full context of your dataset and can answer complex analytical questions.*")

        # Suggested questions
        st.markdown("**Quick questions to try:**")
        suggestions = [
            "What are the top 5 most important patterns in this data?",
            "Are there any outliers or anomalies I should know about?",
            "What correlations exist between the numeric columns?",
            "Summarize the distribution of categorical columns.",
            "What would you recommend analyzing further?",
        ]
        q_cols = st.columns(3)
        for i, s in enumerate(suggestions[:3]):
            with q_cols[i]:
                if st.button(s, key=f"sug_{i}"):
                    st.session_state["prefill_q"] = s

        # Chat history
        for turn in st.session_state.chat_history:
            st.markdown(f'<div class="user-bubble">👤 {turn["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="agent-bubble">🧠 {turn["agent"]}</div>', unsafe_allow_html=True)

        # Input
        prefill = st.session_state.pop("prefill_q", "")
        question = st.text_input(
            "Ask a question...",
            value=prefill,
            placeholder="e.g. What's the average sales by region?",
            label_visibility="collapsed",
        )

        col_send, col_clear = st.columns([1, 5])
        with col_send:
            send = st.button("Send 🚀")
        with col_clear:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

        if send and question.strip():
            with st.spinner("🧠 Gemini is thinking..."):
                answer = ask_agent(question, df, profile, llm)

                # Auto-generate relevant chart
                chart_rec = ai_recommend_chart(question, profile, llm)
                st.session_state.chat_history.append({
                    "user": question,
                    "agent": answer,
                    "chart_rec": chart_rec,
                })

            st.markdown(f'<div class="user-bubble">👤 {question}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="agent-bubble">🧠 {answer}</div>', unsafe_allow_html=True)

            # Show recommended chart
            if chart_rec:
                st.markdown(f"*📊 Suggested chart: **{chart_rec['chart_type']}** — {chart_rec.get('reason','')}*")
                try:
                    fig = make_plotly_chart(
                        chart_rec["chart_type"], df, profile,
                        x_col=chart_rec.get("x_col"),
                        y_col=chart_rec.get("y_col"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass


    # ════════════════════════════════════════════════════════════════
    # TAB 3 — Charts
    # ════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("#### 🎨 Custom Chart Builder")

        chart_options = {
            "Correlation Heatmap": "correlation_heatmap",
            "Distribution Plot": "distribution_plots",
            "Box Plots": "box_plots",
            "Bar Chart": "bar_chart",
            "Pie Chart": "pie_chart",
            "Scatter Plot": "scatter",
            "Line Chart": "line",
            "Scatter Matrix": "scatter_matrix",
        }
        if profile["datetime_columns"]:
            chart_options["Time Series"] = "time_series"

        c1, c2, c3 = st.columns(3)
        with c1:
            chart_label = st.selectbox("Chart Type", list(chart_options.keys()))
        with c2:
            all_cols = ["(auto)"] + df.columns.tolist()
            x_col = st.selectbox("X Column", all_cols)
        with c3:
            y_col = st.selectbox("Y Column", all_cols)

        x_val = None if x_col == "(auto)" else x_col
        y_val = None if y_col == "(auto)" else y_col

        if st.button("🎨 Generate Chart"):
            with st.spinner("Rendering..."):
                try:
                    fig = make_plotly_chart(
                        chart_options[chart_label], df, profile,
                        x_col=x_val, y_col=y_val
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {e}")

        st.markdown("---")
        st.markdown("#### 📊 All Auto-Suggested Charts")
        suggested_all = auto_suggest_charts(profile)
        for i in range(0, len(suggested_all), 2):
            cols = st.columns(2)
            for j, ctype in enumerate(suggested_all[i:i+2]):
                with cols[j]:
                    try:
                        fig = make_plotly_chart(ctype, df, profile)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not render {ctype}")


    # ════════════════════════════════════════════════════════════════
    # TAB 4 — Raw Data
    # ════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("#### 🔍 Raw Data Explorer")

        # Search/filter
        search = st.text_input("🔎 Filter rows containing...", placeholder="Type to filter...")
        if search:
            mask = df.astype(str).apply(lambda row: row.str.contains(search, case=False, na=False)).any(axis=1)
            display_df = df[mask]
            st.info(f"Showing {len(display_df):,} of {len(df):,} rows matching '{search}'")
        else:
            display_df = df

        st.dataframe(display_df, use_container_width=True, height=500)

        # Download
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download as CSV",
            data=csv_buf.getvalue(),
            file_name="analyzed_data.csv",
            mime="text/csv"
        )
