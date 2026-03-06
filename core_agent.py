"""
core_agent.py
=============
LangChain + Gemini Data Analyst Agent — Core Logic
Supports CSV, Excel (.xlsx, .xls), and JSON files
"""

import os
import io
import json
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage

warnings.filterwarnings("ignore")
load_dotenv()

# ─── Palette ─────────────────────────────────────────────────────────────────
PALETTE = ["#6C63FF", "#FF6584", "#43E97B", "#F7971E", "#4FC3F7", "#CE93D8"]
DARK_BG  = "#0F0F1A"
CARD_BG  = "#1A1A2E"


# ─── LLM Setup ───────────────────────────────────────────────────────────────
def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3,
        convert_system_message_to_human=True,
    )


# ─── File Loading ─────────────────────────────────────────────────────────────
def load_file(file) -> tuple[pd.DataFrame, str]:
    """Load uploaded file into a DataFrame. Returns (df, file_type)."""
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
        return df, "CSV"
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)
        return df, "Excel"
    elif name.endswith(".json"):
        content = json.load(file)
        if isinstance(content, list):
            df = pd.DataFrame(content)
        elif isinstance(content, dict):
            df = pd.DataFrame([content]) if not any(isinstance(v, list) for v in content.values()) \
                 else pd.DataFrame(content)
        return df, "JSON"
    else:
        raise ValueError(f"Unsupported file type: {name}")


# ─── Data Profile ─────────────────────────────────────────────────────────────
def profile_dataframe(df: pd.DataFrame) -> dict:
    """Generate a rich statistical profile of the dataframe."""
    numeric_cols  = df.select_dtypes(include="number").columns.tolist()
    category_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    profile = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_columns": numeric_cols,
        "categorical_columns": category_cols,
        "datetime_columns": datetime_cols,
        "null_counts": df.isnull().sum().to_dict(),
        "null_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }

    if numeric_cols:
        desc = df[numeric_cols].describe().round(3)
        profile["numeric_stats"] = desc.to_dict()

    if category_cols:
        profile["top_categories"] = {
            col: df[col].value_counts().head(5).to_dict()
            for col in category_cols
        }

    return profile


def profile_to_text(profile: dict, df: pd.DataFrame) -> str:
    """Convert profile dict to LLM-readable text summary."""
    rows, cols = profile["shape"]
    lines = [
        f"Dataset: {rows} rows × {cols} columns",
        f"Numeric columns : {', '.join(profile['numeric_columns']) or 'None'}",
        f"Categorical cols : {', '.join(profile['categorical_columns']) or 'None'}",
        f"Datetime cols    : {', '.join(profile['datetime_columns']) or 'None'}",
        f"Missing values   : {sum(profile['null_counts'].values())} total",
        f"Duplicate rows   : {profile['duplicates']}",
        "",
        "--- Sample Data (first 5 rows) ---",
        df.head(5).to_string(index=False),
    ]
    if profile.get("numeric_stats"):
        lines += ["", "--- Numeric Stats ---"]
        for col, stats in profile["numeric_stats"].items():
            lines.append(f"  {col}: mean={stats.get('mean','?')}, std={stats.get('std','?')}, "
                         f"min={stats.get('min','?')}, max={stats.get('max','?')}")
    return "\n".join(lines)


# ─── AI Question Answering ────────────────────────────────────────────────────
def ask_agent(question: str, df: pd.DataFrame, profile: dict, llm) -> str:
    """Send a question + data context to Gemini and return the answer."""
    data_context = profile_to_text(profile, df)

    system = """You are an expert data analyst AI. You receive a dataset summary and answer questions about it.
Be precise, insightful, and helpful. When relevant, suggest what visualizations would best illustrate the answer.
Format your response clearly. Use bullet points for lists. Use numbers and percentages when quoting statistics."""

    user_msg = f"""Here is the dataset context:

{data_context}

User question: {question}

Provide a thorough, accurate analysis. If you perform calculations, show the logic briefly."""

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ]

    response = llm.invoke(messages)
    return response.content


# ─── Visualization Engine ─────────────────────────────────────────────────────
def auto_suggest_charts(profile: dict) -> list[str]:
    """Suggest relevant chart types based on data profile."""
    suggestions = []
    if len(profile["numeric_columns"]) >= 2:
        suggestions.append("correlation_heatmap")
        suggestions.append("scatter_matrix")
    if profile["numeric_columns"]:
        suggestions.append("distribution_plots")
        suggestions.append("box_plots")
    if profile["categorical_columns"] and profile["numeric_columns"]:
        suggestions.append("bar_chart")
        suggestions.append("pie_chart")
    if profile["datetime_columns"] and profile["numeric_columns"]:
        suggestions.append("time_series")
    return suggestions


def make_plotly_chart(chart_type: str, df: pd.DataFrame, profile: dict,
                      x_col: str = None, y_col: str = None, color_col: str = None):
    """Generate a Plotly figure for the given chart type."""
    num_cols = profile["numeric_columns"]
    cat_cols = profile["categorical_columns"]

    template = "plotly_dark"

    if chart_type == "correlation_heatmap" and len(num_cols) >= 2:
        corr = df[num_cols].corr().round(2)
        fig = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r",
            title="Correlation Heatmap", template=template,
            color_continuous_midpoint=0,
        )

    elif chart_type == "distribution_plots" and num_cols:
        col = y_col or num_cols[0]
        fig = px.histogram(
            df, x=col, nbins=30, marginal="box",
            title=f"Distribution of {col}",
            color_discrete_sequence=PALETTE,
            template=template,
        )

    elif chart_type == "box_plots" and num_cols:
        cols = num_cols[:6]
        fig = go.Figure()
        for i, col in enumerate(cols):
            fig.add_trace(go.Box(y=df[col], name=col, marker_color=PALETTE[i % len(PALETTE)]))
        fig.update_layout(title="Box Plots — Numeric Columns", template=template)

    elif chart_type == "bar_chart" and cat_cols and num_cols:
        xc = x_col or cat_cols[0]
        yc = y_col or num_cols[0]
        agg = df.groupby(xc)[yc].mean().reset_index().sort_values(yc, ascending=False).head(15)
        fig = px.bar(
            agg, x=xc, y=yc, color=yc,
            color_continuous_scale="Viridis",
            title=f"Average {yc} by {xc}", template=template,
        )

    elif chart_type == "pie_chart" and cat_cols:
        col = x_col or cat_cols[0]
        counts = df[col].value_counts().head(8)
        fig = px.pie(
            values=counts.values, names=counts.index,
            title=f"Distribution of {col}",
            color_discrete_sequence=PALETTE,
            template=template,
        )

    elif chart_type == "scatter_matrix" and len(num_cols) >= 2:
        cols = num_cols[:4]
        fig = px.scatter_matrix(
            df, dimensions=cols,
            color=cat_cols[0] if cat_cols else None,
            color_discrete_sequence=PALETTE,
            title="Scatter Matrix", template=template,
        )
        fig.update_traces(diagonal_visible=False, showupperhalf=False)

    elif chart_type == "time_series" and profile["datetime_columns"] and num_cols:
        dt_col = profile["datetime_columns"][0]
        yc = y_col or num_cols[0]
        fig = px.line(
            df.sort_values(dt_col), x=dt_col, y=yc,
            title=f"{yc} over Time",
            color_discrete_sequence=PALETTE,
            template=template,
        )

    elif chart_type == "scatter" and len(num_cols) >= 2:
        xc = x_col or num_cols[0]
        yc = y_col or num_cols[1]
        fig = px.scatter(
            df, x=xc, y=yc,
            color=color_col or (cat_cols[0] if cat_cols else None),
            color_discrete_sequence=PALETTE,
            title=f"{xc} vs {yc}",
            trendline="ols",
            template=template,
        )

    elif chart_type == "line" and num_cols:
        xc = x_col or (profile["datetime_columns"][0] if profile["datetime_columns"] else num_cols[0])
        yc = y_col or num_cols[0]
        fig = px.line(
            df, x=xc, y=yc,
            color_discrete_sequence=PALETTE,
            title=f"{yc} trend",
            template=template,
        )

    else:
        # Fallback: summary bar
        if num_cols:
            means = df[num_cols[:8]].mean()
            fig = px.bar(
                x=means.index, y=means.values,
                labels={"x": "Column", "y": "Mean Value"},
                color=means.values, color_continuous_scale="Viridis",
                title="Column Means Overview", template=template,
            )
        else:
            fig = go.Figure()
            fig.add_annotation(text="No numeric data available for this chart type.",
                               showarrow=False, font=dict(size=14))
            fig.update_layout(template=template, title="Chart Unavailable")

    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="DM Sans, sans-serif", color="#E0E0FF"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


# ─── AI-Driven Chart Recommendation ──────────────────────────────────────────
def ai_recommend_chart(question: str, profile: dict, llm) -> dict:
    """Ask Gemini which chart best answers the user's question."""
    num_cols  = profile["numeric_columns"]
    cat_cols  = profile["categorical_columns"]
    dt_cols   = profile["datetime_columns"]

    prompt = f"""Given this dataset profile:
- Numeric columns: {num_cols}
- Categorical columns: {cat_cols}
- Datetime columns: {dt_cols}

The user asked: "{question}"

Recommend ONE chart type from this list that best answers their question:
[correlation_heatmap, distribution_plots, box_plots, bar_chart, pie_chart, scatter, line, time_series, scatter_matrix]

Also suggest the best x_col and y_col from the available columns.

Respond ONLY in valid JSON like:
{{"chart_type": "bar_chart", "x_col": "category_col", "y_col": "numeric_col", "reason": "short explanation"}}"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()
        # strip markdown fences if present
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception:
        return {"chart_type": "distribution_plots", "x_col": None, "y_col": None, "reason": "Default chart"}
