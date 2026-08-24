import os
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st
import config

st.set_page_config(page_title="RAG Monitor Dashboard")
st.title("RAG Chatbot - Monitoring Dashboard")

@st.cache_data(ttl=10)
def load_logs(db_path: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM query_logs ORDER BY timestamp DESC", conn)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        if "total_latency_ms" not in df.columns:
            df["total_latency_ms"] = (
                df["retrieval_latency_ms"] + df["generation_latency_ms"]
            )
    return df

df=load_logs(config.LOG_DB_PATH)

if df.empty:
    st.info("No query logs yet. Ask some questions in the chatbot first.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total queries", len(df))
col2.metric("Avg total latency", f"{df['total_latency_ms'].mean():.0f} ms")
col3.metric("Avg retrieval latency", f"{df['retrieval_latency_ms'].mean():.0f} ms")
col4.metric("Avg generation latency", f"{df['generation_latency_ms'].mean():.0f} ms")

st.divider()

st.subheader("Latency over time")
fig_latency = px.line(
    df.sort_values("timestamp"),
    x="timestamp",
    y=["retrieval_latency_ms", "generation_latency_ms", "total_latency_ms"],
    labels={"value": "Latency (ms)", "timestamp": "Time", "variable": "Stage"},
)
st.plotly_chart(fig_latency, use_container_width=True)

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Retrieved chunks per query")
    fig_chunks = px.histogram(df, x="retrieved_chunks", nbins=10)
    st.plotly_chart(fig_chunks, use_container_width=True)

with col_b:
    st.subheader("Most frequently retrieved sources")
    if "top_source" in df.columns and df["top_source"].notna().any():
        top_sources = df["top_source"].value_counts().reset_index()
        top_sources.columns = ["source", "count"]
        fig_sources = px.bar(top_sources, x="source", y="count")
        st.plotly_chart(fig_sources, use_container_width=True)
    else:
        st.write("No source data logged yet.")

st.subheader("Latency by chunking configuration")
if {"chunk_size", "chunk_overlap"}.issubset(df.columns):
    df["config_label"] = (
        "size=" + df["chunk_size"].astype(str) + ", overlap=" + df["chunk_overlap"].astype(str)
    )
    config_summary = (
        df.groupby("config_label")["total_latency_ms"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_total_latency_ms", "count": "num_queries"})
    )
    st.dataframe(config_summary, use_container_width=True)

st.divider()

st.subheader("Recent queries")
st.dataframe(
    df[["timestamp", "question", "answer", "retrieved_chunks", "total_latency_ms"]].head(50),
    use_container_width=True,
)
