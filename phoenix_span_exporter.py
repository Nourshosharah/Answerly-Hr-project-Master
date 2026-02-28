#!/usr/bin/env python
# coding: utf-8

# In[6]:


"""
Phoenix Span Export & Analysis Utilities
========================================

This module provides a clean, production-ready interface
to extract, query, and persist Phoenix tracing spans.

Compatible with:
- Phoenix (Arize)
- OpenTelemetry OTLP ingestion
- RAG pipelines (retrieval, LLM, evaluation spans)

Author: Answerly Project
"""

from __future__ import annotations

import os
from typing import Optional, List
from datetime import datetime

import pandas as pd
import phoenix as px
from phoenix.trace.dsl import SpanQuery


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", None)
EXPORT_DIR = os.getenv("PHOENIX_EXPORT_DIR", "phoenix_exports")


# ---------------------------------------------------------------------
# Phoenix Client Wrapper
# ---------------------------------------------------------------------

class PhoenixSpanExporter:
    """
    High-level wrapper around Phoenix Client for
    span extraction and analysis.
    """

    def __init__(self, project_name: Optional[str] = DEFAULT_PROJECT_NAME):
        self.client = px.Client()
        self.project_name = project_name

    # -----------------------------------------------------------------
    # Core Export APIs
    # -----------------------------------------------------------------

    def get_all_spans(self) -> pd.DataFrame:
        """
        Export ALL spans visible to Phoenix into a DataFrame.
        """
        return self.client.get_spans_dataframe(
            project_name=self.project_name
        )

    def query_spans(self, query: SpanQuery) -> pd.DataFrame:
        """
        Run an arbitrary SpanQuery and return a DataFrame.
        """
        return self.client.query_spans(query)

    # -----------------------------------------------------------------
    # Common RAG Queries
    # -----------------------------------------------------------------

    def get_llm_spans(self) -> pd.DataFrame:
        """
        Retrieve all LLM spans with token & cost info.
        """
        query = (
            SpanQuery()
            .where("openinference.span.kind == 'LLM'")
            .select(
                "trace_id",
                "span_id",
                "start_time",
                "end_time",
                "llm.model_name",
                "llm.tokens.prompt",
                "llm.tokens.completion",
                "llm.tokens.total",
                "llm.cost.total",
                "input.value",
                "output.value",
            )
        )
        return self.query_spans(query)

    def get_retrieval_spans(self) -> pd.DataFrame:
        """
        Retrieve retriever spans (vector search).
        """
        query = (
            SpanQuery()
            .where("span_name == 'retrieval'")
            .select(
                "trace_id",
                "span_id",
                "retrieval.doc_count",
                "retrieval.score_avg",
                "retrieval.score_min",
                "retrieval.score_max",
                "retrieval.context_preview",
            )
        )
        return self.query_spans(query)

    def get_rag_traces(self) -> pd.DataFrame:
        """
        Retrieve root RAG pipeline spans.
        """
        query = (
            SpanQuery()
            .where("span_name == 'rag_pipeline'")
            .select(
                "trace_id",
                "span_id",
                "session.id",
                "user.id",
                "user.query",
                "llm.token_count.total",
                "llm.cost.total",
                "response.length",
                "response.is_error",
            )
        )
        return self.query_spans(query)

    # -----------------------------------------------------------------
    # Session / Trace Reconstruction
    # -----------------------------------------------------------------

    def get_trace(self, trace_id: str) -> pd.DataFrame:
        """
        Retrieve ALL spans belonging to a single trace.
        """
        query = (
            SpanQuery()
            .where(f"trace_id == '{trace_id}'")
            .select("*")
        )
        return self.query_spans(query)

    def get_session(self, session_id: str) -> pd.DataFrame:
        """
        Retrieve all spans linked to a given session_id.
        """
        query = (
            SpanQuery()
            .where(f"session.id == '{session_id}'")
            .select("*")
        )
        return self.query_spans(query)

    # -----------------------------------------------------------------
    # Persistence Utilities
    # -----------------------------------------------------------------

    def save_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        fmt: str = "parquet"
    ) -> str:
        """
        Persist a DataFrame to disk.
        """
        os.makedirs(EXPORT_DIR, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        path = os.path.join(
            EXPORT_DIR,
            f"{name}_{timestamp}.{fmt}"
        )

        if fmt == "parquet":
            df.to_parquet(path, index=False)
        elif fmt == "csv":
            df.to_csv(path, index=False)
        else:
            raise ValueError("Unsupported format: use parquet or csv")

        return path

    # -----------------------------------------------------------------
    # One-shot Export Helpers
    # -----------------------------------------------------------------

    def export_everything(self) -> dict:
        """
        Export ALL major span groups in one call.
        """
        outputs = {}

        outputs["all_spans"] = self.save_dataframe(
            self.get_all_spans(), "all_spans"
        )

        outputs["rag_spans"] = self.save_dataframe(
            self.get_rag_traces(), "rag_spans"
        )

        outputs["llm_spans"] = self.save_dataframe(
            self.get_llm_spans(), "llm_spans"
        )

        outputs["retrieval_spans"] = self.save_dataframe(
            self.get_retrieval_spans(), "retrieval_spans"
        )

        return outputs

    

    # -----------------------------------------------------------------
    # Load / Concurrency Analysis
    # -----------------------------------------------------------------

    def get_concurrent_users(
        self,
        freq: str = "1min"
    ) -> pd.DataFrame:
        """
        Compute number of concurrent sessions over time.

        freq examples:
        - "1min"  (recommended for load tests)
        - "30s"
        - "5min"
        """
        sessions = self.get_sessions_with_users()

        if sessions.empty:
            return sessions

        # Build time grid
        timeline = pd.date_range(
            start=sessions["session_start"].min(),
            end=sessions["session_end"].max(),
            freq=freq,
        )

        records = []
        for t in timeline:
            concurrent = (
                (sessions["session_start"] <= t) &
                (sessions["session_end"] > t)
            ).sum()

            records.append(
                {"timestamp": t, "concurrent_users": concurrent}
            )

        return pd.DataFrame(records)
        # -----------------------------------------------------------------
    # Burst / Same-second Analysis
    # -----------------------------------------------------------------

    def get_users_started_same_second(self) -> pd.DataFrame:
        """
        Return sessions grouped by the exact second they started.
        Useful for detecting burst / simultaneous logins.
        """
        sessions = self.get_sessions_with_users()

        if sessions.empty:
            return sessions

        # Floor to second resolution
        sessions["start_second"] = (
            sessions["session_start"]
            .dt.floor("S")
        )

        grouped = (
            sessions
            .groupby("start_second")
            .agg(
                session_count=("session.id", "count"),
                users=("user_id", lambda x: list(x.dropna().unique())),
                sessions=("session.id", list),
            )
            .reset_index()
            .sort_values("session_count", ascending=False)
        )

        return grouped
     
    # -----------------------------------------------------------------
    # Internal Utilities
    # -----------------------------------------------------------------

    @staticmethod
    def _to_utc(dt: datetime) -> pd.Timestamp:
        """
        Convert datetime to UTC-aware pandas Timestamp.
        Accepts tz-naive or tz-aware datetimes.
        """
        ts = pd.Timestamp(dt)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")
    def get_sessions_with_users(self) -> pd.DataFrame:
        """
        Return one row per session with associated user info
        and session start/end time (UTC).
        """
        query = (
            SpanQuery()
            .select(
                "session.id",
                "user.id",
                "start_time",
                "end_time",
            )
        )

        df = self.query_spans(query)

        if df.empty:
            return df

        # Ensure UTC-aware timestamps
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
        df["end_time"] = pd.to_datetime(df["end_time"], utc=True)

        sessions = (
            df.groupby("session.id", as_index=False)
              .agg(
                  user_id=("user.id", "first"),
                  session_start=("start_time", "min"),
                  session_end=("end_time", "max"),
              )
        )

        return sessions
    def get_full_sessions(self) -> pd.DataFrame:
        """
        Return full session spans with all columns intact, no aggregation.
        """
        df = self.get_all_spans()  # retrieves everything
        if df.empty:
            return df

        if "start_time" in df.columns:
            df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
        if "end_time" in df.columns:
            df["end_time"] = pd.to_datetime(df["end_time"], utc=True)

        return df
    def filter_sessions_by_date(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Filter sessions that overlap with a given time window.
        Datetimes are normalized to UTC.
        """
        sessions = self.get_sessions_with_users()

        if sessions.empty:
            return sessions

        start_utc = self._to_utc(start)
        end_utc = self._to_utc(end)

        mask = (
            (sessions["session_start"] <= end_utc) &
            (sessions["session_end"] >= start_utc)
        )

        return sessions.loc[mask].reset_index(drop=True)

# ---------------------------------------------------------------------
# CLI Usage (Optional)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    exporter = PhoenixSpanExporter()
    paths = exporter.export_everything()

    print("Phoenix span export completed:")
    for k, v in paths.items():
        print(f"  - {k}: {v}")


# In[ ]:




