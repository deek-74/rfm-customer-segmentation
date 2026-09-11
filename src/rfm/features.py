"""DuckDB-backed customer features and v1 customer-table replica."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def run_sql(
    con: duckdb.DuckDBPyConnection,
    sql_path: Path,
    replacements: dict[str, str] | None = None,
) -> pd.DataFrame:
    sql = sql_path.read_text(encoding="utf-8")
    for key, value in (replacements or {}).items():
        sql = sql.replace("{{" + key + "}}", str(value))
    logger.info("Running %s", sql_path.name)
    return con.execute(sql).df()


def snapshot_date_from_fact(fact: pd.DataFrame) -> pd.Timestamp:
    """Max invoice date normalised to midnight, plus one day."""
    max_date = pd.to_datetime(fact["Date"]).max().normalize()
    return max_date + pd.Timedelta(days=1)


def customer_metrics_v1(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Customer RFM inputs matching the R loop (first-appearance order, first Country)."""
    reference_date = clean["InvoiceDate"].max() + pd.Timedelta(days=1)
    order = clean["CustomerID"].drop_duplicates()
    grouped = clean.groupby("CustomerID", sort=False)
    recency = (reference_date - grouped["InvoiceDate"].max()) / pd.Timedelta(days=1)
    out = pd.DataFrame(
        {
            "CustomerID": recency.index.astype("int64"),
            "Recency": recency.to_numpy(),
            "Frequency": grouped["InvoiceNo"].nunique().to_numpy(),
            "Monetary": grouped["Revenue"].sum().to_numpy(),
            "Country": grouped["Country"].first().to_numpy(),
        }
    )
    out["AvgOrderValue"] = out["Monetary"] / out["Frequency"]
    out = out.set_index("CustomerID").loc[order.to_numpy()].reset_index()
    logger.info(
        "v1 customers=%s invoices=%s revenue=%.2f",
        len(out),
        int(clean["InvoiceNo"].nunique()),
        float(clean["Revenue"].sum()),
    )
    return out, pd.Timestamp(reference_date)


def build_customer_features(
    fact: pd.DataFrame, cfg: dict[str, Any], snapshot: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Run sql/02_customer_metrics.sql against fact_sales for one snapshot."""
    snap = snapshot if snapshot is not None else snapshot_date_from_fact(fact)
    sql_dir: Path = cfg["paths"]["sql_dir"]
    con = duckdb.connect()
    try:
        con.register("fact_sales", fact)
        customers = run_sql(
            con, sql_dir / "02_customer_metrics.sql",
            {"snapshot_date": snap.strftime("%Y-%m-%d")},
        )
    finally:
        con.close()
    customers["CustomerID"] = customers["CustomerID"].astype("int64")
    customers["recency_days"] = customers["recency_days"].astype(int)
    customers["tenure_days"] = customers["tenure_days"].astype(int)
    customers["frequency"] = customers["frequency"].astype(int)
    logger.info("v2 customers=%s monetary=%.2f", len(customers), float(customers["monetary"].sum()))
    return customers


def monthly_kpis(fact: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    sql_dir: Path = cfg["paths"]["sql_dir"]
    con = duckdb.connect()
    try:
        con.register("fact_sales", fact)
        out = run_sql(con, sql_dir / "03_monthly_kpis.sql")
    finally:
        con.close()
    return out


def product_summary(fact: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    sql_dir: Path = cfg["paths"]["sql_dir"]
    con = duckdb.connect()
    try:
        con.register("fact_sales", fact)
        out = run_sql(con, sql_dir / "05_product_summary.sql")
    finally:
        con.close()
    return out


def cohort_activity(fact: pd.DataFrame, dim_customer: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    sql_dir: Path = cfg["paths"]["sql_dir"]
    con = duckdb.connect()
    try:
        con.register("fact_sales", fact)
        con.register("dim_customer", dim_customer)
        out = run_sql(con, sql_dir / "04_cohorts.sql")
    finally:
        con.close()
    return out


def stage_fact(stg_lines: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    sql_dir: Path = cfg["paths"]["sql_dir"]
    con = duckdb.connect()
    try:
        con.register("stg_lines", stg_lines)
        out = run_sql(con, sql_dir / "01_stage_transactions.sql")
    finally:
        con.close()
    return out
