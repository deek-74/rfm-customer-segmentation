"""Transaction cleaning: v1 (R-parity) and v2 (classification + partial months)."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_v1(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Reproduce reference/script.R cleaning, in the same order.

    Returns the cleaned frame plus row-count checkpoints used by the parity tests.
    """
    counts: dict[str, int] = {"raw": int(len(raw))}
    df = raw.copy()
    df = df.drop_duplicates()
    counts["after_dedupe"] = int(len(df))
    df = df.loc[df["CustomerID"].notna()].copy()
    counts["after_customer_id"] = int(len(df))
    df = df.loc[df["Description"].notna()].copy()
    counts["after_description"] = int(len(df))
    invoice = df["InvoiceNo"].astype(str)
    df = df.loc[~invoice.str.startswith("C")].copy()
    counts["after_cancellations"] = int(len(df))
    df = df.loc[df["Quantity"] > 0].copy()
    df = df.loc[df["UnitPrice"] > 0].copy()
    counts["final"] = int(len(df))
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["CustomerID"] = df["CustomerID"].astype("int64")
    logger.info("v1 cleaning checkpoints: %s", counts)
    return df.reset_index(drop=True), counts


def classify_lines(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Tag each row as sale, cancellation, or non_product. Does not drop rows."""
    clean = cfg["cleaning"]
    codes = {c.upper() for c in clean["non_product_stock_codes"]}
    gift_re = re.compile(clean["gift_voucher_pattern"], re.IGNORECASE)
    prefix = clean["invoice_cancel_prefix"]

    out = df.copy()
    stock = out["StockCode"].astype(str).str.strip()
    invoice = out["InvoiceNo"].astype(str)
    is_non_product = stock.str.upper().isin(codes) | stock.str.match(gift_re)
    is_cancel = invoice.str.startswith(prefix) | (out["Quantity"] < 0)

    line_type = np.where(
        is_non_product, "non_product", np.where(is_cancel, "cancellation", "sale")
    )
    out["StockCode"] = stock
    out["line_type"] = pd.Series(line_type, index=out.index)
    out["gross_revenue"] = out["Quantity"] * out["UnitPrice"]
    logger.info(
        "line types: sale=%s cancellation=%s non_product=%s",
        int((out["line_type"] == "sale").sum()),
        int((out["line_type"] == "cancellation").sum()),
        int((out["line_type"] == "non_product").sum()),
    )
    return out


def atypical_stock_codes(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """StockCodes that do not match the 5-digit product pattern (review list, not exclusions)."""
    pattern = re.compile(cfg["cleaning"]["product_stockcode_pattern"])
    stock = df["StockCode"].astype(str).str.strip()
    flagged = df.loc[~stock.str.match(pattern)].copy()
    flagged["StockCode"] = stock.loc[flagged.index]
    review = (
        flagged.groupby("StockCode", dropna=False)
        .agg(
            n_lines=("InvoiceNo", "size"),
            n_customers=("CustomerID", lambda s: int(s.dropna().nunique())),
            examples=(
                "Description",
                lambda s: "; ".join(sorted(s.dropna().astype(str).unique())[:3]),
            ),
        )
        .reset_index()
        .sort_values(["n_lines", "StockCode"], ascending=[False, True])
    )
    return review


def flag_partial_months(
    df: pd.DataFrame, coverage_threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add invoice month / date and is_partial_month using calendar-day coverage."""
    out = df.copy()
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"])
    out["Date"] = out["InvoiceDate"].dt.normalize()
    out["month_start"] = out["Date"].dt.to_period("M").dt.to_timestamp()

    data_min = out["Date"].min()
    data_max = out["Date"].max()
    coverage_rows = []
    partial_months: set[pd.Timestamp] = set()
    for month_start, _grp in out.groupby("month_start", sort=True):
        month_start = pd.Timestamp(month_start)
        days_in_month = int(pd.Period(month_start, freq="M").days_in_month)
        cal_end = month_start + pd.offsets.MonthEnd(0)
        # Days of this calendar month that fall inside the extract window.
        available_start = max(month_start, data_min)
        available_end = min(cal_end.normalize(), data_max)
        days_available = int((available_end - available_start).days + 1)
        coverage = days_available / days_in_month
        is_partial = bool(coverage < coverage_threshold)
        if is_partial:
            partial_months.add(month_start)
        coverage_rows.append(
            {
                "month_start": month_start,
                "days_present": days_available,
                "days_in_month": days_in_month,
                "coverage": coverage,
                "is_partial_month": is_partial,
            }
        )
    coverage_df = pd.DataFrame(coverage_rows)
    out["is_partial_month"] = out["month_start"].isin(partial_months)
    logger.info(
        "partial months: %s", [d.strftime("%Y-%m") for d in sorted(partial_months)]
    )
    return out, coverage_df
