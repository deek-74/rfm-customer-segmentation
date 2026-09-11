"""Score and segment customers at two snapshots; build a transition matrix."""

from __future__ import annotations

from typing import Any

import pandas as pd

from rfm.features import build_customer_features
from rfm.scoring import apply_edges, fit_score_edges, frequency_band_score
from rfm.segments import assign_segments_v2


def _score_with_edges(
    customers: pd.DataFrame,
    cfg: dict[str, Any],
    recency_edges: list[float],
    monetary_edges: list[float],
) -> pd.DataFrame:
    out = customers.copy()
    # Recency edges are fitted on -recency so higher (less negative) is better.
    out["r_score"] = apply_edges(-out["recency_days"], recency_edges, higher_is_better=True)
    out["f_score"] = frequency_band_score(out["frequency"], cfg["scoring"]["frequency_bands"])
    out["m_score"] = apply_edges(out["monetary"], monetary_edges, higher_is_better=True)
    out["rfm_code"] = out["r_score"] * 100 + out["f_score"] * 10 + out["m_score"]
    return assign_segments_v2(out, cfg)


def run_migration(
    fact: pd.DataFrame,
    dim_b: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Snapshot A vs end-of-period B using B's R/M edges and fixed F bands."""
    snapshot_a = pd.Timestamp(cfg["migration"]["snapshot_a"])
    fact_a = fact.loc[pd.to_datetime(fact["Date"]) < snapshot_a].copy()
    dim_a = build_customer_features(fact_a, cfg, snapshot=snapshot_a)

    recency_edges = fit_score_edges(-dim_b["recency_days"], dim_b["r_score"])
    monetary_edges = fit_score_edges(dim_b["monetary"], dim_b["m_score"])

    dim_a = _score_with_edges(dim_a, cfg, recency_edges, monetary_edges)

    label_new = cfg["migration"]["not_yet_acquired_label"]
    b = dim_b.loc[:, ["CustomerID", "segment", "r_score", "f_score", "m_score"]].rename(
        columns=lambda c: c if c == "CustomerID" else f"{c}_b"
    )
    a = dim_a.loc[:, ["CustomerID", "segment", "r_score", "f_score", "m_score"]].rename(
        columns=lambda c: c if c == "CustomerID" else f"{c}_a"
    )
    merged = b.merge(a, on="CustomerID", how="left")
    merged["segment_a"] = merged["segment_a"].fillna(label_new)

    counts = (
        merged.groupby(["segment_a", "segment_b"], as_index=False)
        .size()
        .rename(columns={"size": "customers"})
    )
    row_tot = counts.groupby("segment_a")["customers"].transform("sum")
    counts["row_pct"] = counts["customers"] / row_tot

    a_counts = a.groupby("segment_a", as_index=False).size().rename(columns={"size": "n_at_a"})

    champions = cfg["migration"]["champion_segment"]
    at_risk = cfg["migration"]["at_risk_segment"]
    reactivated = set(cfg["migration"]["reactivated_segments"])

    june_champs = merged.loc[merged["segment_a"] == champions]
    still_champs = (
        float((june_champs["segment_b"] == champions).mean()) if len(june_champs) else float("nan")
    )
    june_risk = merged.loc[merged["segment_a"] == at_risk]
    reactivated_share = (
        float(june_risk["segment_b"].isin(reactivated).mean()) if len(june_risk) else float("nan")
    )

    return {
        "dim_a": dim_a,
        "transition": counts,
        "snapshot_a_counts": a_counts,
        "n_not_yet_acquired": int((merged["segment_a"] == label_new).sum()),
        "pct_champions_still_champions": still_champs,
        "pct_at_risk_reactivated": reactivated_share,
        "recency_edges": recency_edges,
        "monetary_edges": monetary_edges,
        "n_june_champions": int(len(june_champs)),
        "n_june_at_risk": int(len(june_risk)),
    }
