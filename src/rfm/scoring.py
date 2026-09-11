"""v1 dplyr::ntile replica and v2 tie-safe RFM scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def dplyr_ntile(values: np.ndarray | pd.Series, n: int = 5) -> np.ndarray:
    """Replicate dplyr::ntile: rank(method='first'), floor(n * (rank-1) / N) + 1.

    Ties are broken by original row order, matching R's rank(ties.method='first').
    """
    arr = np.asarray(values)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.int64)
    ranks[order] = np.arange(1, len(arr) + 1)
    return (np.floor(n * (ranks - 1) / len(arr)) + 1).astype(int)


def score_v1(customers: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Score customers in first-appearance order, matching reference/script.R."""
    out = customers.copy()
    out["R_Score"] = dplyr_ntile(-out["Recency"].to_numpy(), n_bins)
    out["F_Score"] = dplyr_ntile(out["Frequency"].to_numpy(), n_bins)
    out["M_Score"] = dplyr_ntile(out["Monetary"].to_numpy(), n_bins)
    out["RFM_Score"] = out["R_Score"] * 100 + out["F_Score"] * 10 + out["M_Score"]
    return out


def _pct_quintile_score(values: pd.Series, n: int, higher_is_better: bool) -> pd.Series:
    ranked = values.rank(method="min", pct=True, ascending=higher_is_better)
    scores = np.ceil(ranked.to_numpy() * n)
    return pd.Series(np.clip(scores, 1, n).astype(int), index=values.index)


def frequency_band_score(frequency: pd.Series, bands: list[dict[str, Any]]) -> pd.Series:
    """Map raw order counts to scores using inclusive max_frequency cut-offs."""
    scores = np.full(len(frequency), np.nan)
    remaining = np.ones(len(frequency), dtype=bool)
    values = frequency.to_numpy()
    for band in bands:
        cap = band["max_frequency"]
        score = int(band["score"])
        if cap is None:
            mask = remaining
        else:
            mask = remaining & (values <= cap)
        scores[mask] = score
        remaining = remaining & ~mask
    if remaining.any():
        raise ValueError("Frequency band config does not cover every customer.")
    return pd.Series(scores.astype(int), index=frequency.index)


def score_v2(customers: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Tie-safe R/M quintiles and fixed Frequency bands. Equal values → equal scores."""
    scoring = cfg["scoring"]
    n = int(scoring["n_quintiles"])
    out = customers.copy()
    out["r_score"] = _pct_quintile_score(out["recency_days"], n, higher_is_better=False)
    out["f_score"] = frequency_band_score(out["frequency"], scoring["frequency_bands"])
    out["m_score"] = _pct_quintile_score(out["monetary"], n, higher_is_better=True)
    out["rfm_code"] = out["r_score"] * 100 + out["f_score"] * 10 + out["m_score"]
    return out


def fit_score_edges(values: pd.Series, scores: pd.Series) -> list[float]:
    """Midpoints between adjacent score groups, used to score snapshot A on B's scale."""
    edges: list[float] = []
    ordered = pd.DataFrame({"v": values, "s": scores}).dropna()
    for s in range(int(scores.min()), int(scores.max())):
        left = ordered.loc[ordered["s"] == s, "v"]
        right = ordered.loc[ordered["s"] == s + 1, "v"]
        if left.empty or right.empty:
            continue
        edges.append(float((left.max() + right.min()) / 2.0))
    return edges


def apply_edges(values: pd.Series, edges: list[float], higher_is_better: bool) -> pd.Series:
    """Assign 1..len(edges)+1 using cutpoints. Recency uses higher_is_better=False."""
    arr = values.to_numpy()
    edges_arr = np.asarray(edges, dtype=float)
    if higher_is_better:
        idx = np.searchsorted(edges_arr, arr, side="right")
        scores = idx + 1
    else:
        idx = np.searchsorted(edges_arr, arr, side="right")
        scores = (len(edges_arr) + 1) - idx
    n = len(edges_arr) + 1
    return pd.Series(np.clip(scores, 1, n).astype(int), index=values.index)
