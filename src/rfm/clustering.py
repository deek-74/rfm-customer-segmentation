"""Log-transformed K-means with stability-aware k selection and profile-based names."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from rfm.segments import conditions_match

logger = logging.getLogger(__name__)


def _feature_matrix(customers: pd.DataFrame, cfg: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    cluster_cfg = cfg["clustering"]
    cols = list(cluster_cfg["features"])
    if cluster_cfg.get("use_avg_order_value"):
        cols.append(cluster_cfg["optional_aov_feature"])
    frame = customers.loc[:, cols].astype(float)
    if cluster_cfg.get("log1p", True):
        frame = np.log1p(frame.clip(lower=0))
    return frame.to_numpy(), cols


def _fit(X: np.ndarray, k: int, seed: int, n_init: int, max_iter: int) -> tuple[np.ndarray, float]:
    model = KMeans(
        n_clusters=k,
        n_init=n_init,
        random_state=seed,
        max_iter=max_iter,
        algorithm="lloyd",
    )
    labels = model.fit_predict(X)
    return labels, float(model.inertia_)


def evaluate_k_range(customers: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    cluster_cfg = cfg["clustering"]
    X_raw, _ = _feature_matrix(customers, cfg)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    seeds = list(cfg["seeds"]["stability"])
    n_init = int(cluster_cfg["n_init"])
    max_iter = int(cluster_cfg["max_iter"])
    rows = []
    for k in range(int(cluster_cfg["k_min"]), int(cluster_cfg["k_max"]) + 1):
        labels_main, inertia = _fit(X, k, int(cfg["seeds"]["kmeans"]), n_init, max_iter)
        counts = np.bincount(labels_main, minlength=k)
        min_share = float(counts.min() / len(labels_main))
        sil = float(silhouette_score(X, labels_main)) if k > 1 else float("nan")
        db = float(davies_bouldin_score(X, labels_main)) if k > 1 else float("nan")
        label_runs = [_fit(X, k, seed, n_init, max_iter)[0] for seed in seeds]
        aris = [
            adjusted_rand_score(label_runs[i], label_runs[j])
            for i in range(len(label_runs))
            for j in range(i + 1, len(label_runs))
        ]
        mean_ari = float(np.mean(aris)) if aris else float("nan")
        rows.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette": sil,
                "davies_bouldin": db,
                "smallest_cluster_share": min_share,
                "stability_ari": mean_ari,
                "n_customers": len(labels_main),
            }
        )
        logger.info(
            "k=%s silhouette=%.3f min_share=%.3f ARI=%.3f",
            k, sil, min_share, mean_ari,
        )
    return pd.DataFrame(rows)


def select_k(metrics: pd.DataFrame, cfg: dict[str, Any]) -> tuple[int, str]:
    cluster_cfg = cfg["clustering"]
    lo = int(cluster_cfg["candidate_k_min"])
    hi = int(cluster_cfg["candidate_k_max"])
    min_share = float(cluster_cfg["min_cluster_share"])
    min_ari = float(cluster_cfg["min_stability_ari"])
    tie = float(cluster_cfg["silhouette_tie_delta"])

    qualified = metrics.loc[
        (metrics["k"] >= lo)
        & (metrics["k"] <= hi)
        & (metrics["smallest_cluster_share"] >= min_share)
        & (metrics["stability_ari"] >= min_ari)
    ]
    reason_parts = []
    if not qualified.empty:
        best_sil = qualified["silhouette"].max()
        close = qualified.loc[qualified["silhouette"] >= best_sil - tie]
        chosen = int(close.sort_values(["k", "silhouette"]).iloc[0]["k"])
        reason_parts.append(
            f"k={chosen} qualifies (k {lo}-{hi}, smallest cluster >= {min_share:.0%}, "
            f"stability ARI >= {min_ari:.2f}) with silhouette "
            f"{float(metrics.loc[metrics['k'] == chosen, 'silhouette'].iloc[0]):.3f}."
        )
        if len(close) > 1:
            reason_parts.append(
                f"Silhouettes within {tie:.2f} of the best; fewer clusters preferred."
            )
        return chosen, " ".join(reason_parts)

    fallback = metrics.loc[
        (metrics["k"] >= lo)
        & (metrics["k"] <= hi)
        & (metrics["smallest_cluster_share"] >= min_share)
    ]
    pool = fallback if not fallback.empty else metrics
    chosen = int(pool.sort_values("silhouette", ascending=False).iloc[0]["k"])
    reason_parts.append(
        f"No k in {lo}-{hi} met both min cluster share {min_share:.0%} and ARI {min_ari:.2f}. "
        f"Selected k={chosen} as the best available by silhouette among the fallback pool."
    )
    return chosen, " ".join(reason_parts)


def fit_final(customers: pd.DataFrame, cfg: dict[str, Any], k: int) -> tuple[np.ndarray, StandardScaler]:
    cluster_cfg = cfg["clustering"]
    X_raw, _ = _feature_matrix(customers, cfg)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    labels, _inertia = _fit(
        X,
        k,
        int(cfg["seeds"]["kmeans"]),
        int(cluster_cfg["n_init"]),
        int(cluster_cfg["max_iter"]),
    )
    return labels, scaler


def name_clusters(profile: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Name from median R/F/M scores using ordered config rules. Never maps on cluster id."""
    rules = cfg["clustering"]["naming_rules"]
    names: list[str] = []
    for _, row in profile.iterrows():
        record = {
            "r_score_median": float(row["r_score_median"]),
            "f_score_median": float(row["f_score_median"]),
            "m_score_median": float(row["m_score_median"]),
        }
        hit = None
        for rule in rules:
            if conditions_match(record, rule["conditions"]):
                hit = rule["name"]
                break
        if hit is None:
            raise ValueError(f"Cluster {row.get('cluster')} matched no naming rule.")
        names.append(hit)
    out = profile.copy()
    out["cluster_name_base"] = names
    # Deduplicate identical names with a spend qualifier (higher median monetary first).
    out["cluster_name"] = out["cluster_name_base"]
    for base, grp in out.groupby("cluster_name_base"):
        if len(grp) == 1:
            continue
        ranked = grp.sort_values("monetary_median", ascending=False)
        for i, idx in enumerate(ranked.index):
            qualifier = "Higher Spend" if i == 0 else "Lower Spend" if i == len(ranked) - 1 else f"Spend tier {i + 1}"
            out.loc[idx, "cluster_name"] = f"{base} ({qualifier})"
    return out


def build_cluster_profile(customers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(customers)
    total_rev = float(customers["monetary"].sum())
    for cluster, grp in customers.groupby("cluster"):
        rows.append(
            {
                "cluster": int(cluster),
                "customers": int(len(grp)),
                "share": len(grp) / n,
                "recency_days_median": float(grp["recency_days"].median()),
                "recency_days_mean": float(grp["recency_days"].mean()),
                "frequency_median": float(grp["frequency"].median()),
                "frequency_mean": float(grp["frequency"].mean()),
                "monetary_median": float(grp["monetary"].median()),
                "monetary_mean": float(grp["monetary"].mean()),
                "r_score_median": float(grp["r_score"].median()),
                "f_score_median": float(grp["f_score"].median()),
                "m_score_median": float(grp["m_score"].median()),
                "revenue": float(grp["monetary"].sum()),
                "revenue_share": float(grp["monetary"].sum() / total_rev) if total_rev else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("cluster")


def sensitivity_check(
    customers: pd.DataFrame, cfg: dict[str, Any], k: int, labels: np.ndarray
) -> dict[str, Any]:
    """Refit without the top monetary percentile; compare structure and names."""
    top_pct = float(cfg["clustering"]["sensitivity_top_pct"])
    cutoff = customers["monetary"].quantile(1.0 - top_pct)
    keep = customers["monetary"] < cutoff
    subset = customers.loc[keep].reset_index(drop=True)
    labels_orig = labels[keep.to_numpy()]
    labels_new, _ = fit_final(subset, cfg, k)
    ari = float(adjusted_rand_score(labels_orig, labels_new))
    tmp = subset.copy()
    tmp["cluster"] = labels_new
    prof = name_clusters(build_cluster_profile(tmp), cfg)
    names_new = sorted(prof["cluster_name_base"].unique().tolist())
    tmp_orig = customers.copy()
    tmp_orig["cluster"] = labels
    prof_orig = name_clusters(build_cluster_profile(tmp_orig), cfg)
    names_orig = sorted(prof_orig["cluster_name_base"].unique().tolist())
    return {
        "excluded_customers": int((~keep).sum()),
        "monetary_cutoff": float(cutoff),
        "ari_vs_original_on_overlap": ari,
        "names_original": names_orig,
        "names_refit": names_new,
        "names_hold": names_orig == names_new,
        "structure_holds": ari >= 0.5,
    }
