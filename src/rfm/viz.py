"""Matplotlib / Plotly figures. Titles state the insight; numbers come from tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _style(cfg: dict[str, Any]) -> None:
    viz = cfg["viz"]
    try:
        plt.style.use(viz["style"])
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.size": viz["font_size"], "axes.titlesize": viz["title_size"]})


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Failed to write figure {path}")


def _seg_colours(cfg: dict[str, Any]) -> dict[str, str]:
    return {s["name"]: s["colour"] for s in cfg["segments"]}


def render_all_figures(
    *,
    cfg: dict[str, Any],
    monthly: pd.DataFrame,
    dim_customer: pd.DataFrame,
    segment_summary: pd.DataFrame,
    k_metrics: pd.DataFrame,
    chosen_k: int,
    cluster_profile: pd.DataFrame,
    cluster_crosstab: pd.DataFrame,
    retention: pd.DataFrame,
    migration: pd.DataFrame,
    winback: pd.DataFrame,
    concentration: dict[str, Any],
) -> list[str]:
    _style(cfg)
    dpi = int(cfg["viz"]["dpi"])
    fig_dir: Path = cfg["paths"]["figures_dir"]
    colours = _seg_colours(cfg)
    written: list[str] = []

    written.append(_monthly_revenue(monthly, fig_dir / "01_monthly_revenue.png", dpi, cfg))
    written.append(_pareto(dim_customer, concentration, fig_dir / "02_revenue_concentration.png", dpi, cfg))
    written.append(_frequency(dim_customer, fig_dir / "03_frequency_distribution.png", dpi))
    written.append(_rf_heatmap(dim_customer, fig_dir / "04_rf_heatmap_avg_monetary.png", dpi, cfg))
    written.append(_share_vs_revenue(segment_summary, colours, fig_dir / "05_segment_share_vs_revenue.png", dpi, cfg))
    written.append(_segment_profiles(segment_summary, colours, fig_dir / "06_segment_profiles.png", dpi, cfg))
    written.append(_k_selection(k_metrics, chosen_k, fig_dir / "07_k_selection.png", dpi, cfg))
    written.append(_cluster_profiles(cluster_profile, fig_dir / "08_cluster_profiles.png", dpi, cfg))
    written.append(_cluster_vs_segment(cluster_crosstab, fig_dir / "09_cluster_vs_segment.png", dpi))
    written.append(_cohort(retention, fig_dir / "10_cohort_retention.png", dpi, cfg))
    written.append(_migration_heatmap(migration, fig_dir / "11_segment_migration.png", dpi, cfg))
    _migration_sankey(migration, colours, fig_dir / "11_segment_migration.html")
    written.append("11_segment_migration.html")
    written.append(_winback(winback, colours, fig_dir / "12_winback_scenarios.png", dpi, cfg))
    return written


def _monthly_revenue(monthly: pd.DataFrame, path: Path, dpi: int, cfg: dict[str, Any]) -> str:
    df = monthly.copy()
    df["month_start"] = pd.to_datetime(df["month_start"])
    complete = df.loc[~df["is_partial_month"]]
    partial = df.loc[df["is_partial_month"]]
    fig, ax = plt.subplots(figsize=tuple(cfg["viz"]["figsize"]))
    if complete.empty:
        ax.plot(df["month_start"], df["net_revenue"], color="#1B4F72", marker="o", lw=2)
        ax.set_title("Monthly historical net revenue")
    else:
        peak = complete.loc[complete["net_revenue"].idxmax()]
        ax.plot(complete["month_start"], complete["net_revenue"], color="#1B4F72", marker="o", lw=2)
        ax.set_title(
            f"Revenue peaked in {pd.Timestamp(peak['month_start']).strftime('%b %Y')}; "
            "Dec 2011 is a partial month, not a collapse"
        )
    if not partial.empty:
        ax.scatter(
            partial["month_start"],
            partial["net_revenue"],
            color="#CA6F1E",
            s=80,
            zorder=5,
            label="Partial month (excluded from trend)",
        )
        for _, row in partial.iterrows():
            ax.annotate(
                "Partial month — not used in growth rates",
                xy=(row["month_start"], row["net_revenue"]),
                xytext=(10, 20),
                textcoords="offset points",
                color="#CA6F1E",
                fontsize=9,
            )
    ax.set_ylabel(f"Historical net revenue ({cfg['project']['currency_symbol']})")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    _save(fig, path, dpi)
    return path.name


def _pareto(
    dim_customer: pd.DataFrame,
    concentration: dict[str, Any],
    path: Path,
    dpi: int,
    cfg: dict[str, Any],
) -> str:
    n = len(dim_customer)
    cum = np.asarray(concentration["pareto_cum_share"])
    x = np.arange(1, n + 1) / n * 100
    top20 = concentration.get("top_20_pct_revenue_share", cum[max(1, int(np.ceil(0.2 * n))) - 1])
    fig, ax = plt.subplots(figsize=tuple(cfg["viz"]["figsize"]))
    ax.plot(x, cum * 100, color="#1B4F72", lw=2)
    ax.axvline(20, color="#CA6F1E", ls="--", lw=1)
    ax.set_xlabel("% of customers (highest spenders first)")
    ax.set_ylabel("% of historical net revenue")
    ax.set_title(f"Top 20% of customers generate {top20 * 100:.0f}% of historical net revenue")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    _save(fig, path, dpi)
    return path.name


def _frequency(dim_customer: pd.DataFrame, path: Path, dpi: int) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    freq = dim_customer["frequency"].clip(upper=20)
    ax.hist(freq, bins=np.arange(1, 22) - 0.5, color="#1E8449", edgecolor="white")
    one = int((dim_customer["frequency"] == 1).sum())
    share = one / len(dim_customer)
    ax.set_title(f"{share * 100:.0f}% of customers placed a single countable order (capped at 20)")
    ax.set_xlabel("Countable orders per customer")
    ax.set_ylabel("Customers")
    _save(fig, path, dpi)
    return path.name


def _rf_heatmap(dim_customer: pd.DataFrame, path: Path, dpi: int, cfg: dict[str, Any]) -> str:
    pivot = dim_customer.pivot_table(
        index="r_score", columns="f_score", values="monetary", aggfunc="mean"
    )
    pivot = pivot.reindex(index=[5, 4, 3, 2, 1], columns=[1, 2, 3, 4, 5])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(5), labels=[1, 2, 3, 4, 5])
    ax.set_yticks(range(5), labels=[5, 4, 3, 2, 1])
    ax.set_xlabel("Frequency score")
    ax.set_ylabel("Recency score (5 = most recent)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.to_numpy()[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{cfg['project']['currency_symbol']}{val:,.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean historical net revenue")
    ax.set_title("High recency and frequency cells hold the highest average historical spend")
    _save(fig, path, dpi)
    return path.name


def _share_vs_revenue(
    summary: pd.DataFrame, colours: dict[str, str], path: Path, dpi: int, cfg: dict[str, Any]
) -> str:
    df = summary.sort_values("sort_order")
    top = df.sort_values("pct_revenue", ascending=False).iloc[0]
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, df["pct_customers"] * 100, width, color=[colours[s] for s in df["segment"]], alpha=0.9, label="% customers")
    ax.bar(x + width / 2, df["pct_revenue"] * 100, width, color=[colours[s] for s in df["segment"]], alpha=0.45, hatch="//", label="% historical net revenue")
    ax.set_xticks(x, df["segment"], rotation=35, ha="right")
    ax.set_ylabel("Share (%)")
    ax.set_title(
        f"{top['segment']} are {top['pct_customers'] * 100:.0f}% of customers but "
        f"{top['pct_revenue'] * 100:.0f}% of historical net revenue"
    )
    ax.legend()
    _save(fig, path, dpi)
    return path.name


def _segment_profiles(
    summary: pd.DataFrame, colours: dict[str, str], path: Path, dpi: int, cfg: dict[str, Any]
) -> str:
    df = summary.sort_values("sort_order")
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    metrics = [
        ("mean_recency", "Mean recency (days)", False),
        ("mean_frequency", "Mean frequency (orders)", True),
        ("mean_monetary", f"Mean historical net ({cfg['project']['currency_symbol']})", True),
    ]
    for ax, (col, title, _higher) in zip(axes, metrics, strict=True):
        ax.barh(df["segment"], df[col], color=[colours[s] for s in df["segment"]])
        ax.set_title(title)
        ax.invert_yaxis()
    fig.suptitle("Segment profiles: recency, order count and historical net spend")
    _save(fig, path, dpi)
    return path.name


def _k_selection(metrics: pd.DataFrame, chosen_k: int, path: Path, dpi: int, cfg: dict[str, Any]) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(metrics["k"], metrics["silhouette"], marker="o", color="#1B4F72")
    axes[0].axvline(chosen_k, color="#CA6F1E", ls="--")
    axes[0].set_title("Silhouette (higher better)")
    axes[1].plot(metrics["k"], metrics["smallest_cluster_share"] * 100, marker="o", color="#1E8449")
    axes[1].axhline(5, color="#7D3C98", ls=":")
    axes[1].axvline(chosen_k, color="#CA6F1E", ls="--")
    axes[1].set_title("Smallest cluster share (%)")
    axes[2].plot(metrics["k"], metrics["stability_ari"], marker="o", color="#7D3C98")
    axes[2].axhline(0.8, color="#7D3C98", ls=":")
    axes[2].axvline(chosen_k, color="#CA6F1E", ls="--")
    axes[2].set_title("Stability ARI")
    for ax in axes:
        ax.set_xlabel("k")
        ax.set_xticks(list(metrics["k"]))
    fig.suptitle(f"k={chosen_k} selected under size and stability constraints (not max silhouette alone)")
    _save(fig, path, dpi)
    return path.name


def _cluster_profiles(profile: pd.DataFrame, path: Path, dpi: int, cfg: dict[str, Any]) -> str:
    df = profile.sort_values("cluster")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["cluster_name"], df["share"] * 100, color="#1B4F72", alpha=0.85, label="% customers")
    ax.barh(df["cluster_name"], df["revenue_share"] * 100, color="#CA6F1E", alpha=0.45, label="% historical net revenue")
    ax.set_xlabel("Share (%)")
    ax.set_title("Unsupervised clusters named from median RFM profiles, not from cluster ids")
    ax.legend()
    _save(fig, path, dpi)
    return path.name


def _cluster_vs_segment(crosstab: pd.DataFrame, path: Path, dpi: int) -> str:
    mat = crosstab.pivot_table(index="cluster_name", columns="segment", values="customers", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(mat.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(mat.shape[1]), labels=list(mat.columns), rotation=35, ha="right")
    ax.set_yticks(range(mat.shape[0]), labels=list(mat.index))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, int(mat.to_numpy()[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Customers")
    ax.set_title("Rule-based segments vs unsupervised clusters (they answer different questions)")
    _save(fig, path, dpi)
    return path.name


def _cohort(retention: pd.DataFrame, path: Path, dpi: int, cfg: dict[str, Any]) -> str:
    if retention.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.set_title("Cohort retention (no complete-month cohorts)")
        _save(fig, path, dpi)
        return path.name
    pivot = retention.pivot_table(
        index="cohort_month", columns="months_since", values="customer_retention", aggfunc="mean"
    )
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(pivot.to_numpy() * 100, cmap="YlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(pivot.shape[1]), labels=list(pivot.columns))
    ax.set_yticks(
        range(pivot.shape[0]),
        labels=[pd.Timestamp(i).strftime("%b %Y") for i in pivot.index],
    )
    ax.set_xlabel("Months since first purchase")
    ax.set_ylabel("Acquisition cohort")
    fig.colorbar(im, ax=ax, label="Customer retention %")
    ax.set_title("Month-0 is 100% by definition; Dec 2010 is left-censored (pre-period buyers mixed in)")
    _save(fig, path, dpi)
    return path.name


def _migration_heatmap(migration: pd.DataFrame, path: Path, dpi: int, cfg: dict[str, Any]) -> str:
    mat = migration.pivot_table(index="segment_a", columns="segment_b", values="row_pct", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(mat.to_numpy() * 100, cmap="Oranges", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(mat.shape[1]), labels=list(mat.columns), rotation=35, ha="right")
    ax.set_yticks(range(mat.shape[0]), labels=list(mat.index))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.to_numpy()[i, j] * 100:.0f}%", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Row % of snapshot-A segment")
    ax.set_xlabel("Segment at period end")
    ax.set_ylabel("Segment at snapshot A")
    ax.set_title("Where June customers sat by December (rows sum to 100%)")
    _save(fig, path, dpi)
    return path.name


def _migration_sankey(migration: pd.DataFrame, colours: dict[str, str], path: Path) -> None:
    nodes = sorted(set(migration["segment_a"]) | set(migration["segment_b"]))
    index = {n: i for i, n in enumerate(nodes)}
    node_colours = [colours.get(n, "#95A5A6") for n in nodes]
    fig = go.Figure(
        go.Sankey(
            node=dict(label=nodes, color=node_colours, pad=12, thickness=14),
            link=dict(
                source=[index[s] for s in migration["segment_a"]],
                target=[index[s] for s in migration["segment_b"]],
                value=migration["customers"],
            ),
        )
    )
    fig.update_layout(title="Segment migration Sankey (snapshot A → end)", font_size=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")


def _winback(winback: pd.DataFrame, colours: dict[str, str], path: Path, dpi: int, cfg: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    rates = sorted(winback["reactivation_rate"].unique())
    segments = list(winback["segment"].unique())
    x = np.arange(len(rates))
    width = 0.35 if len(segments) > 1 else 0.5
    for i, segment in enumerate(segments):
        sub = winback.loc[winback["segment"] == segment].sort_values("reactivation_rate")
        ax.bar(
            x + (i - (len(segments) - 1) / 2) * width,
            sub["expected_winback_revenue"],
            width,
            label=segment,
            color=colours.get(segment, "#1B4F72"),
        )
    ax.set_xticks(x, [f"{r:.0%}" for r in rates])
    ax.set_xlabel("Assumed reactivation rate")
    ax.set_ylabel(f"Scenario expected historical-AOV revenue ({cfg['project']['currency_symbol']})")
    ax.set_title("Win-back is a scenario, not a forecast — expected revenue at 5%, 10% and 20% reactivation")
    ax.legend()
    _save(fig, path, dpi)
    return path.name
