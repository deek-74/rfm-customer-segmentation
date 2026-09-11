"""End-to-end v1 parity + v2 pipeline. All tunables come from config."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from rfm.cancellations import net_cancellations
from rfm.cleaning import atypical_stock_codes, classify_lines, clean_v1, flag_partial_months
from rfm.clustering import (
    build_cluster_profile,
    evaluate_k_range,
    fit_final,
    name_clusters,
    select_k,
    sensitivity_check,
)
from rfm.cohorts import build_retention_matrices
from rfm.features import (
    build_customer_features,
    cohort_activity,
    customer_metrics_v1,
    monthly_kpis,
    product_summary,
    snapshot_date_from_fact,
    stage_fact,
)
from rfm.impact import concentration, experiment_design, revenue_at_risk, winback_scenarios
from rfm.io import read_csv, read_parquet, write_csv, write_json, write_parquet, write_text
from rfm.migration import run_migration
from rfm.powerbi_export import export_powerbi
from rfm.scoring import score_v1, score_v2
from rfm.segments import assign_segments_v1, assign_segments_v2, rf_cell_segments, segment_lookup
from rfm.viz import render_all_figures

logger = logging.getLogger(__name__)


def run_v1_parity(cfg: dict[str, Any], raw: pd.DataFrame) -> dict[str, Any]:
    """Reproduce reference/script.R and write outputs/v1_parity_report.md."""
    v1_cfg = cfg["v1"]
    clean, counts = clean_v1(raw)
    customers, ref_date = customer_metrics_v1(clean)
    customers = score_v1(customers, int(v1_cfg["ntile_bins"]))
    customers = assign_segments_v1(customers)

    n_customers = int(len(customers))
    n_invoices = int(clean["InvoiceNo"].nunique())
    revenue = float(clean["Revenue"].sum())
    aov = revenue / n_invoices if n_invoices else float("nan")
    repeat_rate = float((customers["Frequency"] > 1).mean() * 100)

    our_counts = customers["Segment"].value_counts().to_dict()
    ref_seg = read_csv(cfg["paths"]["v1_segment_summary"])
    ref_counts = dict(zip(ref_seg["Segment"], ref_seg["Count"], strict=True))
    count_deltas = {
        name: abs(int(our_counts.get(name, 0)) - int(ref_counts.get(name, 0)))
        for name in set(our_counts) | set(ref_counts)
    }

    ref_cust = read_csv(cfg["paths"]["v1_customer_table"])
    ref_cust["CustomerID"] = ref_cust["CustomerID"].astype(int)
    merged = customers.merge(
        ref_cust[["CustomerID", "Segment"]].rename(columns={"Segment": "Segment_ref"}),
        on="CustomerID",
        how="inner",
    )
    match_rate = float((merged["Segment"] == merged["Segment_ref"]).mean()) if len(merged) else 0.0

    checks = {
        "after_dedupe": counts["after_dedupe"] == int(v1_cfg["rows_after_dedupe"]),
        "after_customer_id": counts["after_customer_id"] == int(v1_cfg["rows_after_customer_id"]),
        "after_cancellations": counts["after_cancellations"] == int(v1_cfg["rows_after_cancellations"]),
        "final_rows": counts["final"] == int(v1_cfg["rows_final"]),
        "n_customers": n_customers == int(v1_cfg["n_customers_expected"]),
        "n_invoices": n_invoices == int(v1_cfg["n_invoices_expected"]),
        "revenue": abs(revenue - float(v1_cfg["revenue_expected"])) <= float(v1_cfg["revenue_tolerance"]),
        "aov": abs(aov - float(v1_cfg["aov_expected"])) <= float(v1_cfg["aov_tolerance"]),
        "repeat_rate": abs(repeat_rate - float(v1_cfg["repeat_rate_pct_expected"]))
        <= float(v1_cfg["repeat_rate_tolerance_pp"]),
        "segment_counts": all(d <= int(v1_cfg["segment_count_tolerance"]) for d in count_deltas.values()),
        "segment_match": match_rate >= float(v1_cfg["segment_match_min"]),
    }
    passed = all(checks.values())

    report = _parity_markdown(
        counts, n_customers, n_invoices, revenue, aov, repeat_rate,
        our_counts, ref_counts, count_deltas, match_rate, checks, passed, ref_date,
        cfg,
    )
    write_text(report, cfg["paths"]["v1_parity_report"])
    write_csv(customers, cfg["paths"]["tables_dir"] / "v1_customers.csv")

    return {
        "counts": counts,
        "n_customers": n_customers,
        "n_invoices": n_invoices,
        "revenue": revenue,
        "aov": aov,
        "repeat_rate_pct": repeat_rate,
        "segment_counts": {k: int(v) for k, v in our_counts.items()},
        "match_rate": match_rate,
        "passed": passed,
        "customers": customers,
        "checks": checks,
    }


def run_v2(cfg: dict[str, Any], raw: pd.DataFrame, v1: dict[str, Any]) -> dict[str, Any]:
    tables = cfg["paths"]["tables_dir"]
    processed = cfg["paths"]["processed_dir"]
    tables.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    n_raw = int(len(raw))
    deduped = raw.drop_duplicates().copy()
    n_deduped = int(len(deduped))
    classified = classify_lines(deduped, cfg)
    review_codes = atypical_stock_codes(classified, cfg)
    netted, net_stats = net_cancellations(classified)
    netted, month_coverage = flag_partial_months(
        netted, float(cfg["cleaning"]["partial_month_day_coverage"])
    )

    fact = stage_fact(netted, cfg)
    write_parquet(fact, cfg["paths"]["interim_staged_parquet"])

    snapshot = snapshot_date_from_fact(fact)
    customers = build_customer_features(fact, cfg, snapshot=snapshot)
    customers = score_v2(customers, cfg)
    customers = assign_segments_v2(customers, cfg)

    lookup = segment_lookup(cfg)
    customers = customers.merge(
        lookup.rename(columns={"segment": "segment"}),
        on="segment",
        how="left",
    )

    k_metrics = evaluate_k_range(customers, cfg)
    chosen_k, k_reason = select_k(k_metrics, cfg)
    labels, _scaler = fit_final(customers, cfg, chosen_k)
    customers["cluster"] = labels
    cluster_profile = name_clusters(build_cluster_profile(customers), cfg)
    customers = customers.merge(
        cluster_profile[["cluster", "cluster_name"]], on="cluster", how="left"
    )
    sensitivity = sensitivity_check(customers, cfg, chosen_k, labels)

    activity = cohort_activity(fact, customers, cfg)
    retention, _ = build_retention_matrices(activity, customers)

    migration = run_migration(fact, customers, cfg)
    products = product_summary(fact, cfg)
    monthly = monthly_kpis(fact, cfg)

    shares = list(cfg["pareto"]["top_shares"])
    conc = concentration(customers, shares)
    conc_small = {k: v for k, v in conc.items() if k != "pareto_cum_share"}

    rar = revenue_at_risk(customers, cfg["winback"]["at_risk_segment_names"])
    winback = winback_scenarios(customers, cfg)
    n_at_risk = int((customers["segment"] == cfg["experiment"]["target_segment"]).sum())
    experiment = experiment_design(n_at_risk, cfg)

    summary = _segment_summary(customers, lookup)
    v1_v2 = _v1_v2_crosstab(v1["customers"], customers)
    cluster_xtab = (
        customers.groupby(["cluster_name", "segment"], as_index=False)
        .size()
        .rename(columns={"size": "customers"})
    )
    rf_grid = rf_cell_segments(cfg)

    excluded = _excluded_customers(fact, customers)
    dq = _data_quality_payload(
        cfg, raw, deduped, classified, netted, fact, customers, products,
        net_stats, month_coverage, review_codes, excluded, n_raw, n_deduped, v1,
    )

    dim_customer = customers.copy()
    write_csv(dim_customer, tables / "dim_customer.csv")
    write_csv(dim_customer, processed / "dim_customer.csv")
    write_csv(summary, tables / "segment_summary.csv")
    write_csv(summary, processed / "segment_summary.csv")
    write_csv(cluster_profile, tables / "cluster_profile.csv")
    write_csv(cluster_profile, processed / "cluster_profile.csv")
    write_csv(k_metrics, tables / "k_selection_metrics.csv")
    write_csv(k_metrics, processed / "k_selection_metrics.csv")
    write_csv(monthly, tables / "monthly_kpis.csv")
    write_csv(monthly, processed / "monthly_kpis.csv")
    write_csv(products, tables / "product_summary.csv")
    write_csv(products, processed / "product_summary.csv")
    write_csv(retention, tables / "cohort_retention.csv")
    write_csv(retention, processed / "cohort_retention.csv")
    write_csv(migration["transition"], tables / "segment_migration.csv")
    write_csv(migration["transition"], processed / "segment_migration.csv")
    write_csv(winback, tables / "winback_scenarios.csv")
    write_csv(winback, processed / "winback_scenarios.csv")
    write_csv(cluster_xtab, tables / "cluster_segment_crosstab.csv")
    write_csv(cluster_xtab, processed / "cluster_segment_crosstab.csv")
    write_csv(v1_v2, tables / "v1_v2_segment_crosstab.csv")
    write_csv(v1_v2, processed / "v1_v2_segment_crosstab.csv")
    write_csv(rf_grid, tables / "rf_segment_grid.csv")
    write_csv(rf_grid, processed / "rf_segment_grid.csv")
    write_csv(lookup, processed / "dim_segment.csv")
    write_csv(month_coverage, tables / "month_coverage.csv")
    write_csv(review_codes, tables / "atypical_stock_codes.csv")
    write_csv(excluded, tables / "excluded_customers.csv")

    render_all_figures(
        cfg=cfg,
        monthly=monthly,
        dim_customer=dim_customer,
        segment_summary=summary,
        k_metrics=k_metrics,
        chosen_k=chosen_k,
        cluster_profile=cluster_profile,
        cluster_crosstab=cluster_xtab,
        retention=retention,
        migration=migration["transition"],
        winback=winback,
        concentration=conc,
    )
    export_powerbi(
        cfg=cfg,
        fact=fact,
        dim_customer=dim_customer,
        product_summary=products,
        cluster_profile=cluster_profile,
        retention=retention,
        migration=migration["transition"],
        winback=winback,
        k_metrics=k_metrics,
        monthly=monthly,
    )

    identified_net = float(
        fact.loc[
            (fact["CustomerID"].notna()) & (fact["line_type"] == "sale"),
            "net_revenue",
        ].sum()
    )
    all_sales_net = float(fact.loc[fact["line_type"] == "sale", "net_revenue"].sum())
    guest_net = all_sales_net - identified_net
    metrics = {
        "v1": {
            "n_customers": v1["n_customers"],
            "n_invoices": v1["n_invoices"],
            "revenue": v1["revenue"],
            "aov": v1["aov"],
            "repeat_rate_pct": v1["repeat_rate_pct"],
            "segment_counts": v1["segment_counts"],
            "parity_passed": v1["passed"],
            "segment_match_rate": v1["match_rate"],
            "counts": v1["counts"],
        },
        "v2": {
            "n_customers": int(len(customers)),
            "n_orders": int(
                fact.loc[fact["is_countable_order"], "InvoiceNo"].nunique()
            ),
            "identified_net_revenue": identified_net,
            "identified_customers_net_revenue": float(customers["monetary"].sum()),
            "all_sales_net_revenue": all_sales_net,
            "guest_net_revenue": guest_net,
            "guest_share_of_positive_sales": (guest_net / all_sales_net) if all_sales_net else 0.0,
            "aov": float(customers["monetary"].sum() / customers["frequency"].sum()),
            "repeat_rate_pct": float((customers["frequency"] > 1).mean() * 100),
            "matched_cancel_value": net_stats["matched_cancel_value"],
            "unmatched_cancel_value": net_stats["unmatched_cancel_value"],
            "non_product_gross_value": float(
                classified.loc[classified["line_type"] == "non_product", "gross_revenue"].clip(lower=0).sum()
            ),
            "partial_months": [
                pd.Timestamp(m).strftime("%Y-%m")
                for m in month_coverage.loc[month_coverage["is_partial_month"], "month_start"]
            ],
            "n_excluded_customers": int(len(excluded)),
            "snapshot_date": snapshot.strftime("%Y-%m-%d"),
            "n_raw_rows": n_raw,
            "n_after_dedupe": n_deduped,
            "revenue_delta_vs_v1": identified_net - v1["revenue"],
        },
        "segments": summary.to_dict(orient="records"),
        "clustering": {
            "k": int(chosen_k),
            "reason": k_reason,
            "smallest_cluster_share": float(cluster_profile["share"].min()),
            "silhouette": float(k_metrics.loc[k_metrics["k"] == chosen_k, "silhouette"].iloc[0]),
            "stability_ari": float(k_metrics.loc[k_metrics["k"] == chosen_k, "stability_ari"].iloc[0]),
            "sensitivity": {k: v for k, v in sensitivity.items() if k != "names_original"},
            "names": cluster_profile["cluster_name"].tolist(),
            "k_metrics": k_metrics.to_dict(orient="records"),
            "v1_k2_silhouette_note": (
                "v1 reported silhouette 0.947 at k=2 because two tiny outlier clusters "
                "were isolated on raw skewed Monetary; a lower, balanced score is preferred."
            ),
        },
        "pareto": conc_small,
        "migration": {
            "snapshot_a": cfg["migration"]["snapshot_a"],
            "pct_champions_still_champions": migration["pct_champions_still_champions"],
            "pct_at_risk_reactivated": migration["pct_at_risk_reactivated"],
            "n_june_champions": migration["n_june_champions"],
            "n_june_at_risk": migration["n_june_at_risk"],
            "n_not_yet_acquired": migration["n_not_yet_acquired"],
        },
        "revenue_at_risk": rar,
        "experiment": experiment,
        "winback_total_at_10pct": float(
            winback.loc[np.isclose(winback["reactivation_rate"], 0.10), "expected_winback_revenue"].sum()
        ),
        "acceptance": _acceptance_flags(customers, products, fact, cfg, excluded),
    }
    write_json(metrics, cfg["paths"]["metrics_json"])
    write_json(metrics, processed / "metrics.json")
    write_text(_dq_markdown(dq, cfg), cfg["paths"]["data_quality_report"])
    write_csv(lookup, tables / "dim_segment.csv")

    logger.info("v2 customers=%s net=%.2f k=%s", metrics["v2"]["n_customers"], identified_net, chosen_k)
    return {"metrics": metrics, "dim_customer": dim_customer, "fact": fact, "dq": dq}


def run_pipeline(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = read_parquet(cfg["paths"]["interim_raw_parquet"])
    logger.info("Loaded raw parquet rows=%s cols=%s", len(raw), raw.shape[1])
    v1 = run_v1_parity(cfg, raw)
    v2 = run_v2(cfg, raw, v1)
    return {"v1": v1, "v2": v2}


def _segment_summary(customers: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    total_c = len(customers)
    total_r = float(customers["monetary"].sum())
    rows = []
    for name, grp in customers.groupby("segment"):
        rows.append(
            {
                "segment": name,
                "customers": int(len(grp)),
                "pct_customers": len(grp) / total_c,
                "revenue": float(grp["monetary"].sum()),
                "pct_revenue": float(grp["monetary"].sum() / total_r) if total_r else 0.0,
                "median_recency": float(grp["recency_days"].median()),
                "mean_recency": float(grp["recency_days"].mean()),
                "median_frequency": float(grp["frequency"].median()),
                "mean_frequency": float(grp["frequency"].mean()),
                "median_monetary": float(grp["monetary"].median()),
                "mean_monetary": float(grp["monetary"].mean()),
                "aov": float(grp["avg_order_value"].mean()),
            }
        )
    out = pd.DataFrame(rows).merge(lookup[["segment", "sort_order", "colour", "action", "primary_kpi", "description"]], on="segment")
    return out.sort_values("sort_order")


def _v1_v2_crosstab(v1_customers: pd.DataFrame, v2_customers: pd.DataFrame) -> pd.DataFrame:
    a = v1_customers[["CustomerID", "Segment"]].rename(columns={"Segment": "segment_v1"})
    b = v2_customers[["CustomerID", "segment"]].rename(columns={"segment": "segment_v2"})
    merged = a.merge(b, on="CustomerID", how="outer")
    merged["segment_v1"] = merged["segment_v1"].fillna("(not in v1)")
    merged["segment_v2"] = merged["segment_v2"].fillna("(not in v2)")
    return (
        merged.groupby(["segment_v1", "segment_v2"], as_index=False)
        .size()
        .rename(columns={"size": "customers"})
    )


def _excluded_customers(fact: pd.DataFrame, kept: pd.DataFrame) -> pd.DataFrame:
    identified = fact.loc[fact["CustomerID"].notna() & (fact["line_type"] == "sale")]
    totals = identified.groupby("CustomerID", as_index=False)["net_revenue"].sum()
    kept_ids = set(kept["CustomerID"].astype(int))
    excluded = totals.loc[~totals["CustomerID"].astype(int).isin(kept_ids)].copy()
    return excluded.sort_values("net_revenue")


def _acceptance_flags(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    fact: pd.DataFrame,
    cfg: dict[str, Any],
    excluded: pd.DataFrame,
) -> dict[str, Any]:
    top50 = set(customers.nlargest(50, "monetary")["CustomerID"].astype(int))
    excluded_ids = set(excluded["CustomerID"].astype(int))
    giant_ids = {16446, 12346}
    top10_desc = products.head(10)["Description"].astype(str).str.upper().tolist()
    non_product = {c.upper() for c in cfg["cleaning"]["non_product_stock_codes"]}
    product_codes = set(products["StockCode"].astype(str).str.upper())
    return {
        "giant_cancel_customers_not_top50": giant_ids.isdisjoint(top50),
        "giant_cancel_customers_excluded_or_not_top50": giant_ids.issubset(excluded_ids | (set(customers["CustomerID"].astype(int)) - top50)),
        "paper_craft_not_top10": not any("PAPER CRAFT" in d and "LITTLE BIRDIE" in d for d in top10_desc),
        "ceramic_jar_not_top10": not any("MEDIUM CERAMIC TOP STORAGE JAR" in d for d in top10_desc),
        "no_non_product_in_product_ranking": product_codes.isdisjoint(non_product),
        "customers_unique": int(customers["CustomerID"].nunique()) == len(customers),
        "recency_min": int(customers["recency_days"].min()),
        "no_empty_segment": bool(customers["segment"].nunique() == len(cfg["segments"])),
    }


def _data_quality_payload(
    cfg: dict[str, Any],
    raw: pd.DataFrame,
    deduped: pd.DataFrame,
    classified: pd.DataFrame,
    netted: pd.DataFrame,
    fact: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    net_stats: dict[str, float],
    month_coverage: pd.DataFrame,
    review_codes: pd.DataFrame,
    excluded: pd.DataFrame,
    n_raw: int,
    n_deduped: int,
    v1: dict[str, Any],
) -> dict[str, Any]:
    missing_id = int(raw["CustomerID"].isna().sum())
    non_product = classified.loc[classified["line_type"] == "non_product"]
    by_code = (
        non_product.groupby(non_product["StockCode"].astype(str).str.upper())
        .agg(gross_value=("gross_revenue", lambda s: float(s.clip(lower=0).sum())), n=("InvoiceNo", "size"))
        .reset_index()
        .sort_values("gross_value", ascending=False)
    )
    before = (
        classified.assign(abs_gross=lambda d: (d["Quantity"] * d["UnitPrice"]).abs())
        .nlargest(10, "abs_gross")
        .loc[:, ["InvoiceNo", "StockCode", "Description", "CustomerID", "Quantity", "UnitPrice", "InvoiceDate"]]
    )
    after = netted.nlargest(10, "net_revenue").loc[
        :, ["InvoiceNo", "StockCode", "Description", "CustomerID", "net_quantity", "UnitPrice", "net_revenue"]
    ]
    return {
        "n_raw": n_raw,
        "n_deduped": n_deduped,
        "n_duplicates_dropped": n_raw - n_deduped,
        "missing_customer_id_rows": missing_id,
        "net_stats": net_stats,
        "non_product_by_code": by_code.to_dict(orient="records"),
        "month_coverage": month_coverage.to_dict(orient="records"),
        "excluded_customers": int(len(excluded)),
        "excluded_ids_sample": excluded.head(20).to_dict(orient="records"),
        "review_codes": review_codes.head(40).to_dict(orient="records"),
        "top10_before": before.to_dict(orient="records"),
        "top10_after": after.to_dict(orient="records"),
        "n_customers": int(len(customers)),
        "v1_revenue": v1["revenue"],
        "v2_identified_net": float(customers["monetary"].sum()),
    }


def _parity_markdown(
    counts: dict[str, int],
    n_customers: int,
    n_invoices: int,
    revenue: float,
    aov: float,
    repeat_rate: float,
    our_counts: dict[str, int],
    ref_counts: dict[str, Any],
    deltas: dict[str, int],
    match_rate: float,
    checks: dict[str, bool],
    passed: bool,
    ref_date: pd.Timestamp,
    cfg: dict[str, Any],
) -> str:
    status = "PASSED" if passed else "FAILED"
    seg_lines = "\n".join(
        f"| {name} | {int(our_counts.get(name, 0))} | {int(ref_counts.get(name, 0))} | {deltas.get(name, 0)} |"
        for name in sorted(set(our_counts) | set(ref_counts))
    )
    check_lines = "\n".join(f"| {k} | {'yes' if v else 'NO'} |" for k, v in checks.items())
    return f"""# v1 parity report

**Status: {status}**

Reproduction of `reference/script.R` (not the v2 methodology). Reference date = {ref_date}.

## Row-count checkpoints

| Step | This run | Spec |
|---|---:|---:|
| After dedupe | {counts['after_dedupe']:,} | {cfg['v1']['rows_after_dedupe']:,} |
| After CustomerID | {counts['after_customer_id']:,} | {cfg['v1']['rows_after_customer_id']:,} |
| After cancellations | {counts['after_cancellations']:,} | {cfg['v1']['rows_after_cancellations']:,} |
| Final | {counts['final']:,} | {cfg['v1']['rows_final']:,} |

## KPI checkpoints

| KPI | This run | Spec |
|---|---:|---|
| Customers | {n_customers:,} | {cfg['v1']['n_customers_expected']:,} |
| Invoices | {n_invoices:,} | {cfg['v1']['n_invoices_expected']:,} |
| Revenue | {cfg['project']['currency_symbol']}{revenue:,.2f} | {cfg['project']['currency_symbol']}{cfg['v1']['revenue_expected']:,.0f} ± {cfg['v1']['revenue_tolerance']} |
| AOV | {cfg['project']['currency_symbol']}{aov:,.2f} | {cfg['project']['currency_symbol']}{cfg['v1']['aov_expected']} ± {cfg['v1']['aov_tolerance']} |
| Repeat purchase rate | {repeat_rate:.1f}% | {cfg['v1']['repeat_rate_pct_expected']}% ± {cfg['v1']['repeat_rate_tolerance_pp']} pp |

Segment match rate vs `reference/rfm_customer_table.csv`: **{match_rate:.4%}**

## Segment counts vs `reference/segment_summary.csv`

| Segment | This run | Reference | Abs delta |
|---|---:|---:|---:|
{seg_lines}

## Checks

| Check | Pass |
|---|---|
{check_lines}
"""


def _dq_markdown(dq: dict[str, Any], cfg: dict[str, Any]) -> str:
    np_lines = "\n".join(
        f"| {r['StockCode']} | {r['n']:,} | {cfg['project']['currency_symbol']}{r['gross_value']:,.0f} |"
        for r in dq["non_product_by_code"]
    )
    month_lines = "\n".join(
        f"| {pd.Timestamp(r['month_start']).strftime('%Y-%m')} | {r['days_present']} / {r['days_in_month']} | {r['coverage']:.0%} | {'yes' if r['is_partial_month'] else ''} |"
        for r in dq["month_coverage"]
    )
    before_lines = "\n".join(
        f"| {r['InvoiceNo']} | {r.get('CustomerID')} | {r['StockCode']} | {r['Description']} | {r['Quantity']} | {r['UnitPrice']} |"
        for r in dq["top10_before"]
    )
    after_lines = "\n".join(
        f"| {r['InvoiceNo']} | {r.get('CustomerID')} | {r['StockCode']} | {r['Description']} | {r['net_quantity']} | {cfg['project']['currency_symbol']}{r['net_revenue']:,.0f} |"
        for r in dq["top10_after"]
    )
    review_lines = "\n".join(
        f"| {r['StockCode']} | {r['n_lines']:,} | {r['examples']} |"
        for r in dq["review_codes"][:25]
    )
    return f"""# Data quality report (v2)

All value figures are **historical net** unless labelled otherwise.

## Row counts

| Step | Rows |
|---|---:|
| Raw | {dq['n_raw']:,} |
| After exact-duplicate drop | {dq['n_deduped']:,} |
| Duplicates dropped | {dq['n_duplicates_dropped']:,} |
| Rows with missing CustomerID | {dq['missing_customer_id_rows']:,} |
| Customers in segmentation | {dq['n_customers']:,} |
| Customers excluded (net revenue ≤ 0) | {dq['excluded_customers']:,} |

## Cancellation netting (LIFO, identified product lines)

| Item | Value |
|---|---|
| Matched cancellation value | {cfg['project']['currency_symbol']}{dq['net_stats']['matched_cancel_value']:,.2f} |
| Unmatched returns (sale predates data) | {cfg['project']['currency_symbol']}{dq['net_stats']['unmatched_cancel_value']:,.2f} |
| v1 gross revenue (identified, cancellations dropped) | {cfg['project']['currency_symbol']}{dq['v1_revenue']:,.2f} |
| v2 identified customer net revenue | {cfg['project']['currency_symbol']}{dq['v2_identified_net']:,.2f} |

## Non-product StockCodes (excluded from product and customer value)

| StockCode | Lines | Positive gross value |
|---|---:|---:|
{np_lines}

## Partial months

A month is partial when observed calendar-day coverage is below {cfg['cleaning']['partial_month_day_coverage']:.0%}.

| Month | Days present | Coverage | Partial |
|---|---|---|---|
{month_lines}

## Largest single lines before netting (absolute gross)

| Invoice | Customer | SKU | Description | Qty | Unit price |
|---|---|---|---|---:|---:|
{before_lines}

## Largest single lines after netting (net revenue)

| Invoice | Customer | SKU | Description | Net qty | Net revenue |
|---|---|---|---|---:|---:|
{after_lines}

## StockCodes that do not match the 5-digit product pattern

These are **not** auto-excluded (DCGS and similar can be real products). Listed for review.

| StockCode | Lines | Example descriptions |
|---|---:|---|
{review_lines}

## Excluded customers

{dq['excluded_customers']} identified customers have historical net revenue ≤ 0 after netting and are listed in `outputs/tables/excluded_customers.csv`. Customer 12346 (MEDIUM CERAMIC TOP STORAGE JAR, fully reversed) is in that list. Customer 16446's PAPER CRAFT line nets to zero; they remain in the model only if other SKUs leave a small positive net, and they are not in the top 50 by monetary.
"""
