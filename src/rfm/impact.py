"""Pareto / Gini, win-back scenarios, and experiment sample-size / MDE."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def gini_coefficient(values: np.ndarray) -> float:
    """Gini of a non-negative vector. 0 = equal, 1 = all value on one customer."""
    x = np.sort(np.asarray(values, dtype=float))
    x = x[x >= 0]
    if x.sum() == 0:
        return 0.0
    n = x.size
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * x) / (n * np.sum(x))) - (n + 1) / n)


def concentration(customers: pd.DataFrame, shares: list[float]) -> dict[str, float]:
    ordered = customers.sort_values("monetary", ascending=False)
    total = float(ordered["monetary"].sum())
    n = len(ordered)
    cum = ordered["monetary"].cumsum().to_numpy() / total if total else np.zeros(n)
    out: dict[str, float] = {"gini": gini_coefficient(ordered["monetary"].to_numpy())}
    for share in shares:
        k = max(1, int(np.ceil(share * n)))
        out[f"top_{int(share * 100)}_pct_revenue_share"] = float(cum[k - 1])
    out["pareto_cum_share"] = cum.tolist()
    return out


def revenue_at_risk(dim_customer: pd.DataFrame, segment_names: list[str]) -> dict[str, float]:
    mask = dim_customer["segment"].isin(segment_names)
    subset = dim_customer.loc[mask]
    return {
        "n_customers": int(len(subset)),
        "historical_net_revenue": float(subset["monetary"].sum()),
    }


def winback_scenarios(
    dim_customer: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """rate × customers × segment AOV × expected orders. Labelled as a scenario."""
    names = cfg["winback"]["at_risk_segment_names"]
    rates = list(cfg["winback"]["reactivation_rates"])
    expected_orders = float(cfg["winback"]["expected_orders_per_reactivated"])
    rows = []
    for segment, grp in dim_customer.loc[dim_customer["segment"].isin(names)].groupby("segment"):
        n = int(len(grp))
        aov = float(grp["avg_order_value"].mean())
        for rate in rates:
            expected_revenue = rate * n * aov * expected_orders
            rows.append(
                {
                    "segment": segment,
                    "customers": n,
                    "avg_order_value": aov,
                    "reactivation_rate": rate,
                    "expected_orders_per_reactivated": expected_orders,
                    "expected_winback_revenue": expected_revenue,
                    "label": cfg["winback"]["label"],
                }
            )
    return pd.DataFrame(rows)


def _p_from_cohens_h(h: float, p_control: float) -> float:
    """Invert Cohen's h = 2 arcsin sqrt(p_t) - 2 arcsin sqrt(p_c)."""
    angle = np.arcsin(np.sqrt(np.clip(p_control, 0.0, 1.0))) + h / 2.0
    return float(np.clip(np.sin(angle) ** 2, 0.0, 1.0))


def experiment_design(n_segment: int, cfg: dict[str, Any]) -> dict[str, Any]:
    exp = cfg["experiment"]
    treatment_share = float(exp["treatment_share"])
    control_share = float(exp["control_share"])
    baseline = float(exp["baseline_reactivation_rate"])
    alpha = float(exp["alpha"])
    power = float(exp["power"])
    lift = float(exp["target_lift_pp"])
    alternative = str(exp["alternative"])

    analysis = NormalIndPower()
    n_t = n_segment * treatment_share
    n_c = n_segment * control_share
    ratio = (n_c / n_t) if n_t else np.nan

    mde_pp = float("nan")
    if n_t > 1 and n_c > 1:
        es = analysis.solve_power(
            effect_size=None,
            nobs1=n_t,
            alpha=alpha,
            power=power,
            ratio=ratio,
            alternative=alternative,
        )
        p_treat = _p_from_cohens_h(float(es), baseline)
        mde_pp = p_treat - baseline

    es_target = proportion_effectsize(baseline + lift, baseline)
    n_treat_needed = float(
        analysis.solve_power(
            effect_size=es_target,
            nobs1=None,
            alpha=alpha,
            power=power,
            ratio=control_share / treatment_share,
            alternative=alternative,
        )
    )
    n_total_needed = n_treat_needed / treatment_share
    too_small = bool(n_segment < n_total_needed)

    return {
        "target_segment": exp["target_segment"],
        "n_segment": int(n_segment),
        "treatment_share": treatment_share,
        "control_share": control_share,
        "baseline_reactivation_rate": baseline,
        "alpha": alpha,
        "power": power,
        "target_lift_pp": lift,
        "window_days": int(exp["window_days"]),
        "mde_pp": float(mde_pp) if mde_pp == mde_pp else None,
        "n_total_needed_for_target_lift": float(n_total_needed),
        "too_small": too_small,
    }
