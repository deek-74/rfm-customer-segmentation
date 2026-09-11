"""Cohort retention matrices. Partial months are excluded from denominators."""

from __future__ import annotations

import pandas as pd


def build_retention_matrices(
    activity: pd.DataFrame, dim_customer: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long-format customer-retention and revenue-retention tables.

    Month-0 is always 100% for every complete cohort. Partial activity months and
    cohorts whose first month is partial are dropped from the matrices.
    """
    customers = dim_customer.copy()
    customers["cohort_month"] = pd.to_datetime(customers["first_purchase_date"]).dt.to_period(
        "M"
    ).dt.to_timestamp()
    activity = activity.copy()
    activity["cohort_month"] = pd.to_datetime(activity["cohort_month"])
    activity["activity_month"] = pd.to_datetime(activity["activity_month"])

    partial_months = set(
        pd.to_datetime(activity.loc[activity["is_partial_month"], "activity_month"]).unique()
    )
    complete_cohorts = [c for c in customers["cohort_month"].unique() if c not in partial_months]
    customers = customers.loc[customers["cohort_month"].isin(complete_cohorts)]
    activity = activity.loc[
        activity["cohort_month"].isin(complete_cohorts)
        & ~activity["activity_month"].isin(partial_months)
    ]

    cohort_sizes = customers.groupby("cohort_month").size().rename("cohort_size")
    month0_rev = (
        activity.loc[activity["months_since"] == 0]
        .groupby("cohort_month")["month_revenue"]
        .sum()
        .rename("month0_revenue")
    )

    grouped = activity.groupby(["cohort_month", "months_since"], as_index=False).agg(
        active_customers=("CustomerID", "nunique"),
        month_revenue=("month_revenue", "sum"),
    )
    grouped = grouped.merge(cohort_sizes, on="cohort_month", how="left")
    grouped = grouped.merge(month0_rev, on="cohort_month", how="left")
    grouped["customer_retention"] = grouped["active_customers"] / grouped["cohort_size"]
    grouped["revenue_retention"] = grouped["month_revenue"] / grouped["month0_revenue"]

    # Guarantee month-0 = 100% even if a join is slightly off.
    grouped.loc[grouped["months_since"] == 0, "customer_retention"] = 1.0
    grouped.loc[grouped["months_since"] == 0, "revenue_retention"] = 1.0

    first_cohort = grouped["cohort_month"].min()
    grouped["is_left_censored"] = grouped["cohort_month"].eq(first_cohort)
    grouped = grouped.sort_values(["cohort_month", "months_since"])
    return grouped, customers.loc[:, ["CustomerID", "cohort_month"]]
