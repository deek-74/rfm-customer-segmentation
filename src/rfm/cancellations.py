"""LIFO cancellation netting against prior sale lines of the same customer and SKU."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def net_cancellations(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Allocate cancellations to the most recent prior sale lines (LIFO).

    Only product lines (sale / cancellation) with a CustomerID are netted.
    Non-product rows and guest (null CustomerID) rows pass through with
    net_quantity equal to Quantity for positive sales and 0 otherwise.
    """
    out = df.copy().reset_index(drop=True)
    n = len(out)
    row_id = np.arange(n)
    net_qty = np.zeros(n, dtype=float)
    unmatched = np.zeros(n, dtype=float)

    product_id = out["line_type"].isin(["sale", "cancellation"]) & out["CustomerID"].notna()
    guest_sale = (
        ~product_id & (out["line_type"] == "sale") & (out["Quantity"] > 0)
    )
    net_qty[guest_sale.to_numpy()] = out.loc[guest_sale, "Quantity"].to_numpy(dtype=float)

    work = out.loc[product_id].copy()
    work["_row"] = row_id[product_id.to_numpy()]
    work = work.sort_values(
        ["CustomerID", "StockCode", "InvoiceDate", "_row"], kind="mergesort"
    )

    cust = work["CustomerID"].to_numpy()
    stock = work["StockCode"].astype(str).to_numpy()
    qty = work["Quantity"].to_numpy(dtype=float)
    price = work["UnitPrice"].to_numpy(dtype=float)
    is_sale = (work["line_type"].to_numpy() == "sale") & (qty > 0)
    rows = work["_row"].to_numpy(dtype=int)

    matched_value = 0.0
    unmatched_value = 0.0
    n_groups = 0
    stack_row: list[int] = []
    stack_qty: list[float] = []
    stack_price: list[float] = []

    def new_group(i: int) -> bool:
        return i == 0 or cust[i] != cust[i - 1] or stock[i] != stock[i - 1]

    for i in range(len(work)):
        if new_group(i):
            n_groups += 1
            stack_row.clear()
            stack_qty.clear()
            stack_price.clear()
        if is_sale[i]:
            net_qty[rows[i]] = qty[i]
            stack_row.append(int(rows[i]))
            stack_qty.append(float(qty[i]))
            stack_price.append(float(price[i]))
            continue
        need = abs(float(qty[i]))
        while need > 1e-12 and stack_row:
            take = min(stack_qty[-1], need)
            stack_qty[-1] -= take
            net_qty[stack_row[-1]] -= take
            matched_value += take * stack_price[-1]
            need -= take
            if stack_qty[-1] <= 1e-12:
                stack_row.pop()
                stack_qty.pop()
                stack_price.pop()
        if need > 1e-12:
            unmatched[rows[i]] = need
            unmatched_value += need * float(price[i])

    out["net_quantity"] = net_qty
    out["unmatched_qty"] = unmatched
    out["net_revenue"] = out["net_quantity"] * out["UnitPrice"]
    negative = out["net_quantity"] < 0
    out.loc[negative, "net_quantity"] = 0.0
    out.loc[out["net_quantity"] <= 0, "net_revenue"] = 0.0

    stats = {
        "n_netted_groups": n_groups,
        "matched_cancel_value": float(matched_value),
        "unmatched_cancel_value": float(unmatched_value),
        "net_revenue_all_lines": float(out["net_revenue"].sum()),
    }
    logger.info(
        "LIFO netting groups=%s matched_cancel=%.2f unmatched_cancel=%.2f",
        n_groups,
        matched_value,
        unmatched_value,
    )
    return out, stats
