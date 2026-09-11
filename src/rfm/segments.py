"""Ordered RFM segment rules loaded from config. First match wins; no silent default."""

from __future__ import annotations

from typing import Any

import pandas as pd

OPS = {
    "ge": lambda a, b: a >= b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "lt": lambda a, b: a < b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
}


def _eval_one(record: dict[str, Any], cond: dict[str, Any]) -> bool:
    op = OPS[cond["op"]]
    return bool(op(record[cond["field"]], cond["value"]))


def conditions_match(record: dict[str, Any], conditions: list[dict[str, Any]]) -> bool:
    """AND of conditions; a condition may be {'any': [cond, ...]} for OR."""
    if not conditions:
        return True
    for cond in conditions:
        if "any" in cond:
            if not any(_eval_one(record, item) for item in cond["any"]):
                return False
        elif not _eval_one(record, cond):
            return False
    return True


def assign_segment_v1(r: int, f: int, m: int) -> str:
    """Eight v1 rules from reference/script.R, first match wins."""
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 3:
        return "Loyal Customers"
    if r >= 4 and f <= 2:
        return "New Customers"
    if r >= 3 and f >= 2 and m >= 3:
        return "Potential Loyalists"
    if m >= 4 and f <= 2:
        return "Big Spenders"
    if r <= 2 and f >= 3 and m >= 3:
        return "At-Risk Customers"
    if r <= 2 and f >= 2:
        return "Hibernating Customers"
    return "Lost Customers"


def assign_segments_v1(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Segment"] = [
        assign_segment_v1(int(r), int(f), int(m))
        for r, f, m in zip(out["R_Score"], out["F_Score"], out["M_Score"], strict=True)
    ]
    return out


def assign_segments_v2(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Apply ordered config rules. Raises if any customer matches nothing."""
    rules = cfg["segments"]
    names: list[str] = []
    unmatched: list[int] = []
    for idx, row in df.iterrows():
        record = {
            "r_score": int(row["r_score"]),
            "f_score": int(row["f_score"]),
            "m_score": int(row["m_score"]),
            "tenure_days": int(row["tenure_days"]),
        }
        hit = None
        for rule in rules:
            if conditions_match(record, rule["conditions"]):
                hit = rule["name"]
                break
        if hit is None:
            unmatched.append(int(idx) if isinstance(idx, int) else len(names))
            names.append("UNMATCHED")
        else:
            names.append(hit)
    if unmatched:
        raise ValueError(
            f"{len(unmatched)} customers matched no segment rule (e.g. index {unmatched[:5]})."
        )
    out = df.copy()
    out["segment"] = names
    return out


def segment_lookup(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for rule in cfg["segments"]:
        rows.append(
            {
                "segment": rule["name"],
                "sort_order": rule["sort_order"],
                "description": rule["description"],
                "action": rule["action"],
                "primary_kpi": rule["primary_kpi"],
                "colour": rule["colour"],
                "rule_text": _rule_text(rule["conditions"]),
            }
        )
    return pd.DataFrame(rows).sort_values("sort_order")


def _rule_text(conditions: list[dict[str, Any]]) -> str:
    parts = []
    for cond in conditions:
        if "any" in cond:
            inner = " OR ".join(_fmt(c) for c in cond["any"])
            parts.append(f"({inner})")
        else:
            parts.append(_fmt(cond))
    return " AND ".join(parts) if parts else "(default)"


def _fmt(cond: dict[str, Any]) -> str:
    symbols = {"ge": "≥", "le": "≤", "gt": ">", "lt": "<", "eq": "=", "ne": "≠"}
    return f"{cond['field']} {symbols[cond['op']]} {cond['value']}"


def coverage_grid(cfg: dict[str, Any]) -> pd.DataFrame:
    """Every (R, F, M) × tenure {30, 90} mapped to a segment — used by tests and docs."""
    rows = []
    for r in range(1, 6):
        for f in range(1, 6):
            for m in range(1, 6):
                for tenure in (30, 90):
                    record = {"r_score": r, "f_score": f, "m_score": m, "tenure_days": tenure}
                    hit = None
                    for rule in cfg["segments"]:
                        if conditions_match(record, rule["conditions"]):
                            hit = rule["name"]
                            break
                    rows.append(
                        {
                            "r_score": r,
                            "f_score": f,
                            "m_score": m,
                            "tenure_days": tenure,
                            "segment": hit,
                        }
                    )
    grid = pd.DataFrame(rows)
    if grid["segment"].isna().any():
        missing = grid.loc[grid["segment"].isna()]
        raise ValueError(f"Rule set does not cover {len(missing)} RFM/tenure cells.")
    return grid


def rf_cell_segments(cfg: dict[str, Any]) -> pd.DataFrame:
    """5×5 R×F cells listing which segments can appear (across M and tenure)."""
    grid = coverage_grid(cfg)
    cells = (
        grid.groupby(["r_score", "f_score"], sort=True)["segment"]
        .apply(lambda s: ", ".join(dict.fromkeys(s)))
        .reset_index(name="segments")
    )
    return cells
