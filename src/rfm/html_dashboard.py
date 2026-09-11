"""Standalone HTML executive dashboard for resume screenshots (no server)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rfm.io import read_csv, read_json

logger = logging.getLogger(__name__)

NAVY = "#1B4F72"
ACCENT = "#CA6F1E"
GREEN = "#1E8449"


def export_html_dashboard(cfg: dict[str, Any]) -> Path:
    path: Path = cfg["paths"]["html_dashboard"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tables = cfg["paths"]["tables_dir"]
    metrics = read_json(cfg["paths"]["metrics_json"])
    segments = read_csv(tables / "segment_summary.csv")
    clusters = read_csv(tables / "cluster_profile.csv")
    monthly = read_csv(tables / "monthly_kpis.csv")
    products = read_csv(tables / "product_summary.csv").head(10)
    winback = read_csv(tables / "winback_scenarios.csv")
    customers = read_csv(tables / "dim_customer.csv")

    monthly["month_start"] = pd.to_datetime(monthly["month_start"])
    complete = monthly.loc[~monthly["is_partial_month"]]
    partial = monthly.loc[monthly["is_partial_month"]]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Monthly identified net £ (orange = partial month)",
            "Segment: customer share vs revenue share",
            "Clusters named from median RFM (not cluster id)",
            "Win-back scenarios — not a forecast",
            "Top 10 products (non-product codes excluded)",
            "Top countries by identified net",
        ),
        vertical_spacing=0.12, horizontal_spacing=0.10,
    )
    fig.add_trace(
        go.Scatter(
            x=complete["month_start"], y=complete["identified_net_revenue"],
            mode="lines+markers", name="Complete months",
            line=dict(color=NAVY, width=3),
        ),
        row=1, col=1,
    )
    if not partial.empty:
        fig.add_trace(
            go.Scatter(
                x=partial["month_start"], y=partial["identified_net_revenue"],
                mode="markers", name="Partial month",
                marker=dict(color=ACCENT, size=12),
            ),
            row=1, col=1,
        )
    fig.add_trace(
        go.Bar(
            x=segments["segment"], y=segments["pct_customers"] * 100,
            name="% customers", marker_color=NAVY,
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            x=segments["segment"], y=segments["pct_revenue"] * 100,
            name="% historical net", marker_color=ACCENT,
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(
            y=clusters["cluster_name"], x=clusters["share"] * 100,
            orientation="h", name="% customers", marker_color=NAVY,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(
            y=clusters["cluster_name"], x=clusters["revenue_share"] * 100,
            orientation="h", name="% net", marker_color=ACCENT,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[f"{s} {r:.0%}" for s, r in zip(winback["segment"], winback["reactivation_rate"], strict=True)],
            y=winback["expected_winback_revenue"],
            marker_color=GREEN, name="Scenario £",
        ),
        row=2, col=2,
    )
    fig.add_trace(
        go.Bar(
            y=products["Description"].astype(str).str[:36],
            x=products["net_revenue"], orientation="h",
            marker_color=NAVY, name="Product net",
        ),
        row=3, col=1,
    )
    by_cty = (
        customers.groupby("Country", as_index=False)["monetary"].sum()
        .sort_values("monetary", ascending=False).head(10)
    )
    fig.add_trace(
        go.Bar(
            y=by_cty["Country"], x=by_cty["monetary"], orientation="h",
            marker_color=ACCENT, name="Country net",
        ),
        row=3, col=2,
    )
    fig.update_layout(
        height=1400, width=1180, barmode="group",
        paper_bgcolor="#F4F6F7", plot_bgcolor="white",
        font=dict(family="Segoe UI, Arial", size=12, color="#1B2631"),
        legend=dict(orientation="h", y=1.02),
        margin=dict(t=40, l=40, r=20, b=40),
        showlegend=True,
    )
    charts = fig.to_html(full_html=False, include_plotlyjs="cdn")

    v2 = metrics["v2"]
    rar = metrics["revenue_at_risk"]
    par = metrics["pareto"]
    mig = metrics["migration"]
    exp = metrics["experiment"]
    champs = segments.loc[segments["segment"] == "Champions"].iloc[0]
    kpis = [
        ("Customers", f"{v2['n_customers']:,}"),
        ("Orders", f"{v2['n_orders']:,}"),
        ("Identified net", f"£{v2['identified_customers_net_revenue']:,.0f}"),
        ("AOV", f"£{v2['aov']:,.0f}"),
        ("Repeat rate", f"{v2['repeat_rate_pct']:.1f}%"),
        ("Revenue at risk", f"£{rar['historical_net_revenue']:,.0f}"),
    ]
    cards = "".join(
        f'<div class="card"><div class="lab">{lab}</div><div class="val">{val}</div></div>'
        for lab, val in kpis
    )
    seg_rows = "".join(
        "<tr>"
        f"<td>{s.segment}</td><td>{int(s.customers):,}</td>"
        f"<td>{s.pct_customers*100:.1f}%</td><td>£{s.revenue:,.0f}</td>"
        f"<td>{s.pct_revenue*100:.1f}%</td><td>{s.action}</td>"
        "</tr>"
        for s in segments.itertuples()
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>RFM Executive Dashboard — historical net</title>
<style>
body {{ margin:0; font-family:"Segoe UI",Arial,sans-serif; background:#F4F6F7; color:#1B2631; }}
header {{ background:#1B2631; color:#fff; padding:20px 32px; }}
header h1 {{ margin:0; font-size:22px; }}
header p {{ margin:6px 0 0; opacity:.85; font-size:13px; }}
.kpis {{ display:flex; gap:12px; padding:16px 32px; flex-wrap:wrap; }}
.card {{ background:#fff; border-top:4px solid {NAVY}; padding:12px 16px; min-width:150px; flex:1; }}
.card .lab {{ font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:#5D6D7E; }}
.card .val {{ font-size:22px; font-weight:700; color:{NAVY}; margin-top:4px; }}
section {{ padding:8px 32px 24px; }}
h2 {{ font-size:16px; color:{NAVY}; margin:16px 0 8px; }}
table {{ border-collapse:collapse; width:100%; background:#fff; font-size:13px; }}
th {{ background:{NAVY}; color:#fff; text-align:left; padding:8px; }}
td {{ border-bottom:1px solid #E5E8E8; padding:8px; vertical-align:top; }}
.note {{ font-size:12px; color:#5D6D7E; }}
</style>
</head>
<body>
<header>
  <h1>RFM customer value dashboard</h1>
  <p>UCI Online Retail · UK giftware · 1 Dec 2010 – 9 Dec 2011 · all figures are <strong>historical net</strong>, not forecasts · snapshot {v2['snapshot_date']}</p>
</header>
<div class="kpis">{cards}</div>
<section>
  <p class="note">Champions are {champs.pct_customers*100:.1f}% of customers and {champs.pct_revenue*100:.1f}% of historical net.
  Top 20% of customers generate {par['top_20_pct_revenue_share']*100:.0f}% (Gini {par['gini']:.2f}).
  {mig['pct_champions_still_champions']*100:.0f}% of June Champions were still Champions in December.
  At-Risk test n={exp['n_segment']:,} is too small for a powered +5 pp lift (need ~{exp['n_total_needed_for_target_lift']:,.0f}).
  Dec 2011 is a partial month, not a 55% collapse.</p>
  {charts}
  <h2>Segment playbook</h2>
  <table>
    <thead><tr><th>Segment</th><th>Customers</th><th>%</th><th>Historical net</th><th>% rev</th><th>Action</th></tr></thead>
    <tbody>{seg_rows}</tbody>
  </table>
  <p class="note">Source: pipeline outputs in data/processed. Rebuild with <code>python run.py excel</code>.</p>
</section>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    logger.info("Wrote HTML dashboard %s", path)
    return path
