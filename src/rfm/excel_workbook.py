"""Resume Excel workbook: raw, cleaned, RFM calculations, segments, and charts."""

from __future__ import annotations

import logging
import numbers
from pathlib import Path
from typing import Any

import pandas as pd
import xlsxwriter
from xlsxwriter.worksheet import Worksheet

from rfm.io import read_csv, read_json, read_parquet

logger = logging.getLogger(__name__)

NAVY = "#1B4F72"
ACCENT = "#CA6F1E"
GREEN = "#1E8449"
WHITE = "#FFFFFF"
LIGHT = "#F4F6F7"
HEADER_FILL = "#1B2631"


def export_resume_workbook(cfg: dict[str, Any]) -> Path:
    """Write the RFM analysis workbook (raw, cleaned, scores, segments, charts)."""
    path: Path = cfg["paths"]["resume_workbook"]
    path.parent.mkdir(parents=True, exist_ok=True)
    tables = cfg["paths"]["tables_dir"]
    metrics = read_json(cfg["paths"]["metrics_json"])

    raw = read_parquet(cfg["paths"]["interim_raw_parquet"])
    fact = read_parquet(cfg["paths"]["interim_staged_parquet"])
    customers = read_csv(tables / "dim_customer.csv")
    segments = read_csv(tables / "segment_summary.csv")
    clusters = read_csv(tables / "cluster_profile.csv")
    monthly = read_csv(tables / "monthly_kpis.csv")
    products = read_csv(tables / "product_summary.csv")
    winback = read_csv(tables / "winback_scenarios.csv")
    rules = pd.DataFrame(cfg["segments"])

    logger.info("Writing workbook %s", path)
    wb = xlsxwriter.Workbook(str(path), {"nan_inf_to_errors": True, "strings_to_urls": False})
    fmt = _formats(wb)

    _summary(wb, fmt, metrics, segments, clusters)
    _graphs(wb, fmt, monthly, segments, clusters, winback, products, customers)
    _rules(wb, fmt, rules)
    _write_dataframe(wb, fmt, "Raw_Transactions", raw)
    _write_dataframe(wb, fmt, "Cleaned_Transactions", _cleaned_view(fact))
    _customer_rfm(wb, fmt, customers)
    _segment_summary(wb, fmt, segments)
    _write_dataframe(wb, fmt, "Cluster_Profile", clusters)

    wb.close()
    logger.info("Wrote %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def _formats(wb: xlsxwriter.Workbook) -> dict[str, Any]:
    return {
        "title": wb.add_format({
            "bold": True, "font_size": 20, "font_color": WHITE, "fg_color": HEADER_FILL,
            "valign": "vcenter",
        }),
        "section": wb.add_format({"bold": True, "font_size": 13, "font_color": NAVY}),
        "label": wb.add_format({"bold": True, "font_color": NAVY, "bg_color": LIGHT}),
        "kpi_label": wb.add_format({
            "bold": True, "font_size": 9, "font_color": WHITE, "fg_color": NAVY,
            "align": "center", "valign": "vcenter",
        }),
        "kpi_value": wb.add_format({
            "bold": True, "font_size": 16, "font_color": NAVY, "bg_color": WHITE,
            "align": "center", "valign": "vcenter", "border": 1, "border_color": NAVY,
        }),
        "header": wb.add_format({
            "bold": True, "font_color": WHITE, "fg_color": NAVY, "border": 1,
            "text_wrap": True, "valign": "vcenter",
        }),
        "text": wb.add_format({"text_wrap": True, "valign": "top"}),
        "money": wb.add_format({"num_format": "£#,##0.00"}),
        "money0": wb.add_format({"num_format": "£#,##0"}),
        "pct": wb.add_format({"num_format": "0.0%"}),
        "int": wb.add_format({"num_format": "#,##0"}),
        "date": wb.add_format({"num_format": "yyyy-mm-dd"}),
        "datetime": wb.add_format({"num_format": "yyyy-mm-dd hh:mm"}),
        "calc_text": wb.add_format({"bg_color": "#FDEBD0"}),
        "wrap": wb.add_format({"text_wrap": True, "valign": "top"}),
    }


def _banner(ws: Worksheet, fmt: dict[str, Any], title: str, last_col: int = 10) -> None:
    ws.merge_range(0, 0, 0, last_col, title, fmt["title"])
    ws.set_row(0, 28)
    ws.freeze_panes(2, 0)
    ws.hide_gridlines(2)
    ws.set_tab_color(NAVY)


def _summary(
    wb: xlsxwriter.Workbook,
    fmt: dict[str, Any],
    metrics: dict[str, Any],
    segments: pd.DataFrame,
    clusters: pd.DataFrame,
) -> None:
    ws = wb.add_worksheet("Summary Report")
    _banner(ws, fmt, "Summary Report", 11)
    ws.set_column("A:A", 22)
    ws.set_column("B:L", 16)
    v2 = metrics["v2"]
    kpis = [
        ("Customers", f"{v2['n_customers']:,}"),
        ("Orders (countable)", f"{v2['n_orders']:,}"),
        ("Identified net", f"£{v2['identified_customers_net_revenue']:,.0f}"),
        ("AOV", f"£{v2['aov']:,.0f}"),
        ("Repeat rate", f"{v2['repeat_rate_pct']:.1f}%"),
        ("Revenue at risk", f"£{metrics['revenue_at_risk']['historical_net_revenue']:,.0f}"),
    ]
    for i, (lab, val) in enumerate(kpis):
        ws.write(4, i, lab, fmt["kpi_label"])
        ws.write(5, i, val, fmt["kpi_value"])
        ws.set_column(i, i, 18)
    ws.set_row(5, 28)

    ws.write(8, 0, "v1 vs v2 (identified)", fmt["section"])
    headers = ["Metric", "v1 (gross, R-parity)", "v2 (historical net)"]
    for c, h in enumerate(headers):
        ws.write(9, c, h, fmt["header"])
    pairs = [
        ("Customers", metrics["v1"]["n_customers"], v2["n_customers"]),
        ("Invoices / orders", metrics["v1"]["n_invoices"], v2["n_orders"]),
        ("Revenue £", metrics["v1"]["revenue"], v2["identified_customers_net_revenue"]),
        ("AOV £", metrics["v1"]["aov"], v2["aov"]),
        ("Repeat %", metrics["v1"]["repeat_rate_pct"], v2["repeat_rate_pct"]),
    ]
    for i, (name, a, b) in enumerate(pairs):
        ws.write(10 + i, 0, name)
        ws.write(10 + i, 1, a)
        ws.write(10 + i, 2, b)

    ws.write(16, 0, "RFM segments", fmt["section"])
    seg_headers = [
        "Segment", "Customers", "% customers", "Historical net £", "% revenue",
        "Mean recency", "Mean frequency", "AOV £", "Primary KPI",
    ]
    for c, h in enumerate(seg_headers):
        ws.write(17, c, h, fmt["header"])
    start = 18
    for i, (_, s) in enumerate(segments.iterrows()):
        r = start + i
        ws.write(r, 0, s["segment"])
        ws.write(r, 1, int(s["customers"]), fmt["int"])
        ws.write(r, 2, float(s["pct_customers"]), fmt["pct"])
        ws.write(r, 3, float(s["revenue"]), fmt["money0"])
        ws.write(r, 4, float(s["pct_revenue"]), fmt["pct"])
        ws.write(r, 5, float(s["mean_recency"]))
        ws.write(r, 6, float(s["mean_frequency"]))
        ws.write(r, 7, float(s["aov"]), fmt["money0"])
        ws.write(r, 8, s.get("primary_kpi", ""))

    ws.write(30, 0, "Clusters", fmt["section"])
    for c, h in enumerate(["Cluster", "Name", "Customers", "Share", "Median recency", "Median frequency", "Median £", "Revenue share"]):
        ws.write(31, c, h, fmt["header"])
    for i, (_, s) in enumerate(clusters.iterrows()):
        r = 32 + i
        ws.write(r, 0, int(s["cluster"]))
        ws.write(r, 1, s["cluster_name"])
        ws.write(r, 2, int(s["customers"]), fmt["int"])
        ws.write(r, 3, float(s["share"]), fmt["pct"])
        ws.write(r, 4, float(s["recency_days_median"]))
        ws.write(r, 5, float(s["frequency_median"]))
        ws.write(r, 6, float(s["monetary_median"]), fmt["money0"])
        ws.write(r, 7, float(s["revenue_share"]), fmt["pct"])

    ws.write(37, 0, "Headlines", fmt["section"])
    par = metrics["pareto"]
    mig = metrics["migration"]
    exp = metrics["experiment"]
    bullets = [
        f"Champions are {float(segments.loc[segments['segment']=='Champions','pct_customers'].iloc[0])*100:.1f}% of customers and {float(segments.loc[segments['segment']=='Champions','pct_revenue'].iloc[0])*100:.1f}% of historical net.",
        f"Top 20% of customers generate {par['top_20_pct_revenue_share']*100:.0f}% of historical net (Gini {par['gini']:.2f}).",
        f"{mig['pct_champions_still_champions']*100:.0f}% of June Champions were still Champions by December.",
        f"At-Risk experiment n={exp['n_segment']:,} is too small for a powered +5 pp test (need ~{exp['n_total_needed_for_target_lift']:,.0f}).",
        "December 2011 is 1–9 Dec only — not a 55% revenue collapse.",
    ]
    for i, b in enumerate(bullets):
        ws.merge_range(38 + i, 0, 38 + i, 8, f"• {b}", fmt["text"])
        ws.set_row(38 + i, 18)


def _graphs(
    wb: xlsxwriter.Workbook,
    fmt: dict[str, Any],
    monthly: pd.DataFrame,
    segments: pd.DataFrame,
    clusters: pd.DataFrame,
    winback: pd.DataFrame,
    products: pd.DataFrame,
    customers: pd.DataFrame,
) -> None:
    ws = wb.add_worksheet("Graphs and Charts")
    _banner(ws, fmt, "Graphs and Charts", 16)
    ws.set_column("A:B", 22)
    ws.set_column("C:R", 14)

    # --- source blocks ---
    ws.write(4, 0, "Monthly net revenue", fmt["section"])
    ws.write(5, 0, "Month", fmt["header"])
    ws.write(5, 1, "Net revenue", fmt["header"])
    ws.write(5, 2, "Partial?", fmt["header"])
    monthly = monthly.copy()
    monthly["month_start"] = pd.to_datetime(monthly["month_start"])
    for i, (_, row) in enumerate(monthly.iterrows()):
        ws.write_datetime(6 + i, 0, row["month_start"].to_pydatetime(), fmt["date"])
        ws.write(6 + i, 1, float(row["identified_net_revenue"]), fmt["money0"])
        ws.write(6 + i, 2, bool(row["is_partial_month"]))
    n_m = len(monthly)
    line = wb.add_chart({"type": "line"})
    line.add_series({
        "name": "Identified historical net",
        "categories": ["Graphs and Charts", 6, 0, 5 + n_m, 0],
        "values": ["Graphs and Charts", 6, 1, 5 + n_m, 1],
        "line": {"color": NAVY, "width": 2.25},
        "marker": {"type": "circle", "size": 6, "face_color": NAVY},
    })
    line.set_title({"name": "Monthly identified net revenue — Dec 2011 is partial, not a collapse"})
    line.set_y_axis({"name": "Historical net £", "num_format": "£#,##0"})
    line.set_size({"width": 720, "height": 280})
    line.set_legend({"none": True})
    ws.insert_chart(4, 4, line)

    r0 = 6 + n_m + 2
    ws.write(r0, 0, "Segment share vs revenue", fmt["section"])
    ws.write(r0 + 1, 0, "Segment", fmt["header"])
    ws.write(r0 + 1, 1, "% customers", fmt["header"])
    ws.write(r0 + 1, 2, "% revenue", fmt["header"])
    for i, (_, row) in enumerate(segments.iterrows()):
        ws.write(r0 + 2 + i, 0, row["segment"])
        ws.write(r0 + 2 + i, 1, float(row["pct_customers"]), fmt["pct"])
        ws.write(r0 + 2 + i, 2, float(row["pct_revenue"]), fmt["pct"])
    n_s = len(segments)
    col = wb.add_chart({"type": "column"})
    col.add_series({
        "name": "% customers",
        "categories": ["Graphs and Charts", r0 + 2, 0, r0 + 1 + n_s, 0],
        "values": ["Graphs and Charts", r0 + 2, 1, r0 + 1 + n_s, 1],
        "fill": {"color": NAVY},
    })
    col.add_series({
        "name": "% historical net",
        "categories": ["Graphs and Charts", r0 + 2, 0, r0 + 1 + n_s, 0],
        "values": ["Graphs and Charts", r0 + 2, 2, r0 + 1 + n_s, 2],
        "fill": {"color": ACCENT},
    })
    col.set_title({"name": "Champions punch above their weight in historical net revenue"})
    col.set_y_axis({"name": "Share", "num_format": "0%"})
    col.set_size({"width": 720, "height": 300})
    col.set_legend({"position": "bottom"})
    ws.insert_chart(r0, 4, col)

    r1 = r0 + n_s + 4
    ws.write(r1, 0, "Clusters", fmt["section"])
    ws.write(r1 + 1, 0, "Name", fmt["header"])
    ws.write(r1 + 1, 1, "% customers", fmt["header"])
    ws.write(r1 + 1, 2, "% revenue", fmt["header"])
    for i, (_, row) in enumerate(clusters.iterrows()):
        ws.write(r1 + 2 + i, 0, row["cluster_name"])
        ws.write(r1 + 2 + i, 1, float(row["share"]), fmt["pct"])
        ws.write(r1 + 2 + i, 2, float(row["revenue_share"]), fmt["pct"])
    n_c = len(clusters)
    bar = wb.add_chart({"type": "bar"})
    bar.add_series({
        "name": "% customers",
        "categories": ["Graphs and Charts", r1 + 2, 0, r1 + 1 + n_c, 0],
        "values": ["Graphs and Charts", r1 + 2, 1, r1 + 1 + n_c, 1],
        "fill": {"color": NAVY},
    })
    bar.add_series({
        "name": "% historical net",
        "categories": ["Graphs and Charts", r1 + 2, 0, r1 + 1 + n_c, 0],
        "values": ["Graphs and Charts", r1 + 2, 2, r1 + 1 + n_c, 2],
        "fill": {"color": ACCENT},
    })
    bar.set_title({"name": "Clusters named from median RFM profiles, not from cluster ids"})
    bar.set_size({"width": 720, "height": 240})
    ws.insert_chart(r1, 4, bar)

    r2 = r1 + n_c + 4
    ws.write(r2, 0, "Win-back scenarios (not a forecast)", fmt["section"])
    ws.write(r2 + 1, 0, "Segment", fmt["header"])
    ws.write(r2 + 1, 1, "Rate", fmt["header"])
    ws.write(r2 + 1, 2, "Expected £", fmt["header"])
    for i, (_, row) in enumerate(winback.iterrows()):
        ws.write(r2 + 2 + i, 0, row["segment"])
        ws.write(r2 + 2 + i, 1, float(row["reactivation_rate"]), fmt["pct"])
        ws.write(r2 + 2 + i, 2, float(row["expected_winback_revenue"]), fmt["money0"])
    n_w = len(winback)
    col2 = wb.add_chart({"type": "column"})
    col2.add_series({
        "name": "Scenario expected £",
        "categories": ["Graphs and Charts", r2 + 2, 0, r2 + 1 + n_w, 0],
        "values": ["Graphs and Charts", r2 + 2, 2, r2 + 1 + n_w, 2],
        "fill": {"color": GREEN},
    })
    col2.set_title({"name": "Win-back expected revenue at 5 / 10 / 20% reactivation (scenario)"})
    col2.set_size({"width": 720, "height": 260})
    col2.set_legend({"none": True})
    ws.insert_chart(r2, 4, col2)

    r3 = r2 + n_w + 4
    ws.write(r3, 0, "Top 10 products by historical net", fmt["section"])
    ws.write(r3 + 1, 0, "Product", fmt["header"])
    ws.write(r3 + 1, 1, "Net revenue", fmt["header"])
    top = products.head(10)
    for i, (_, row) in enumerate(top.iterrows()):
        ws.write(r3 + 2 + i, 0, str(row["Description"])[:40])
        ws.write(r3 + 2 + i, 1, float(row["net_revenue"]), fmt["money0"])
    bar2 = wb.add_chart({"type": "bar"})
    bar2.add_series({
        "name": "Net £",
        "categories": ["Graphs and Charts", r3 + 2, 0, r3 + 11, 0],
        "values": ["Graphs and Charts", r3 + 2, 1, r3 + 11, 1],
        "fill": {"color": NAVY},
    })
    bar2.set_title({"name": "Top 10 products — postage and manuals excluded"})
    bar2.set_size({"width": 720, "height": 280})
    bar2.set_legend({"none": True})
    ws.insert_chart(r3, 4, bar2)

    r4 = r3 + 14
    ws.write(r4, 0, "Country (customer monetary)", fmt["section"])
    by_cty = (
        customers.groupby("Country", as_index=False)["monetary"].sum()
        .sort_values("monetary", ascending=False).head(10)
    )
    ws.write(r4 + 1, 0, "Country", fmt["header"])
    ws.write(r4 + 1, 1, "Historical net £", fmt["header"])
    for i, (_, row) in enumerate(by_cty.iterrows()):
        ws.write(r4 + 2 + i, 0, row["Country"])
        ws.write(r4 + 2 + i, 1, float(row["monetary"]), fmt["money0"])
    bar3 = wb.add_chart({"type": "bar"})
    bar3.add_series({
        "name": "Net £",
        "categories": ["Graphs and Charts", r4 + 2, 0, r4 + 1 + len(by_cty), 0],
        "values": ["Graphs and Charts", r4 + 2, 1, r4 + 1 + len(by_cty), 1],
        "fill": {"color": ACCENT},
    })
    bar3.set_title({"name": "Top 10 countries by identified historical net"})
    bar3.set_size({"width": 720, "height": 280})
    bar3.set_legend({"none": True})
    ws.insert_chart(r4, 4, bar3)


def _rules(wb: xlsxwriter.Workbook, fmt: dict[str, Any], rules: pd.DataFrame) -> None:
    ws = wb.add_worksheet("RFM_Rules")
    _banner(ws, fmt, "RFM_Rules", 8)
    ws.set_column("A:A", 22)
    ws.set_column("B:B", 12)
    ws.set_column("C:C", 42)
    ws.set_column("D:G", 36)
    headers = ["Segment", "Sort", "Rule", "Description", "Recommended action", "Primary KPI", "Colour"]
    for c, h in enumerate(headers):
        ws.write(1, c, h, fmt["header"])
    texts = {
        "Champions": "R>=4 AND F>=4 AND M>=4",
        "Loyal Customers": "R>=3 AND F>=3",
        "New Customers": "tenure_days<=60 AND F<=2",
        "Potential Loyalists": "R>=3 AND F<=2 AND M>=3",
        "Needs Attention": "R>=3 AND F<=2",
        "Can't Lose Them": "R<=2 AND F>=4 AND M>=4",
        "At-Risk Customers": "R<=2 AND F>=3",
        "Lapsed High Spenders": "R<=2 AND F<=2 AND M>=4",
        "Hibernating": "R=2 AND F<=2",
        "Lost Customers": "R=1 AND F<=2",
    }
    for i, rule in enumerate(cfg_rows(rules)):
        r = 2 + i
        ws.write(r, 0, rule["name"])
        ws.write(r, 1, rule["sort_order"])
        ws.write(r, 2, texts[rule["name"]], fmt["calc_text"])
        ws.write(r, 3, rule["description"], fmt["wrap"])
        ws.write(r, 4, rule["action"], fmt["wrap"])
        ws.write(r, 5, rule["primary_kpi"])
        ws.write(r, 6, rule["colour"])
        ws.set_row(r, 36)
    last = 1 + len(rules)
    ws.add_table(1, 0, last, 6, {
        "name": "tblRules",
        "columns": [{"header": h} for h in headers],
        "style": "TableStyleMedium2",
    })
    ws.write(last + 2, 0, "Frequency bands", fmt["section"])
    ws.write(last + 3, 0, "Orders", fmt["header"])
    ws.write(last + 3, 1, "F score", fmt["header"])
    for i, (lab, sc) in enumerate([("1", 1), ("2", 2), ("3-4", 3), ("5-9", 4), ("10+", 5)]):
        ws.write(last + 4 + i, 0, lab)
        ws.write(last + 4 + i, 1, sc)


def cfg_rows(rules: pd.DataFrame) -> list[dict[str, Any]]:
    return rules.sort_values("sort_order").to_dict(orient="records")


def _customer_rfm(wb: xlsxwriter.Workbook, fmt: dict[str, Any], customers: pd.DataFrame) -> None:
    ws = wb.add_worksheet("Customer_RFM")
    _banner(ws, fmt, "Customer_RFM", 16)
    cols = [
        "CustomerID", "Country", "first_purchase_date", "last_purchase_date",
        "recency_days", "tenure_days", "frequency", "monetary", "avg_order_value",
        "distinct_products", "r_score", "f_score", "m_score",
        "RFM_Code", "Segment", "cluster_name", "Recommended_Action",
    ]
    for c, h in enumerate(cols):
        ws.write(1, c, h, fmt["header"])
    n = len(customers)
    ws.set_column(0, 0, 14)
    ws.set_column(1, 3, 18)
    ws.set_column(4, 12, 14)
    ws.set_column(13, 16, 22)
    customers = customers.sort_values("CustomerID").reset_index(drop=True)
    for i, row in customers.iterrows():
        r = 2 + int(i)
        ws.write(r, 0, int(row["CustomerID"]))
        ws.write(r, 1, row["Country"])
        ws.write(r, 2, str(row["first_purchase_date"])[:10])
        ws.write(r, 3, str(row["last_purchase_date"])[:10])
        ws.write_number(r, 4, int(row["recency_days"]))
        ws.write_number(r, 5, int(row["tenure_days"]))
        ws.write_number(r, 6, int(row["frequency"]))
        ws.write_number(r, 7, float(row["monetary"]))
        ws.write_number(r, 8, float(row["avg_order_value"]))
        ws.write_number(r, 9, int(row["distinct_products"]) if pd.notna(row["distinct_products"]) else 0)
        ws.write_number(r, 10, int(row["r_score"]))
        ws.write_number(r, 11, int(row["f_score"]))
        ws.write_number(r, 12, int(row["m_score"]))
        excel_r = r + 1
        rfm_code = int(row["r_score"]) * 100 + int(row["f_score"]) * 10 + int(row["m_score"])
        action = str(row["action"]) if pd.notna(row.get("action")) else ""
        ws.write_formula(r, 13, f"=K{excel_r}*100+L{excel_r}*10+M{excel_r}", None, rfm_code)
        ws.write_formula(r, 14, _segment_ifs(excel_r), None, str(row["segment"]))
        ws.write(r, 15, row["cluster_name"])
        ws.write_formula(r, 16, f'=IFERROR(VLOOKUP(O{excel_r},tblRules,5,FALSE),"")', None, action)
    last = 1 + n
    ws.add_table(1, 0, last, 16, {
        "name": "tblCustomer",
        "columns": [{"header": h} for h in cols],
        "style": "TableStyleMedium9",
    })
    ws.freeze_panes(2, 1)


def _segment_ifs(r: int) -> str:
    """Excel IFS matching config order. K=R, L=F, M=M, F_col=tenure (F{r} is tenure_days)."""
    return (
        f'=IFS(AND(K{r}>=4,L{r}>=4,M{r}>=4),"Champions",'
        f'AND(K{r}>=3,L{r}>=3),"Loyal Customers",'
        f'AND(F{r}<=60,L{r}<=2),"New Customers",'
        f'AND(K{r}>=3,L{r}<=2,M{r}>=3),"Potential Loyalists",'
        f'AND(K{r}>=3,L{r}<=2),"Needs Attention",'
        f'AND(K{r}<=2,L{r}>=4,M{r}>=4),"Can\'t Lose Them",'
        f'AND(K{r}<=2,L{r}>=3),"At-Risk Customers",'
        f'AND(K{r}<=2,L{r}<=2,M{r}>=4),"Lapsed High Spenders",'
        f'AND(K{r}=2,L{r}<=2),"Hibernating",'
        f'AND(K{r}=1,L{r}<=2),"Lost Customers")'
    )


def _segment_summary(wb: xlsxwriter.Workbook, fmt: dict[str, Any], segments: pd.DataFrame) -> None:
    ws = wb.add_worksheet("Segment_Summary")
    _banner(ws, fmt, "Segment_Summary", 8)
    ws.set_column("A:A", 22)
    ws.set_column("B:H", 16)
    headers = [
        "Segment", "Customers", "Revenue", "Pct_Customers", "Pct_Revenue",
        "Mean_Recency", "Mean_Frequency", "AOV",
    ]
    for c, h in enumerate(headers):
        ws.write(1, c, h, fmt["header"])
    segments = segments.sort_values("sort_order").reset_index(drop=True)
    n = len(segments)
    for i, s in segments.iterrows():
        r = 2 + int(i)
        ws.write(r, 0, s["segment"])
        ws.write_number(r, 1, int(s["customers"]), fmt["int"])
        ws.write_number(r, 2, float(s["revenue"]), fmt["money0"])
        ws.write_number(r, 3, float(s["pct_customers"]), fmt["pct"])
        ws.write_number(r, 4, float(s["pct_revenue"]), fmt["pct"])
        ws.write_number(r, 5, float(s["mean_recency"]))
        ws.write_number(r, 6, float(s["mean_frequency"]))
        ws.write_number(r, 7, float(s["aov"]), fmt["money"])
    last = 1 + n
    ws.add_table(1, 0, last, 7, {
        "name": "tblSegment",
        "columns": [{"header": h} for h in headers],
        "style": "TableStyleMedium2",
    })
    ws.write(last + 2, 0, "Customers")
    ws.write_number(last + 2, 1, int(segments["customers"].sum()), fmt["int"])
    ws.write(last + 3, 0, "Revenue")
    ws.write_number(last + 3, 1, float(segments["revenue"].sum()), fmt["money0"])


def _cleaned_view(fact: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
        "UnitPrice", "CustomerID", "Country", "line_type", "net_quantity",
        "net_revenue", "is_countable_order", "is_partial_month",
    ]
    out = fact.loc[:, [c for c in cols if c in fact.columns]].copy()
    return out


def _write_dataframe(
    wb: xlsxwriter.Workbook,
    fmt: dict[str, Any],
    name: str,
    df: pd.DataFrame,
) -> None:
    ws = wb.add_worksheet(name[:31])
    last_col = min(12, max(4, len(df.columns) - 1))
    _banner(ws, fmt, name.replace("_", " "), last_col)
    start_row = 1
    headers = list(df.columns)
    for c, h in enumerate(headers):
        ws.write(start_row, c, str(h), fmt["header"])
        ws.set_column(c, c, min(22, max(12, len(str(h)) + 2)))
    date_cols = {i for i, h in enumerate(headers) if "date" in str(h).lower() or str(h) in {"Date", "month_start"}}
    bool_cols = {i for i, h in enumerate(headers) if str(h).lower().startswith("is_")}
    for i, row in enumerate(df.itertuples(index=False), start=1):
        r = start_row + i
        for c, val in enumerate(row):
            if val is None:
                continue
            try:
                if pd.isna(val):
                    continue
            except (ValueError, TypeError):
                pass
            if c in date_cols:
                ts = pd.to_datetime(val, errors="coerce")
                if pd.isna(ts):
                    ws.write(r, c, str(val) if val is not None else None)
                elif getattr(ts, "hour", 0) or getattr(ts, "minute", 0):
                    ws.write_datetime(r, c, ts.to_pydatetime(), fmt["datetime"])
                else:
                    ws.write_datetime(r, c, ts.to_pydatetime(), fmt["date"])
            elif c in bool_cols:
                ws.write_boolean(r, c, bool(val))
            elif isinstance(val, bool):
                ws.write_boolean(r, c, val)
            elif isinstance(val, numbers.Real) and not isinstance(val, bool):
                ws.write_number(r, c, float(val))
            else:
                text = str(val)
                if len(text) > 32767:
                    text = text[:32767]
                ws.write_string(r, c, text)
        if i % 100000 == 0:
            logger.info("%s wrote %s rows", name, f"{i:,}")
    logger.info("%s complete (%s rows)", name, f"{len(df):,}")
