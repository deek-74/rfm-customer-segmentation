"""Export Power BI model CSVs, DAX, theme, and the desktop build guide."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from rfm.io import write_csv, write_text
from rfm.segments import segment_lookup


def export_powerbi(
    *,
    cfg: dict[str, Any],
    fact: pd.DataFrame,
    dim_customer: pd.DataFrame,
    product_summary: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    retention: pd.DataFrame,
    migration: pd.DataFrame,
    winback: pd.DataFrame,
    k_metrics: pd.DataFrame,
    monthly: pd.DataFrame,
) -> None:
    model_dir: Path = cfg["paths"]["powerbi_model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    fact_out = _fact_sales(fact)
    write_csv(fact_out, model_dir / "fact_sales.csv")
    write_csv(dim_customer, model_dir / "dim_customer.csv")
    write_csv(_dim_product(fact, cfg), model_dir / "dim_product.csv")
    write_csv(_dim_date(fact, monthly), model_dir / "dim_date.csv")
    write_csv(segment_lookup(cfg).rename(columns={"segment": "Segment"}), model_dir / "dim_segment.csv")
    write_csv(cluster_profile, model_dir / "cluster_profile.csv")
    write_csv(retention, model_dir / "cohort_retention.csv")
    write_csv(migration, model_dir / "segment_migration.csv")
    write_csv(winback, model_dir / "winback_scenarios.csv")
    write_csv(k_metrics, model_dir / "k_selection_metrics.csv")

    write_text(_measures_dax(cfg), cfg["_project_root"] / "powerbi" / "measures.dax")
    write_text(_theme_json(cfg), cfg["_project_root"] / "powerbi" / "theme.json")
    write_text(_build_guide(), cfg["_project_root"] / "powerbi" / "BUILD_GUIDE.md")


def _fact_sales(fact: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "InvoiceNo": fact["InvoiceNo"],
            "InvoiceDate": fact["InvoiceDate"],
            "Date": pd.to_datetime(fact["Date"]).dt.strftime("%Y-%m-%d"),
            "StockCode": fact["StockCode"],
            "CustomerID": fact["CustomerID"],
            "Country": fact["Country"],
            "Quantity": fact["Quantity"],
            "NetQuantity": fact["net_quantity"],
            "UnitPrice": fact["UnitPrice"],
            "NetRevenue": fact["net_revenue"],
            "LineType": fact["line_type"],
            "IsCountableOrder": fact["is_countable_order"].astype(bool),
            "IsPartialMonth": fact["is_partial_month"].astype(bool),
        }
    )
    return out


def _dim_product(fact: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    codes = {c.upper() for c in cfg["cleaning"]["non_product_stock_codes"]}
    desc = (
        fact.sort_values("InvoiceDate")
        .groupby("StockCode", as_index=False)
        .agg(Description=("Description", "last"))
    )
    non_product_skus = set(
        fact.loc[fact["line_type"] == "non_product", "StockCode"].astype(str).unique()
    )
    listed = desc["StockCode"].astype(str).str.strip().str.upper().isin(codes)
    desc["IsNonProduct"] = listed | desc["StockCode"].astype(str).isin(non_product_skus)
    return desc[["StockCode", "Description", "IsNonProduct"]]


def _dim_date(fact: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    start = pd.to_datetime(fact["Date"]).min().normalize()
    end = pd.to_datetime(fact["Date"]).max().normalize()
    days = pd.date_range(start, end, freq="D")
    partial = set(pd.to_datetime(monthly.loc[monthly["is_partial_month"], "month_start"]))
    df = pd.DataFrame({"Date": days})
    df["MonthStart"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df["MonthYear"] = df["Date"].dt.strftime("%b %Y")
    df["MonthSort"] = df["Date"].dt.year * 100 + df["Date"].dt.month
    df["IsPartialMonth"] = df["MonthStart"].isin(partial)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["MonthStart"] = df["MonthStart"].dt.strftime("%Y-%m-%d")
    return df


def _measures_dax(cfg: dict[str, Any]) -> str:
    names = cfg["winback"]["at_risk_segment_names"]
    names_dax = ", ".join(f'"{n}"' for n in names)
    orders_per = float(cfg["winback"]["expected_orders_per_reactivated"])
    return f'''-- RFM v2 measures. Load pipeline CSVs only (never the raw Excel).
-- Create a what-if parameter named Winback Rate (decimal, 0.00 to 0.30, step 0.01).

-- Historical net revenue of product sale lines with positive net.
Total Revenue =
CALCULATE(
    SUM(fact_sales[NetRevenue]),
    fact_sales[LineType] = "sale"
)

Orders =
CALCULATE(
    DISTINCTCOUNT(fact_sales[InvoiceNo]),
    fact_sales[LineType] = "sale",
    fact_sales[IsCountableOrder] = TRUE()
)

Customers =
CALCULATE(
    DISTINCTCOUNT(dim_customer[CustomerID])
)

AOV =
DIVIDE([Total Revenue], [Orders])

Revenue per Customer =
DIVIDE([Total Revenue], [Customers])

-- Repeat purchase rate uses the customer table (one row per customer), not a visual count.
Repeat Purchase Rate =
VAR Repeaters =
    CALCULATE(
        DISTINCTCOUNT(dim_customer[CustomerID]),
        dim_customer[frequency] > 1
    )
RETURN
    DIVIDE(Repeaters, [Customers])

Customers in Segment =
DISTINCTCOUNT(dim_customer[CustomerID])

Segment Revenue Share % =
DIVIDE(
    CALCULATE(SUM(dim_customer[monetary])),
    CALCULATE(SUM(dim_customer[monetary]), ALL(dim_segment))
)

Revenue at Risk =
CALCULATE(
    SUM(dim_customer[monetary]),
    dim_customer[segment] IN {{{names_dax}}}
)

-- Complete months only: filter out IsPartialMonth on the date table.
MoM Revenue Growth % =
VAR CurrentRev =
    CALCULATE([Total Revenue], dim_date[IsPartialMonth] = FALSE())
VAR PrevRev =
    CALCULATE(
        [Total Revenue],
        DATEADD(dim_date[Date], -1, MONTH),
        dim_date[IsPartialMonth] = FALSE()
    )
RETURN
    DIVIDE(CurrentRev - PrevRev, PrevRev)

Top 20% Revenue Share =
VAR N = COUNTROWS(dim_customer)
VAR Cutoff = CEILING(0.20 * N, 1)
VAR Ranked =
    ADDCOLUMNS(
        dim_customer,
        "Rk", RANKX(ALL(dim_customer), dim_customer[monetary], , DESC, DENSE)
    )
VAR TopRev =
    SUMX(FILTER(Ranked, [Rk] <= Cutoff), dim_customer[monetary])
RETURN
    DIVIDE(TopRev, SUM(dim_customer[monetary]))

-- What-if: 'Winback Rate'[Winback Rate Value] is the parameter table Power BI generates.
-- expected_orders_per_reactivated comes from config/config.yaml ({orders_per:g}).
Expected Win-back Revenue =
VAR Rate = SELECTEDVALUE('Winback Rate'[Winback Rate Value], 0.10)
VAR OrdersPer = {orders_per:g}
RETURN
    SUMX(
        FILTER(
            dim_customer,
            dim_customer[segment] IN {{{names_dax}}}
        ),
        Rate * dim_customer[avg_order_value] * OrdersPer
    )
'''


def _theme_json(cfg: dict[str, Any]) -> str:
    palette = [s["colour"] for s in sorted(cfg["segments"], key=lambda s: s["sort_order"])]
    data_colours = ",\n    ".join(f'"{c}"' for c in palette)
    return f'''{{
  "name": "RFM v2",
  "dataColors": [
    {data_colours}
  ],
  "background": "#FFFFFF",
  "foreground": "#1B2631",
  "tableAccent": "#1B4F72"
}}
'''


def _build_guide() -> str:
    return """# Power BI Desktop build guide (v2)

Do not load `data/raw/Online Retail.xlsx`. Every visual reads pipeline outputs under `powerbi/model/`.

## 1. Parameter and queries

1. Open Power BI Desktop → **Transform data**.
2. **Manage parameters** → New parameter:
   - Name: `ProjectRoot`
   - Type: Text
   - Current value: the folder that contains `run.py` (example `C:\\Users\\You\\rfm-customer-segmentation`). No trailing backslash.
3. For each CSV below, **New source → Blank query** → Advanced Editor:

```powerquery
let
    Source = Csv.Document(
        File.Contents(ProjectRoot & "\\powerbi\\model\\fact_sales.csv"),
        [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true])
in
    Promoted
```

Replace the file name for: `dim_customer.csv`, `dim_product.csv`, `dim_date.csv`, `dim_segment.csv`, `cluster_profile.csv`, `cohort_retention.csv`, `segment_migration.csv`, `winback_scenarios.csv`, `k_selection_metrics.csv`. `fact_sales.csv` is large (~50 MB) and is produced by `python run.py pipeline` (it is gitignored; generate it locally before opening the report).

4. Set types (Date for date columns, Decimal for revenue, Whole number for ids/scores, True/False for flags). Close & Apply.

## 2. Model

- Turn **File → Options → Data Load → Auto date/time** **off**.
- Mark `dim_date` as the date table (`Date` column).
- Sort `dim_date[MonthYear]` by `MonthSort`.
- Sort `dim_segment[Segment]` by `sort_order`.

Relationships (single direction, many-to-one):

| From | To |
|---|---|
| fact_sales[CustomerID] | dim_customer[CustomerID] |
| fact_sales[Date] | dim_date[Date] |
| fact_sales[StockCode] | dim_product[StockCode] |
| dim_customer[segment] | dim_segment[Segment] |

## 3. Measures

Paste `powerbi/measures.dax` into a new measure table. Create a **What-if parameter** named `Winback Rate` (Decimal, 0.00–0.30, increment 0.01).

## 4. Page 1 — Executive Overview

KPI cards: Total Revenue, Orders, Customers, AOV, Repeat Purchase Rate, Revenue at Risk.

Monthly revenue: line of `[Total Revenue]` by `dim_date[MonthStart]` with `IsPartialMonth = FALSE` as the trend; add a second series or annotation for the partial month (December 2011).

Top 10 countries: `fact_sales[Country]` vs `[Total Revenue]`.

Top 10 products: `dim_product[Description]` vs `[Total Revenue]`, visual-level filter `dim_product[IsNonProduct] = FALSE`.

Slicers: Country, `dim_date[Date]`.

## 5. Page 2 — Customer Segmentation

- Table from `dim_segment` plus measures `[Customers in Segment]`, `[Segment Revenue Share %]`.
- Clustered bar: `[Customers in Segment]` and `[Segment Revenue Share %]` by Segment (these are **measures**, not Count of a one-row table).
- Matrix: rows `dim_customer[r_score]`, columns `dim_customer[f_score]`, values Average of `monetary` and Count of `CustomerID`.
- Scatter: X = Recency (`recency_days`), Y = `monetary` (log axis). **Details = CustomerID** so each point is a customer, not a summed segment. Colour by Segment.
- Table of segment description, action, and primary KPI from `dim_segment`.

## 6. Page 3 — Customer Value & Revenue Optimization

- Revenue at Risk card.
- Win-back what-if slicer + `[Expected Win-back Revenue]`.
- `cluster_profile` table and a matrix of cluster name vs segment (from `dim_customer`).
- Cohort retention matrix (`cohort_month` × `months_since`, value `customer_retention`) with conditional formatting 0–1.
- Segment migration matrix (`segment_a` × `segment_b`, value `customers` or `row_pct`).
- Exportable table: `dim_customer` filtered to At-Risk Customers and Can't Lose Them.

## 7. Fixes from v1 (audit items 8 and 9)

- [ ] KPI cards use `[Total Revenue]` / `[Orders]` from **pipeline CSVs**, not the raw Excel. (Item 8)
- [ ] No visual uses Count on `dim_segment` as a stand-in for customer count. (Item 9)
- [ ] Scatter has **CustomerID in Details**; Y is customer monetary, not Sum of recency. (Item 9)
- [ ] Product charts filter `IsNonProduct = FALSE`.
- [ ] Paths go through `ProjectRoot`; nothing is hard-coded to `D:\\`.
"""
