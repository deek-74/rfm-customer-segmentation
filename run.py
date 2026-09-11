"""CLI: python run.py <stage>  — data → pipeline → Excel workbook + HTML dashboard."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rfm.config import load_config  # noqa: E402
from rfm.io import read_raw_excel, write_parquet  # noqa: E402

STAGES = ("setup-check", "data", "pipeline", "excel", "all")
PACKAGES = (
    "pandas",
    "numpy",
    "duckdb",
    "sklearn",
    "scipy",
    "statsmodels",
    "matplotlib",
    "plotly",
    "yaml",
    "openpyxl",
    "xlsxwriter",
    "pyarrow",
)

logger = logging.getLogger("rfm")


def setup_logging() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def stage_setup_check() -> None:
    missing = []
    for name in PACKAGES:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit(f"Missing packages: {missing}. Run: pip install -r requirements.txt")
    logger.info("setup-check: all %s packages import", len(PACKAGES))


def stage_data() -> None:
    cfg = load_config()
    path = cfg["paths"]["raw_xlsx"]
    if not path.exists():
        raise SystemExit(
            f"Raw Excel not found at {path}. Place 'Online Retail.xlsx' in data/raw/."
        )
    df = read_raw_excel(path)
    if df.shape != (541909, 8):
        raise SystemExit(f"Unexpected raw shape {df.shape}; expected (541909, 8).")
    write_parquet(df, cfg["paths"]["interim_raw_parquet"])
    logger.info("data: wrote %s", cfg["paths"]["interim_raw_parquet"])


def stage_pipeline() -> None:
    from rfm.pipeline import run_pipeline

    cfg = load_config()
    if not cfg["paths"]["interim_raw_parquet"].exists():
        stage_data()
    result = run_pipeline(cfg)
    m = result["v2"]["metrics"]
    logger.info(
        "pipeline done | v2 customers=%s net=%.0f",
        m["v2"]["n_customers"],
        m["v2"]["identified_customers_net_revenue"],
    )


def stage_excel() -> None:
    from rfm.excel_workbook import export_resume_workbook
    from rfm.html_dashboard import export_html_dashboard

    cfg = load_config()
    if not cfg["paths"]["metrics_json"].exists():
        raise SystemExit("metrics.json missing; run python run.py pipeline first.")
    xlsx = export_resume_workbook(cfg)
    html = export_html_dashboard(cfg)
    logger.info("workbook: %s", xlsx)
    logger.info("dashboard: %s", html)


def stage_all() -> None:
    stage_setup_check()
    stage_data()
    stage_pipeline()
    stage_excel()


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="RFM workbook + dashboard")
    parser.add_argument("stage", choices=STAGES, help="pipeline stage")
    args = parser.parse_args()
    {
        "setup-check": stage_setup_check,
        "data": stage_data,
        "pipeline": stage_pipeline,
        "excel": stage_excel,
        "all": stage_all,
    }[args.stage]()


if __name__ == "__main__":
    main()
