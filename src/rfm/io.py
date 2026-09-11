"""Read/write helpers. All paths are pathlib.Path objects from config."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

RAW_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_raw_excel(path: Path) -> pd.DataFrame:
    """Read UCI Online Retail with InvoiceNo/StockCode as strings and nullable CustomerID."""
    logger.info("Reading Excel from %s", path)
    df = pd.read_excel(
        path,
        engine="openpyxl",
        dtype={"InvoiceNo": "string", "StockCode": "string"},
    )
    expected = set(RAW_COLUMNS)
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Raw file missing columns: {sorted(missing)}")
    df = df.loc[:, RAW_COLUMNS].copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype("string")
    df["StockCode"] = df["StockCode"].astype("string")
    df["Description"] = df["Description"].astype("string")
    df["Country"] = df["Country"].astype("string")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce").astype("Int64")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    logger.info("Raw rows=%s cols=%s", f"{len(df):,}", df.shape[1])
    return df


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)
    logger.info("Wrote parquet %s (%s rows)", path, f"{len(df):,}")


def read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    logger.info("Read parquet %s (%s rows)", path, f"{len(df):,}")
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    logger.info("Wrote csv %s (%s rows)", path, f"{len(df):,}")


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def write_json(payload: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    logger.info("Wrote json %s", path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(text: str, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote text %s", path)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)
