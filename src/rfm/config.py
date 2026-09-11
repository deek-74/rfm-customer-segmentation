"""Load config/config.yaml and resolve paths relative to the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

_PATH_KEYS = {
    "raw_xlsx",
    "interim_raw_parquet",
    "interim_staged_parquet",
    "processed_dir",
    "outputs_dir",
    "tables_dir",
    "figures_dir",
    "metrics_json",
    "v1_parity_report",
    "data_quality_report",
    "reference_dir",
    "sql_dir",
    "powerbi_model_dir",
    "resume_workbook",
    "html_dashboard",
    "v1_customer_table",
    "v1_segment_summary",
    "v1_cluster_profile",
}


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Return the YAML config with path values resolved against the repo root."""
    cfg_path = path or CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_project_root"] = PROJECT_ROOT
    cfg["_config_path"] = cfg_path
    for key, value in list(cfg.get("paths", {}).items()):
        if key in _PATH_KEYS and isinstance(value, str):
            cfg["paths"][key] = PROJECT_ROOT / value
    return cfg
