"""Rebuild data_cache/ge_parsed_data.pkl from the workbook (default: GE_data.xls found by the app) plus the press-release orders.

Usage: python tools/rebuild_cache.py [workbook.xls]
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os  # noqa: E402

import pandas as pd  # noqa: E402

os.chdir(ROOT)  # the app uses paths relative to the project folder
import app  # noqa: E402


def main() -> None:
    wb = Path(sys.argv[1]) if len(sys.argv) > 1 else app.find_default_workbook()
    if wb is None:
        raise SystemExit("no workbook found")
    economy = app.parse_economy_sheet(wb)
    orders, platforms, unannounced = app.parse_oi_sheets(wb)
    n_wb = len(orders)
    orders, platforms = app.merge_press_orders(orders, platforms)
    st = wb.stat()
    sup = app.SUPPLEMENT_FILE.stat().st_mtime_ns if app.SUPPLEMENT_FILE.exists() else 0
    payload = {
        "cache_key": f"{wb.resolve()}::{st.st_mtime_ns}::{st.st_size}::{sup}",
        "workbook_path": str(wb.resolve()),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "economy": economy, "orders": orders, "platforms": platforms, "unannounced": unannounced,
    }
    out = ROOT / "data_cache" / "ge_parsed_data.pkl"
    pd.to_pickle(payload, out)
    print(f"{wb.name}: {n_wb} workbook orders + {len(orders) - n_wb} press-release orders -> {out}")
    print(orders.groupby("order_year").agg(n=("size_mw", "size"), mw=("size_mw", "sum")).to_string())


if __name__ == "__main__":
    main()
