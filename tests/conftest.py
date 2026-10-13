"""Shared fixtures: the parsed GE Vernova Wind dataset (rebuilt from the workbook if the cache is missing)."""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from intel import common  # noqa: E402
from intel.bundle import build_intel  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    cache = ROOT / "data_cache" / "ge_parsed_data.pkl"
    if not cache.exists():  # data_cache is not committed when the workbook is absent; parse with the app's own loader
        import app  # noqa: PLC0415

        previous = os.getcwd()
        os.chdir(ROOT)
        try:
            wb = app.find_default_workbook()
            economy = app.parse_economy_sheet(wb)
            orders, platforms, _ = app.parse_oi_sheets(wb)
            orders, platforms = app.merge_press_orders(orders, platforms)
        finally:
            os.chdir(previous)
        return {"economy": economy, "orders": orders, "platforms": platforms}
    with open(cache, "rb") as fh:
        return pickle.load(fh)


@pytest.fixture(scope="session")
def orders(raw):
    return common.prepare_orders(raw["orders"])


@pytest.fixture(scope="session")
def platforms(raw):
    return common.prepare_platforms(raw["platforms"])


@pytest.fixture(scope="session")
def results():
    previous = os.getcwd()
    os.chdir(ROOT)  # the loader uses a path relative to the working directory
    try:
        return common.load_quarterly_results()
    finally:
        os.chdir(previous)


@pytest.fixture(scope="session")
def intel(raw, results):
    market = pd.read_json(ROOT / "data" / "market_prices_monthly.json")
    return build_intel(raw["economy"], raw["orders"], raw["platforms"], None, market, results)
