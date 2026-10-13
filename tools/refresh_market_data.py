"""Refresh monthly GE Vernova stock / steel / copper JSON files from Yahoo Finance's chart endpoint.

Usage: python tools/refresh_market_data.py
Existing months are kept; the most recent months are overwritten with fresh values.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

DATA = Path(__file__).resolve().parent.parent / "data"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch(symbol: str) -> list[dict]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(url, params={"range": "max", "interval": "1mo"}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    offset = res["meta"].get("gmtoffset", 0)  # timestamps are exchange-local midnight
    rows = []
    for i, ts in enumerate(res["timestamp"]):
        d = datetime.fromtimestamp(ts + offset, tz=timezone.utc).strftime("%Y-%m-01")
        row = {k: q[k][i] for k in ("open", "high", "low", "close", "volume")}
        row = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
        rows.append({"date": d, **row, "adj_close": round(adj[i], 4) if adj and adj[i] is not None else row["close"]})
    return rows


def main() -> None:
    stock_path = DATA / "ge_stock_monthly.json"
    stock = {r["date"][:7] + "-01": {**r, "date": r["date"][:7] + "-01"} for r in json.loads(stock_path.read_text(encoding="utf-8-sig") or "[]")}
    n = 0
    for r in fetch("GEV"):
        if r["close"] is None:
            continue
        stock[r["date"]] = r
        n += 1
    stock_path.write_text(json.dumps([stock[k] for k in sorted(stock)], indent=2), encoding="utf-8")
    print(f"stock: {n} months merged, last {max(stock)}")

    mkt_path = DATA / "market_prices_monthly.json"
    mkt = {r["date"][:7] + "-01": {**r, "date": r["date"][:7] + "-01"} for r in json.loads(mkt_path.read_text(encoding="utf-8"))}  # month-start keys
    for sym, key in (("HRC=F", "steel_hrc_usd_per_short_ton"), ("HG=F", "copper_hg_usd_per_lb")):
        for r in fetch(sym):
            if r["close"] is None:
                continue
            mkt.setdefault(r["date"], {"date": r["date"]})[key] = round(r["close"], 4)
    mkt_path.write_text(json.dumps([mkt[k] for k in sorted(mkt)], indent=2), encoding="utf-8")
    print(f"market: last {max(mkt)}", mkt[max(mkt)])


if __name__ == "__main__":
    main()
