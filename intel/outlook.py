"""Momentum, announced-vs-booked comparison, guidance arithmetic and timing analytics (pure pandas).

Two order measures are used:
  * announced MW - orders from the yearly sheets and press releases. Releases are dated by announcement but often say the
    order was booked in an earlier quarter, so the booking quarter is used where it is stated;
  * Wind segment orders - GE Vernova's reported orders in USD per calendar quarter (equipment and services, onshore and offshore).
The two are compared through an indicative "USD per announced MW" ratio, not reconciled one-to-one.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import GIGA_MW, QUARTER_END_WINDOW_DAYS, as_of_date, pct_change, same_day

MIN_RATIO_MW = 300.0  # below this many announced MW a USD-per-MW ratio is not meaningful


# --------------------------------------------------------------------------------------
# Quarterly results
# --------------------------------------------------------------------------------------
def results_table(results: dict[str, Any] | None) -> pd.DataFrame:
    """Reported Wind-segment quarters with margins, year-on-year changes and rolling four-quarter sums."""
    if not results:
        return pd.DataFrame()
    q = results["quarterly"].copy()
    q["ebitda_margin"] = q["ebitda_usd_m"] / q["revenue_usd_m"]
    q["book_to_bill"] = q["orders_usd_m"] / q["revenue_usd_m"]
    for col in ("orders_usd_m", "revenue_usd_m", "ebitda_usd_m"):
        q[f"ltm_{col}"] = q[col].rolling(4).sum()
        q[f"{col}_yoy"] = q[col].pct_change(4, fill_method=None)
    q["ltm_book_to_bill"] = q["ltm_orders_usd_m"] / q["ltm_revenue_usd_m"]
    q["ltm_ebitda_margin"] = q["ltm_ebitda_usd_m"] / q["ltm_revenue_usd_m"]
    q["ebitda_change_yoy_usd_m"] = q["ebitda_usd_m"] - q["ebitda_usd_m"].shift(4)
    return q


def year_table(res: pd.DataFrame, results: dict[str, Any] | None) -> pd.DataFrame:
    """Calendar-year orders, revenue and EBITDA: the stated totals where available, else the sum of the quarters."""
    if res.empty:
        return pd.DataFrame()
    totals = (results or {}).get("fy_totals", {})
    rows = []
    for year, g in res.groupby("year"):
        row: dict[str, Any] = {"year": int(year), "quarters": int(len(g))}
        for col in ("orders_usd_m", "revenue_usd_m", "ebitda_usd_m"):
            stated = totals.get(int(year), {}).get(col, np.nan)
            row[col] = float(stated) if len(g) == 4 and pd.notna(stated) else float(g[col].sum())
        rows.append(row)
    for year, t in totals.items():  # years with stated totals but no quarterly series (FY2023)
        if year not in {r["year"] for r in rows}:
            rows.append({"year": int(year), "quarters": 0, **{k: float(v) for k, v in t.items() if k.endswith("usd_m")}})
    t = pd.DataFrame(rows).set_index("year").sort_index()
    t["complete"] = t["quarters"].isin([0, 4])
    t["ebitda_margin"] = t["ebitda_usd_m"] / t["revenue_usd_m"]
    t["book_to_bill"] = t["orders_usd_m"] / t["revenue_usd_m"]
    return t


def half_year_table(res: pd.DataFrame) -> pd.DataFrame:
    if res.empty:
        return pd.DataFrame()
    r = res.copy()
    r["half"] = np.where(r["quarter"] <= 2, "H1", "H2")
    g = r.groupby(["year", "half"])
    t = g[["orders_usd_m", "revenue_usd_m", "ebitda_usd_m"]].sum()
    t["quarters"] = g.size()
    t = t[t["quarters"] == 2].drop(columns="quarters")
    t["ebitda_margin"] = t["ebitda_usd_m"] / t["revenue_usd_m"]
    return t


# --------------------------------------------------------------------------------------
# Announced vs booked
# --------------------------------------------------------------------------------------
def announced_by_booking_quarter(orders: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """Announced MW per reported quarter, by booking quarter (stated, else announcement), next to reported Wind orders."""
    if res.empty:
        return pd.DataFrame()
    key = ["book_year", "book_quarter"]
    tot = orders.groupby(key)["size_mw"].sum()
    cnt = orders.groupby(key).size()
    ann = orders.groupby(["order_year", "order_quarter"])["size_mw"].sum()
    out = res[["year", "quarter", "label", "orders_usd_m"]].copy()
    idx = list(zip(out["year"], out["quarter"]))
    out["announced_mw_booked"] = [float(tot.get(k, 0.0)) for k in idx]
    out["announced_mw_by_announcement"] = [float(ann.get(k, 0.0)) for k in idx]
    out["announced_orders"] = [int(cnt.get(k, 0)) for k in idx]
    out["usd_m_per_announced_mw"] = (out["orders_usd_m"] / out["announced_mw_booked"]).where(out["announced_mw_booked"] >= MIN_RATIO_MW)
    return out


def announced_by_year(orders: pd.DataFrame, yt: pd.DataFrame, meta: dict[str, Any] | None = None) -> pd.DataFrame:
    t = orders.groupby("book_year").agg(announced_mw=("size_mw", "sum"), announced_orders=("size_mw", "size"))
    t.index.name = "year"
    unlisted = {int(x["year"]): float(x["mw"]) for x in (meta or {}).get("unlisted_bookings") or []}
    t["unlisted_mw"] = pd.Series(unlisted).reindex(t.index).fillna(0.0)
    if not yt.empty:
        t = t.join(yt[["orders_usd_m", "quarters"]].rename(columns={"orders_usd_m": "reported_orders_usd_m"}), how="left")
        t["usd_m_per_announced_mw"] = (t["reported_orders_usd_m"] / t["announced_mw"]).where(t["announced_mw"] >= MIN_RATIO_MW)
        t["usd_m_per_mw_incl_unlisted"] = (t["reported_orders_usd_m"] / (t["announced_mw"] + t["unlisted_mw"])).where(t["announced_mw"] >= MIN_RATIO_MW)
    return t


# --------------------------------------------------------------------------------------
# Momentum on announced orders
# --------------------------------------------------------------------------------------
def ytd_table(orders: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    as_of = as_of or as_of_date(orders)
    rows = []
    for year in range(int(orders["order_year"].min()), as_of.year + 1):
        cut = same_day(year, as_of)
        y = orders[orders["order_year"] == year]
        ytd = y[y["order_date"] <= cut]
        rows.append(
            {
                "year": year,
                "ytd_mw": float(ytd["size_mw"].sum()),
                "ytd_orders": int(len(ytd)),
                "ytd_mw_ex_giga": float(ytd.loc[ytd["size_mw"] < GIGA_MW, "size_mw"].sum()),
                "ytd_giga_orders": int((ytd["size_mw"] >= GIGA_MW).sum()),
                "full_mw": float(y["size_mw"].sum()) if year < as_of.year else np.nan,
                "full_orders": int(len(y)) if year < as_of.year else np.nan,
            }
        )
    t = pd.DataFrame(rows).set_index("year")
    t["ytd_share_of_full"] = t["ytd_mw"] / t["full_mw"]
    t["ytd_rank"] = t["ytd_mw"].rank(ascending=False, method="min")
    return t


def ltm_summary(orders: pd.DataFrame, res: pd.DataFrame | None = None, as_of: pd.Timestamp | None = None) -> dict[str, Any]:
    as_of = as_of or as_of_date(orders)
    start = as_of - pd.DateOffset(years=1)
    prior_start = as_of - pd.DateOffset(years=2)
    cur = orders[(orders["order_date"] > start) & (orders["order_date"] <= as_of)]
    prev = orders[(orders["order_date"] > prior_start) & (orders["order_date"] <= start)]

    def stats(frame: pd.DataFrame) -> dict[str, float]:
        return {
            "mw": float(frame["size_mw"].sum()),
            "orders": int(len(frame)),
            "avg_mw": float(frame["size_mw"].mean()) if len(frame) else np.nan,
            "median_mw": float(frame["size_mw"].median()) if len(frame) else np.nan,
            "mw_ex_giga": float(frame.loc[frame["size_mw"] < GIGA_MW, "size_mw"].sum()),
            "giga_orders": int((frame["size_mw"] >= GIGA_MW).sum()),
            "countries": int(frame.loc[~frame["multi_country"], "country_clean"].nunique()),
        }

    c, p = stats(cur), stats(prev)
    out: dict[str, Any] = {"as_of": as_of, "window": (start, as_of), "current": c, "prior": p,
                           "mw_growth": pct_change(c["mw"], p["mw"]), "orders_growth": pct_change(c["orders"], p["orders"]), "reported": None}
    if res is not None and len(res) >= 8:
        cur4, prior4 = res.iloc[-4:], res.iloc[-8:-4]
        out["reported"] = {
            "orders": float(cur4["orders_usd_m"].sum()), "prior_orders": float(prior4["orders_usd_m"].sum()),
            "orders_growth": pct_change(cur4["orders_usd_m"].sum(), prior4["orders_usd_m"].sum()),
            "revenue": float(cur4["revenue_usd_m"].sum()), "prior_revenue": float(prior4["revenue_usd_m"].sum()),
            "revenue_growth": pct_change(cur4["revenue_usd_m"].sum(), prior4["revenue_usd_m"].sum()),
            "ebitda": float(cur4["ebitda_usd_m"].sum()), "prior_ebitda": float(prior4["ebitda_usd_m"].sum()),
            "book_to_bill": float(cur4["orders_usd_m"].sum() / cur4["revenue_usd_m"].sum()),
            "first_label": str(cur4.iloc[0]["label"]), "last_label": str(cur4.iloc[-1]["label"]),
        }
    return out


def cumulative_curves(orders: pd.DataFrame, as_of: pd.Timestamp | None = None, first_year: int = 2020) -> pd.DataFrame:
    as_of = as_of or as_of_date(orders)
    frames = []
    for year in range(first_year, as_of.year + 1):
        end = as_of if year == as_of.year else pd.Timestamp(year=year, month=12, day=31)
        days = pd.date_range(pd.Timestamp(year=year, month=1, day=1), end, freq="D")
        daily = orders[orders["order_year"] == year].groupby(orders["order_date"].dt.normalize())["size_mw"].sum().reindex(days, fill_value=0.0)
        frame = pd.DataFrame({"year": year, "date": days, "cum_mw": daily.cumsum().to_numpy()})
        frame["plot_date"] = frame["date"].map(lambda d: d.replace(year=2024))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# Guidance arithmetic
# --------------------------------------------------------------------------------------
def project_year(results: dict[str, Any] | None, res: pd.DataFrame, lookback: int = 8) -> dict[str, Any]:
    """What the rest of the year must deliver for GE Vernova's Wind guidance, and a scenario for orders (not a forecast).

    Guidance (as stated): organic revenue down low double digits and about USD 400m of segment EBITDA losses. 'Low double digits'
    is read as a decline of 10-13% for the arithmetic below, an interpretation that is flagged in the app.
    """
    if res.empty:
        return {}
    last = res.iloc[-1]
    year, n = int(last["year"]), int(last["quarter"])
    cur = res[res["year"] == year]
    prev = res[res["year"] == year - 1]
    remaining = 4 - n
    if remaining <= 0 or len(prev) < 4:
        return {}
    guid = (results or {}).get("guidance", {})
    prev_full = {c: float(prev[c].sum()) for c in ("orders_usd_m", "revenue_usd_m", "ebitda_usd_m")}
    ytd = {c: float(cur[c].sum()) for c in ("orders_usd_m", "revenue_usd_m", "ebitda_usd_m")}
    prev_rem = prev[prev["quarter"] > n]
    prev_ytd = prev[prev["quarter"] <= n]
    hist = res.iloc[-lookback:]["orders_usd_m"]
    q_low, q_med, q_high = float(hist.min()), float(hist.median()), float(hist.max())
    out: dict[str, Any] = {
        "year": year, "quarters_reported": n, "remaining": remaining, "label": str(last["label"]),
        "ytd": ytd, "prev_full": prev_full, "prev_ytd": {c: float(prev_ytd[c].sum()) for c in ytd},
        "prev_remaining": {c: float(prev_rem[c].sum()) for c in ytd},
        "orders_scenario": {"conservative": ytd["orders_usd_m"] + remaining * q_low, "base": ytd["orders_usd_m"] + remaining * q_med, "strong": ytd["orders_usd_m"] + remaining * q_high},
        "quarter_orders": {"low": q_low, "median": q_med, "high": q_high, "lookback": int(len(hist))},
        "guidance": guid,
    }
    if guid.get("ebitda_usd_m") is not None:
        out["ebitda_needed_remaining"] = float(guid["ebitda_usd_m"]) - ytd["ebitda_usd_m"]
        out["ebitda_prev_remaining"] = out["prev_remaining"]["ebitda_usd_m"]
    lo, hi = guid.get("revenue_decline_assumed", [-0.13, -0.10])
    full_lo, full_hi = prev_full["revenue_usd_m"] * (1 + lo), prev_full["revenue_usd_m"] * (1 + hi)
    out["revenue_full_year"] = (full_lo, full_hi)
    out["revenue_needed_remaining"] = (full_lo - ytd["revenue_usd_m"], full_hi - ytd["revenue_usd_m"])
    return out


# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------
def quarter_end_overall(orders: pd.DataFrame, first_year: int = 2020) -> dict[str, float]:
    o = orders[orders["date_known"] & (orders["order_year"] >= first_year)]
    expected = QUARTER_END_WINDOW_DAYS / 91.25
    q_start = o["order_date"].dt.to_period("Q").dt.start_time
    early = (o["order_date"].dt.normalize() - q_start).dt.days < QUARTER_END_WINDOW_DAYS
    return {
        "mw_share": float(o.loc[o["in_quarter_end_window"], "size_mw"].sum() / o["size_mw"].sum()),
        "orders_share": float(o["in_quarter_end_window"].mean()),
        "early_mw_share": float(o.loc[early, "size_mw"].sum() / o["size_mw"].sum()),
        "expected_if_uniform": float(expected), "window_days": QUARTER_END_WINDOW_DAYS,
        "first_year": first_year, "years": int(o["order_year"].nunique()), "orders": int(len(o)),
        "booked_earlier_share": float(o.loc[o["booked_earlier"], "size_mw"].sum() / o["size_mw"].sum()),
    }


def quarter_end_share_by_year(orders: pd.DataFrame) -> pd.DataFrame:
    g = orders.groupby("order_year")
    mw_total = g["size_mw"].sum()
    mw_qe = orders[orders["in_quarter_end_window"]].groupby("order_year")["size_mw"].sum().reindex(mw_total.index).fillna(0.0)
    n_total = g.size()
    n_qe = orders[orders["in_quarter_end_window"]].groupby("order_year").size().reindex(n_total.index).fillna(0)
    out = pd.DataFrame({"mw_share": mw_qe / mw_total, "orders_share": n_qe / n_total})
    out.index.name = "year"
    return out


def month_in_quarter_share(orders: pd.DataFrame, first_year: int = 2020) -> pd.Series:
    o = orders[orders["order_year"] >= first_year]
    pos = ((o["order_date"].dt.month - 1) % 3) + 1
    mw = o.groupby(pos)["size_mw"].sum()
    return (mw / mw.sum()).rename("share")
