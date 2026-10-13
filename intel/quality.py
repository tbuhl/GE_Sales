"""Data coverage and integrity checks, so users know how far to trust each analysis."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .common import UNKNOWN, as_of_date
from .economics import economics_table

FIELD_LABELS = {
    "customer": "Customer named",
    "country": "Country known",
    "platform": "Platform identified",
    "turbines": "Turbine count stated",
    "service": "Service term stated",
}


def completeness_by_year(orders: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in orders.groupby("order_year"):
        total = g["size_mw"].sum()
        if total <= 0:
            continue

        def share(mask: pd.Series) -> float:
            return float(g.loc[mask, "size_mw"].sum() / total)

        rows.append(
            {
                "year": int(year),
                FIELD_LABELS["customer"]: share(g["customer"] != UNKNOWN),
                FIELD_LABELS["country"]: share(~g["multi_country"]),
                FIELD_LABELS["platform"]: share(g["primary_platform"] != UNKNOWN),
                FIELD_LABELS["turbines"]: share(g["turbines_qty_sum"].notna()),
                FIELD_LABELS["service"]: share(g["service_time_years"].notna()),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def _gap_groups(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rel = (orders["mw_gap"].abs() / orders["size_mw"]).where(orders["size_mw"] > 0)
    big = orders["mw_gap"].abs() > 1.0
    return orders[(rel > 0.05) & (rel <= 0.25) & big], orders[(rel > 0.25) & big]


def anomaly_summary(
    orders: pd.DataFrame,
    platforms: pd.DataFrame,
    res: pd.DataFrame,
    results: dict[str, Any] | None,
    economy: pd.DataFrame,
    yt: pd.DataFrame,
    ann_year: pd.DataFrame,
    stock: pd.DataFrame | None = None,
    market: pd.DataFrame | None = None,
    meta: dict[str, Any] | None = None,
) -> pd.DataFrame:
    as_of = as_of_date(orders)
    rows: list[dict[str, Any]] = []

    def add(check: str, severity: str, affected: Any, detail: str) -> None:
        rows.append({"Check": check, "Severity": severity, "Affected": str(affected), "Detail": detail})

    # --- announced list vs reported bookings -------------------------------------------------
    if not ann_year.empty and "usd_m_per_announced_mw" in ann_year:
        a = ann_year[ann_year["reported_orders_usd_m"].notna()]
        y_ref = a.dropna(subset=["usd_m_per_announced_mw"])
        if len(a) >= 2 and len(y_ref) >= 2:
            ref_year = int(y_ref["usd_m_per_announced_mw"].idxmin())
            ref = float(y_ref.loc[ref_year, "usd_m_per_announced_mw"])
            last = a.iloc[-1] if a.index[-1] < as_of.year else a.iloc[-2]
            ly = int(last.name)
            implied_cover = float(last["announced_mw"] + last["unlisted_mw"]) / (last["reported_orders_usd_m"] / ref)
            add("Announced MW covers a shrinking part of bookings", "High" if implied_cover < 0.5 else "Medium", f"{ly}: {last['announced_mw']:,.0f} MW listed",
                f"Reported Wind orders were USD {last['reported_orders_usd_m']:,.0f}m in {ly} against USD {a.loc[ref_year, 'reported_orders_usd_m']:,.0f}m in {ref_year}, but the list holds only {last['announced_mw']:,.0f} MW "
                f"(plus {last['unlisted_mw']:,.0f} MW of US repowers announced without detail) against {a.loc[ref_year, 'announced_mw']:,.0f} MW in {ref_year}. "
                f"At {ref_year}'s USD {ref:.2f}m per announced MW the list covers about {implied_cover:.0%} of {ly} orders; GE Vernova now announces fewer orders individually and sells more services.")
    unl = (meta or {}).get("unlisted_bookings") or []
    for u in unl:
        add("Bookings announced without detail", "Medium", f"{u['year']}: {u['mw']:,.0f} MW", u["what"] + ". They are not in the order list, so announced MW understates bookings.")

    # --- booking lag -----------------------------------------------------------------------------
    lag = orders[orders["booked_earlier"]]
    if len(lag):
        add("Announced after booking", "Info", f"{len(lag)} orders, {lag['size_mw'].sum():,.0f} MW",
            f"{len(lag)} orders were announced up to {int(((lag['order_year'] * 4 + lag['order_quarter']) - (lag['book_year'] * 4 + lag['book_quarter'])).max())} quarters after the quarter they were booked in "
            "(stated in the release). Booking-quarter analytics use the stated quarter; every other order is dated by announcement.")

    # --- reported results vs workbook -----------------------------------------------------------
    if not yt.empty and economy is not None and not economy.empty:
        e = economics_table(economy)
        diffs = []
        for year in e.index:
            if year in yt.index and yt.loc[year, "complete"]:
                for wb_col, yt_col in (("orders", "orders_usd_m"), ("revenue", "revenue_usd_m"), ("ebitda", "ebitda_usd_m")):
                    if pd.notna(e.loc[year, wb_col]) and pd.notna(yt.loc[year, yt_col]):
                        diffs.append((int(year), wb_col, float(yt.loc[year, yt_col]), float(e.loc[year, wb_col])))
        bad = [d for d in diffs if abs(d[2] - d[3]) > 0.01 * abs(d[3])]
        if bad:
            add("Workbook economy differs from the earnings releases", "Medium", "; ".join(f"{y} {c}" for y, c, _, _ in bad),
                "; ".join(f"{y} {c}: workbook {w:,.0f} vs release {r:,.0f} (USD m)" for y, c, r, w in bad) + ". A re-segmentation of the Wind business (restated comparatives) is the likely cause; the dashboard uses the releases for trends.")
        ok = [d for d in diffs if d not in bad]
        if ok:
            add("Workbook economy matches the earnings releases", "OK", f"{len(ok)} comparisons",
                f"Orders, revenue and segment EBITDA agree for {sorted({d[0] for d in ok})} within 1% (the workbook rounds orders to USD 100m).")
    if results:
        totals = results.get("fy_totals", {})
        for year, t in totals.items():
            g = res[res["year"] == year]
            if len(g) == 4:
                d = abs(g["ebitda_usd_m"].sum() - t["ebitda_usd_m"])
                add(f"Quarterly sums vs stated {year} totals", "OK" if d <= 3 else "Medium", str(year),
                    f"Quarterly orders, revenue and EBITDA sum to the stated {year} totals within USD {max(abs(g[c].sum() - t[c]) for c in ('orders_usd_m', 'revenue_usd_m', 'ebitda_usd_m')):,.0f}m (rounding).")
        add("Wind backlog is annual only", "Info", "year-end", "The quarterly releases give no Wind backlog; the workbook holds year-end backlog for 2023-2025.")
        add("Quarterly series starts in 2024", "Info", f"{res.iloc[0]['label']}", "Earlier quarters are not in the data, so year-on-year comparisons begin in Q1 2025.")

    # --- order list --------------------------------------------------------------------------------
    small, large = _gap_groups(orders)
    if len(small):
        add("Platform MW differs somewhat from order size", "Info", len(small), f"{len(small)} orders ({small['size_mw'].sum():,.0f} MW) have platform MW 5-25% away from the order size (rounded turbine counts or ratings).")
    if len(large):
        add("Platform MW differs a lot from order size", "Medium", len(large), f"{len(large)} orders ({large['size_mw'].sum():,.0f} MW) differ by more than 25%.")
    raw_names = set(orders["primary_platform"].unique())
    variants = sorted(n for n in raw_names if n in {"GE6-164", "2.8-127", "GE2.X-127"})
    if variants:
        add("Platform names are not consistent", "Low", ", ".join(variants), "The same model appears under several spellings ('GE6-164' vs 'GE6.0-164'; '2.8-127' without the GE prefix; 'GE2.X-127' without a rating). The intelligence layer normalises them.")
    grouped = orders[orders["customer"] != orders["customer_raw"]]
    if len(grouped):
        add("Customer spellings grouped", "Info", f"{grouped['customer_raw'].nunique()} names -> {grouped['customer'].nunique()}",
            "Variants such as 'Greenvolt Power (Greenvolt Group)' / 'Greenvolt International Power (Greenvolt Group)' or the four Continuum Green Energy entries are grouped for account analytics.")
    unk = orders[orders["customer"] == UNKNOWN]
    if len(unk):
        add("Customer undisclosed", "Low", len(unk), f"{len(unk)} orders ({unk['size_mw'].sum():,.0f} MW) have no customer.")
    twins = orders[orders.duplicated(["order_date", "customer_raw", "size_mw"], keep=False)]
    if len(twins):
        add("Possible duplicate order rows", "Low", len(twins), f"{len(twins)} rows share date, customer and size.")
    add("Offshore wind is not in the order list", "Info", "offshore", "The order sheets cover onshore turbine orders; offshore wind (about 7% of 2025 revenue) is visible only in the reported revenue.")
    add("Delivery and service fields are mostly empty", "Info", f"{int(orders['service_time_years'].notna().sum())} of {len(orders)} with a service term", "Delivery dates are not recorded and only the newest orders state a service term, so lead-time and service analytics are not available.")

    for label, frame in (("GE Vernova share price", stock), ("Steel/copper", market)):
        if frame is not None and not frame.empty:
            last = pd.Timestamp(frame["date"].max())
            months = (as_of.year - last.year) * 12 + as_of.month - last.month
            if months >= 3:
                add(f"{label} series are stale", "Info", f"{months} months", f"{label} data ends {last.date()}; run tools/refresh_market_data.py.")
    if stock is not None and not stock.empty:
        add("Share price covers GE Vernova only", "Info", f"from {pd.Timestamp(stock['date'].min()):%b %Y}", "GE Vernova (GEV) trades since March/April 2024; GE's wind business has no earlier standalone share price.")
    return pd.DataFrame(rows)


def anomaly_rows(orders: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cols = ["order_id", "order_date", "country", "customer_raw", "size_mw", "primary_platform"]
    out: dict[str, pd.DataFrame] = {}
    lag = orders[orders["booked_earlier"]][cols + ["booked_label"]]
    if len(lag):
        out["Announced after booking"] = lag.reset_index(drop=True)
    small, large = _gap_groups(orders)
    for name, frame in (("Platform MW differs a lot from order size", large), ("Platform MW differs somewhat from order size", small)):
        if len(frame):
            f = frame[cols + ["platform_mw_sum", "mw_gap"]].copy()
            f["platform / order"] = (f["platform_mw_sum"] / f["size_mw"]).round(2)
            out[name] = f.reset_index(drop=True)
    sup = orders[orders["sheet_name"] == "press releases"][cols]
    if len(sup):
        out["Orders taken from press releases"] = sup.reset_index(drop=True)
    twins = orders[orders.duplicated(["order_date", "customer_raw", "size_mw"], keep=False)][cols]
    if len(twins):
        out["Possible duplicate order rows"] = twins.sort_values(["order_date"]).reset_index(drop=True)
    return out
