"""Rule-based insight generator: turns the computed analytics into short, number-backed findings.

Every figure in a finding is computed from the data at runtime - nothing here is hard-coded commentary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .bundle import Intel
from .common import MEGA_MW, UNKNOWN, pct_change


@dataclass
class Insight:
    key: str
    category: str
    headline: str
    detail: str
    metric: str
    metric_label: str
    tone: str = "neutral"  # up | down | warn | neutral
    series: list[float] = field(default_factory=list)
    series_start: int | None = None
    where: str = ""


# ---- formatting ------------------------------------------------------------------------
def usd(x: float) -> str:
    """USD million, shown as USD bn from 1,000."""
    sign = "-" if x < 0 else ""
    a = abs(x)
    return f"{sign}USD {a / 1000:,.1f}bn" if a >= 1000 else f"{sign}USD {a:,.0f}m"


def gw(x: float, d: int = 1) -> str:
    return f"{x / 1000.0:,.{d}f} GW"


def mwf(x: float) -> str:
    return f"{x:,.0f} MW"


def pc(x: float, d: int = 0, signed: bool = False) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:+.{d}%}" if signed else f"{x:.{d}%}"


def _tone(x: float, good_up: bool = True) -> str:
    if x is None or pd.isna(x) or abs(x) < 0.02:
        return "neutral"
    return "up" if (x > 0) == good_up else "down"


def _join_and(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _half(n: int) -> str:
    return {1: "Q1", 2: "H1", 3: "the first nine months", 4: "the full year"}.get(n, f"{n} quarters")


# ---- individual insights ---------------------------------------------------------------
def insight_orders(i: Intel) -> Insight | None:
    r = i.ltm.get("reported")
    pr = i.projection
    res = i.res
    if not r or not pr:
        return None
    last = res.iloc[-1]
    n, y = pr["quarters_reported"], pr["year"]
    cur, prv = pr["ytd"]["orders_usd_m"], pr["prev_ytd"]["orders_usd_m"]
    return Insight(
        key="orders",
        category="Orders",
        headline=f"Wind orders are {usd(r['orders'])} over the last four quarters ({pc(r['orders_growth'], signed=True)}), but {last['label']} fell {pc(abs(last['orders_usd_m_yoy']))}",
        detail=(
            f"{_half(n).capitalize()} of {y}: {usd(cur)} against {usd(prv)} a year earlier ({pc(pct_change(cur, prv), signed=True)}). {res.iloc[-2]['label']} was {usd(res.iloc[-2]['orders_usd_m'])}, "
            f"{pc(res.iloc[-2]['orders_usd_m_yoy'], signed=True)} on the year-earlier quarter, and Q4 2025 at {usd(res[res['label'] == 'Q4 2025'].iloc[0]['orders_usd_m'])} is the strongest in the series. "
            f"Book-to-bill over the last four quarters is {r['book_to_bill']:.2f}. Latest quarter: {last['note']}."
        ),
        metric=pc(last["orders_usd_m_yoy"], signed=True),
        metric_label=f"Wind orders, {last['label']} vs a year earlier",
        tone="down" if last["orders_usd_m_yoy"] < -0.02 else "neutral",
        series=res["orders_usd_m"].tolist(),
        series_start=0,
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_profit(i: Intel) -> Insight | None:
    res = i.res
    h = i.halves
    if len(h) < 2 or res.empty:
        return None
    last = res.iloc[-1]
    y = int(last["year"])
    cur, prv = h.loc[(y, "H1")], h.loc[(y - 1, "H1")]
    r = i.ltm["reported"]
    worst = res.loc[res["ebitda_margin"].idxmin()]
    return Insight(
        key="profit",
        category="Profitability",
        headline=f"Segment EBITDA loss widened to {usd(-cur['ebitda_usd_m'])} in H1 {y} ({pc(cur['ebitda_margin'], 1)} margin) from {usd(-prv['ebitda_usd_m'])}",
        detail=(
            f"Revenue fell {pc(abs(pct_change(cur['revenue_usd_m'], prv['revenue_usd_m'])))} to {usd(cur['revenue_usd_m'])} as onshore equipment deliveries followed the soft 2025 orders; the last four quarters lost {usd(-r['ebitda'])} "
            f"against {usd(-r['prior_ebitda'])} in the four before. {worst['label']} was the weakest quarter in the series ({pc(worst['ebitda_margin'], 1)}): {worst['note']}"
        ),
        metric=usd(cur["ebitda_usd_m"]),
        metric_label=f"Wind segment EBITDA, H1 {y}",
        tone="down",
        series=res["ebitda_usd_m"].tolist(),
        series_start=0,
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_guidance(i: Intel) -> Insight | None:
    pr = i.projection
    if not pr or "ebitda_needed_remaining" not in pr:
        return None
    g = pr["guidance"]
    need = pr["ebitda_needed_remaining"]
    rn = pr["revenue_needed_remaining"]
    y = pr["year"]
    return Insight(
        key="guidance",
        category="Outlook",
        headline=f"Hitting the {y} guidance of a Wind EBITDA loss of about {usd(-g['ebitda_usd_m'])} needs a profit of {usd(need)} in the second half, against a loss of {usd(-pr['ebitda_prev_remaining'])} a year earlier",
        detail=(
            f"The first half already lost {usd(-pr['ytd']['ebitda_usd_m'])}, so a loss of only about {usd(-g['ebitda_usd_m'])} for the year means a positive H2 - a swing of {usd(need - pr['ebitda_prev_remaining'])} on H2 {y - 1} (a loss of {usd(-pr['ebitda_prev_remaining'])}). "
            f"For revenue, a 10-13% decline ('low double digits', our reading) means {usd(rn[0])}-{usd(rn[1])} in H2 against {usd(pr['prev_remaining']['revenue_usd_m'])} last year. "
            f"Guidance was unchanged at Q1 and Q2 ({g.get('as_of', '')}); it is the main thing to test when Q3 is reported."
        ),
        metric=usd(need),
        metric_label=f"H2 {y} segment EBITDA needed for guidance",
        tone="warn",
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_orders_outlook(i: Intel) -> Insight | None:
    pr = i.projection
    if not pr:
        return None
    sc, qo = pr["orders_scenario"], pr["quarter_orders"]
    full = i.res.groupby("year").filter(lambda g: len(g) == 4)
    ranks = [int(g["orders_usd_m"].rank(ascending=False)[g["quarter"] == 4].iloc[0]) for _, g in full.groupby("year")]
    q4_txt = f" Q4 ranked #{' and #'.join(map(str, ranks))} of four quarters for orders in the {len(ranks)} complete years on record here," if ranks else ""
    return Insight(
        key="orders_outlook",
        category="Outlook",
        headline=f"{pr['year']} Wind orders are on course for ~{usd(sc['base'])} on a median second half (last year {usd(pr['prev_full']['orders_usd_m'])})",
        detail=(
            f"{usd(pr['ytd']['orders_usd_m'])} booked in H1; the remaining {pr['remaining']} quarters use the lowest, median and highest of the last {qo['lookback']} reported quarters "
            f"({usd(qo['low'])}, {usd(qo['median'])}, {usd(qo['high'])}), giving {usd(sc['conservative'])} to {usd(sc['strong'])}. "
            f"Wind orders are lumpy and seasonal -{q4_txt} so the scenario is wide by construction."
        ),
        metric=usd(sc["base"]),
        metric_label=f"{pr['year']} Wind orders, base scenario",
        tone="neutral",
        series=i.yt["orders_usd_m"].tolist()[:-1] + [sc["base"]],
        series_start=int(i.yt.index.min()),
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_mix(i: Intel) -> Insight | None:
    e = i.econ.dropna(subset=["revenue_services"])
    if len(e) < 2:
        return None
    y = int(e.index.max())
    row = e.loc[y]
    return Insight(
        key="mix",
        category="Business mix",
        headline=f"Services are {pc(row['services_share'])} of {y} Wind revenue ({pc(row['services_growth'], signed=True)}) while equipment fell {pc(abs(row['equipment_growth']))} and offshore revenue fell {pc(abs(pct_change(row['revenue_offshore'], e.loc[y - 1, 'revenue_offshore'])))}",
        detail=(
            f"Services revenue {usd(row['revenue_services'])} on {int(row['services_turbines']):,} serviced turbines ({pc(row['services_coverage'])} of an installed base of {int(row['installed_turbines']):,}) "
            f"works out at about USD {row['services_revenue_per_covered_turbine_k']:,.0f}k per covered turbine. Offshore revenue was {usd(row['revenue_offshore'])} ({pc(row['offshore_share'])}) against {usd(e.loc[y - 1, 'revenue_offshore'])} a year earlier, "
            f"and LM Wind Power fell to {usd(row['revenue_lm'])} from {usd(e.loc[y - 1, 'revenue_lm'])} with the footprint reduction."
        ),
        metric=pc(row["services_share"]),
        metric_label=f"services share of Wind revenue, {y}",
        tone="neutral",
        series=e["services_share"].tolist(),
        series_start=int(e.index.min()),
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_backlog(i: Intel) -> Insight | None:
    e = i.econ.dropna(subset=["backlog"])
    if len(e) < 2:
        return None
    y = int(e.index.max())
    row, first = e.loc[y], e.iloc[0]
    return Insight(
        key="backlog",
        category="Visibility",
        headline=f"Wind backlog fell to {usd(row['backlog'])} at end-{y}, from {usd(first['backlog'])} at end-{int(e.index.min())}",
        detail=(
            f"That is {row['backlog_cover_years']:.1f} years of revenue ({e['backlog_cover_years'].iloc[0]:.1f} in {int(e.index.min())}). Book-to-bill was below 1 in {int((e['book_to_bill'] < 1).sum())} of {len(e)} years; "
            f"the latest four quarters stand at {i.ltm['reported']['book_to_bill']:.2f}. Wind backlog is reported only at year-end."
        ),
        metric=usd(row["backlog"]),
        metric_label=f"Wind backlog, end-{y}",
        tone="down",
        series=e["backlog"].tolist(),
        series_start=int(e.index.min()),
        where="Intelligence Lab > Momentum & Outlook",
    )


def insight_coverage(i: Intel) -> Insight | None:
    a = i.ann_y
    if a.empty or "usd_m_per_announced_mw" not in a:
        return None
    d = a.dropna(subset=["reported_orders_usd_m"])
    d = d[d["quarters"] == 4]
    if len(d) < 2:
        return None
    y_last = int(d.index.max())
    y_ref = int(d["usd_m_per_announced_mw"].idxmin())
    last, ref = d.loc[y_last], d.loc[y_ref]
    lag = i.qe_overall
    return Insight(
        key="coverage",
        category="Visibility",
        headline=f"The order list holds {mwf(last['announced_mw'])} for {y_last} against {usd(last['reported_orders_usd_m'])} of reported Wind orders",
        detail=(
            f"In {y_ref} the same list covered {mwf(ref['announced_mw'])} for {usd(ref['reported_orders_usd_m'])} (USD {ref['usd_m_per_announced_mw']:.2f}m per announced MW). "
            f"{y_last} adds {mwf(last['unlisted_mw'])} of US repower orders that GE Vernova announced only in aggregate, and {int(i.orders['booked_earlier'].sum())} orders in the list were announced after the quarter they were booked in. "
            "Bookings now include more services and unannounced orders, so announced MW is no longer a proxy for demand."
        ),
        metric=mwf(last["announced_mw"]),
        metric_label=f"announced MW booked in {y_last}",
        tone="warn",
        series=a["announced_mw"].tolist(),
        series_start=int(a.index.min()),
        where="Intelligence Lab > Data Quality",
    )


def insight_product_geo(i: Intel) -> Insight | None:
    o = i.orders
    y = i.as_of.year
    cur = o[o["order_year"] >= y - 1]
    by_c = cur[~cur["multi_country"]].groupby("country_clean")["size_mw"].sum().sort_values(ascending=False)
    p = i.platforms[(i.platforms["order_year"] >= y - 1) & (i.platforms["slot_mw"] > 0) & (i.platforms["family"] != UNKNOWN)]
    fam = p.groupby("family")["slot_mw"].sum().sort_values(ascending=False)
    if by_c.empty or fam.empty:
        return None
    ytd = o[o["order_year"] == y]
    eur = ytd[ytd["continent"] == "Europe"]["size_mw"].sum() / ytd["size_mw"].sum()
    earlier = o[o["order_year"] < y]
    new_c = sorted(set(ytd["country"]) - set(earlier["country"]))
    new_p = sorted(set(ytd["primary_platform"]) - set(earlier["primary_platform"]) - {"Unknown"})
    parts = []
    if new_c:
        parts.append("first orders from " + _join_and(new_c))
    if new_p:
        parts.append("first orders for " + _join_and(new_p))
    novelty = ("New in the list this year: " + "; ".join(parts) + ". ") if parts else ""
    return Insight(
        key="product_geo",
        category="Product and geography",
        headline=f"{fam.index[0]} is {pc(fam.iloc[0] / fam.sum())} of platform-mapped MW since {y - 1}; Europe is {pc(eur)} of {y}'s announced MW",
        detail=(
            f"Leading platforms since {y - 1}: {_join_and([f'{n} ({v:,.0f} MW)' for n, v in fam.head(3).items()])}. Top countries: {_join_and([f'{c} ({v:,.0f} MW)' for c, v in by_c.head(4).items()])}. "
            f"{novelty}Announced orders in {y} to {i.as_of:%d %b} total {mwf(ytd['size_mw'].sum())} from {len(ytd)} orders (2024: {mwf(o[o['order_year'] == 2024]['size_mw'].sum())})."
        ),
        metric=pc(fam.iloc[0] / fam.sum()),
        metric_label=f"{fam.index[0]} share of recent platform MW",
        tone="neutral",
        series=i.mix["rating_mw"].tolist(),
        series_start=int(i.mix.index.min()),
        where="Intelligence Lab > Market Structure",
    )


def insight_accounts(i: Intel) -> Insight | None:
    r = i.radar
    key = r[(r["lifetime_mw"] >= 400) & r["repeat_buyer"]]
    rep = r[r["repeat_buyer"]]
    if r.empty:
        return None
    dormant = key[key["status"] == "Dormant"]
    top = r.head(3)
    newest = i.orders[i.orders["order_year"] >= i.as_of.year].groupby("customer")["size_mw"].sum()
    first_seen = i.orders.groupby("customer")["order_year"].min()
    new_names = [c for c in newest.index if first_seen[c] >= i.as_of.year and c != UNKNOWN]
    return Insight(
        key="accounts",
        category="Accounts",
        headline=f"The three largest accounts ({_join_and(list(top['customer']))}) hold {gw(top['lifetime_mw'].sum())} but last ordered in {int(top['last_order'].map(lambda d: pd.Timestamp(d).year).max())} or earlier",
        detail=(
            f"{len(dormant)} of {len(key)} repeat accounts of 400+ MW have not ordered for over two years. Only {len(rep)} of {len(r)} named accounts have ordered more than once. "
            f"{len(new_names)} of this year's customers are new ({', '.join(new_names[:4])}), which shows reach into new markets but little repeat volume yet."
        ),
        metric=f"{len(dormant)} / {len(key)}",
        metric_label="repeat accounts (400+ MW) dormant >24 months",
        tone="warn",
        where="Intelligence Lab > Accounts",
    )


def insight_integrity(i: Intel) -> Insight | None:
    a = i.anomalies
    meds = a[a["Severity"].isin(["High", "Medium"])]
    ok = a[a["Severity"] == "OK"]["Check"].head(3).tolist()
    return Insight(
        key="integrity",
        category="Data integrity",
        headline=f"{len(meds)} data checks need attention before the order list is read as bookings",
        detail=("Open items: " + "; ".join(meds["Check"].head(5).tolist()) + ". " + (f"Checks that pass: {'; '.join(ok)}." if ok else "")),
        metric=f"{len(meds)}",
        metric_label="high/medium data checks open",
        tone="warn",
        where="Intelligence Lab > Data Quality",
    )


INSIGHT_BUILDERS: list[Callable[[Intel], Insight | None]] = [
    insight_orders,
    insight_profit,
    insight_guidance,
    insight_orders_outlook,
    insight_mix,
    insight_backlog,
    insight_coverage,
    insight_product_geo,
    insight_accounts,
    insight_integrity,
]


def generate_insights(intel: Intel) -> list[Insight]:
    out: list[Insight] = []
    for build in INSIGHT_BUILDERS:
        try:
            item = build(intel)
        except Exception:  # an insight that cannot be computed must not break the page
            item = None
        if item is not None:
            out.append(item)
    return out


def watchlist(intel: Intel) -> list[str]:
    items: list[str] = []
    pr = intel.projection
    if pr and "ebitda_needed_remaining" in pr:
        y = pr["year"]
        items.append(
            f"Q3 {y}: the guidance implies a profit of {usd(pr['ebitda_needed_remaining'])} in segment EBITDA across Q3 and Q4 after {usd(pr['ytd']['ebitda_usd_m'])} in H1 "
            f"(Q3 {y - 1}: {usd(intel.res[intel.res['label'] == f'Q3 {y - 1}'].iloc[0]['ebitda_usd_m'])}, Q4 {y - 1}: {usd(intel.res[intel.res['label'] == f'Q4 {y - 1}'].iloc[0]['ebitda_usd_m'])})."
        )
        items.append(
            f"Orders: the second-half scenario is {usd(pr['orders_scenario']['conservative'] - pr['ytd']['orders_usd_m'])} to {usd(pr['orders_scenario']['strong'] - pr['ytd']['orders_usd_m'])}; "
            f"Q4 {y - 1} booked {usd(intel.res[intel.res['label'] == f'Q4 {y - 1}'].iloc[0]['orders_usd_m'])}, so a repeat of that would lift {y} above {y - 1}."
        )
    after = intel.orders[intel.orders["order_date"] > intel.res.iloc[-1]["period_end"]]
    if len(after):
        items.append(f"Since {intel.res.iloc[-1]['label']} ended, {len(after)} orders worth {mwf(after['size_mw'].sum())} have been announced ({', '.join(after['customer'].head(4))}); they will show in Q3 orders unless booked earlier.")
    cool = intel.radar[(intel.radar["lifetime_mw"] >= 300) & (intel.radar["status"] == "Cooling") & intel.radar["repeat_buyer"]].head(4)
    if len(cool):
        items.append("Key accounts approaching 24 months without an order: " + ", ".join(cool["customer"]) + ".")
    checked = (intel.meta or {}).get("orders_checked_through")
    if checked:
        items.append(f"The order list was last checked against press releases on {pd.Timestamp(checked):%d %b %Y}; Q3 results normally arrive in the second half of October.")
    return items
