"""Tests for the intelligence layer.

Each analytic is checked against an independent brute-force calculation on the dataset, so the tests keep
passing when the data is refreshed. Tests tagged "data sanity" assert properties of the data, not of the code.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from intel import economics, insights, outlook, quality, structure
from intel.common import (
    GIGA_MW,
    MEGA_MW,
    UNKNOWN,
    as_of_date,
    customer_group,
    platform_family,
    product_series,
    rating_bucket,
    same_day,
)


# ---------------------------------------------------------------- helpers
def test_platform_family_and_series():
    assert platform_family("GE6-164") == "GE 6.0-164" == platform_family("GE6.0-164")
    assert platform_family("GE6.1-158") == "GE 6.1-158" and platform_family("2.8-127") == "GE 2.8-127"
    assert platform_family("Unknown") == UNKNOWN and platform_family(None) == UNKNOWN
    assert product_series("GE6.1-158") == product_series("GE6-164") == "5.5-6.1 MW class (158-164 m rotors)"
    assert product_series("GE3.8-154") == "3.6-4.2 MW class" and product_series("GE2.5-120") == "1.5-3.4 MW (older)"
    assert product_series("Unknown") == UNKNOWN
    assert rating_bucket(2.0) == "<2.5 MW" and rating_bucket(14) == "10+ MW" and rating_bucket(np.nan) == UNKNOWN


def test_customer_grouping():
    assert customer_group("Greenvolt Power") == customer_group("Greenvolt") == "Greenvolt"
    assert customer_group(None) == UNKNOWN and customer_group("unknown") == UNKNOWN
    assert customer_group("Some New Developer") == "Some New Developer"


def test_same_day_handles_leap_day():
    assert same_day(2023, pd.Timestamp("2024-02-29")) == pd.Timestamp("2023-02-28")


def test_prepare_orders_adds_columns(orders, raw):
    for col in ["country_clean", "multi_country", "date_known", "book_year", "book_quarter", "booked_earlier", "series", "is_mega", "customer_raw"]:
        assert col in orders
    assert (orders["is_mega"] == (orders["size_mw"] >= MEGA_MW)).all()
    assert len(orders) == len(raw["orders"])
    # booking quarter equals the stated one where given, else the announcement quarter
    stated = orders["booked_year"].notna()
    assert (orders.loc[stated, "book_year"] == orders.loc[stated, "booked_year"]).all()
    assert (orders.loc[~stated, "book_year"] == orders.loc[~stated, "order_year"]).all()
    assert (orders.loc[~stated, "book_quarter"] == orders.loc[~stated, "order_quarter"]).all()
    assert not (orders.loc[~stated, "booked_earlier"]).any()
    assert orders["date_known"].all()  # data sanity: every order carries an exact date


def test_press_orders_are_in_the_cache(orders):
    """data sanity: the tracked supplement is merged into the parsed data."""
    sup = json.loads(open("data/ge_orders_supplement.json", encoding="utf-8-sig").read())
    assert sup["orders"], "supplement has no orders"
    for item in sup["orders"]:
        hit = orders[(orders["order_date"] == pd.Timestamp(item["date"])) & (orders["size_mw"].round(1) == round(item["size_mw"], 1))]
        assert len(hit) == 1, item
    assert not orders.duplicated(["order_date", "size_mw", "country"]).any()


# ---------------------------------------------------------------- quarterly results
def test_results_file_is_consistent(results):
    q = results["quarterly"]
    assert not q.duplicated(["year", "quarter"]).any() and q["quarter"].between(1, 4).all()
    assert q["period_end"].is_monotonic_increasing
    assert len(str(q["note"].iloc[-1])) > 0  # data sanity: the latest quarter carries a release note


def test_year_table_matches_stated_totals(results):
    res = outlook.results_table(results)
    yt = outlook.year_table(res, results)
    for year, tot in results["fy_totals"].items():
        for col, value in tot.items():
            assert yt.loc[year, col] == pytest.approx(value), (year, col)
    for year in yt.index[yt["quarters"] == 4]:  # data sanity: quarters add up to the stated totals within rounding
        q = res[res["year"] == year]
        for col in ("orders_usd_m", "revenue_usd_m", "ebitda_usd_m"):
            assert q[col].sum() == pytest.approx(yt.loc[year, col], abs=5)


def test_results_table_definitions(results):
    s = outlook.results_table(results)
    assert np.allclose(s["book_to_bill"], s["orders_usd_m"] / s["revenue_usd_m"])
    assert np.allclose(s["ebitda_margin"], s["ebitda_usd_m"] / s["revenue_usd_m"])
    for col in ("orders_usd_m", "revenue_usd_m", "ebitda_usd_m"):
        brute = s[col].rolling(4).sum()
        assert np.allclose(s[f"ltm_{col}"].dropna(), brute.dropna())
    assert s["ltm_orders_usd_m"].isna().sum() == 3
    i = len(s) - 1
    assert s.loc[i, "orders_usd_m_yoy"] == pytest.approx(s.loc[i, "orders_usd_m"] / s.loc[i - 4, "orders_usd_m"] - 1)
    assert s.loc[i, "ebitda_change_yoy_usd_m"] == pytest.approx(s.loc[i, "ebitda_usd_m"] - s.loc[i - 4, "ebitda_usd_m"])


def test_half_year_table(results):
    res = outlook.results_table(results)
    h = outlook.half_year_table(res)
    y = int(res["year"].max())
    first = res[(res["year"] == y) & (res["quarter"] <= 2)]
    if len(first) == 2:
        assert h.loc[(y, "H1"), "orders_usd_m"] == pytest.approx(first["orders_usd_m"].sum())
        assert h.loc[(y, "H1"), "ebitda_usd_m"] == pytest.approx(first["ebitda_usd_m"].sum())
    assert not h.isna().any().any()


def test_announced_by_booking_quarter_matches_bruteforce(orders, results):
    res = outlook.results_table(results)
    t = outlook.announced_by_booking_quarter(orders, res)
    for _, row in t.iterrows():
        d = orders[(orders["book_year"] == row["year"]) & (orders["book_quarter"] == row["quarter"])]
        assert row["announced_mw_booked"] == pytest.approx(d["size_mw"].sum()) and row["announced_orders"] == len(d)
        a = orders[(orders["order_year"] == row["year"]) & (orders["order_quarter"] == row["quarter"])]
        assert row["announced_mw_by_announcement"] == pytest.approx(a["size_mw"].sum())
    assert t.loc[t["announced_mw_booked"] < outlook.MIN_RATIO_MW, "usd_m_per_announced_mw"].isna().all()
    # stated booking quarters move MW between quarters but never create or lose any
    window = orders[(orders["book_year"] * 4 + orders["book_quarter"]).between(2024 * 4 + 1, int(res.iloc[-1]["year"]) * 4 + int(res.iloc[-1]["quarter"]))]
    assert t["announced_mw_booked"].sum() == pytest.approx(window["size_mw"].sum())


def test_announced_by_year_includes_unlisted(orders, results, intel):
    res = outlook.results_table(results)
    yt = outlook.year_table(res, results)
    meta = intel.meta
    t = outlook.announced_by_year(orders, yt, meta)
    assert t["announced_mw"].sum() == pytest.approx(orders["size_mw"].sum())
    for item in meta.get("unlisted_bookings") or []:
        assert t.loc[int(item["year"]), "unlisted_mw"] == pytest.approx(item["mw"])
    y = int(t.index[(t["announced_mw"] >= outlook.MIN_RATIO_MW) & t["reported_orders_usd_m"].notna()][0])
    assert t.loc[y, "usd_m_per_announced_mw"] == pytest.approx(t.loc[y, "reported_orders_usd_m"] / t.loc[y, "announced_mw"])


# ---------------------------------------------------------------- momentum
def test_ytd_matches_bruteforce(orders):
    as_of = as_of_date(orders)
    t = outlook.ytd_table(orders, as_of)
    for year in (as_of.year, as_of.year - 1, as_of.year - 3):
        cut = same_day(year, as_of)
        sel = orders[(orders["order_year"] == year) & (orders["order_date"] <= cut)]
        assert t.loc[year, "ytd_mw"] == pytest.approx(sel["size_mw"].sum())
        assert t.loc[year, "ytd_mw_ex_giga"] == pytest.approx(sel.loc[sel["size_mw"] < GIGA_MW, "size_mw"].sum())


def test_ltm_summary(orders, results):
    as_of = as_of_date(orders)
    res = outlook.results_table(results)
    s = outlook.ltm_summary(orders, res, as_of)
    start = as_of - pd.DateOffset(years=1)
    cur = orders[(orders["order_date"] > start) & (orders["order_date"] <= as_of)]
    assert s["current"]["mw"] == pytest.approx(cur["size_mw"].sum()) and s["current"]["orders"] == len(cur)
    r = s["reported"]
    assert r["orders"] == pytest.approx(res["orders_usd_m"].iloc[-4:].sum()) and r["prior_orders"] == pytest.approx(res["orders_usd_m"].iloc[-8:-4].sum())
    assert r["book_to_bill"] == pytest.approx(res["orders_usd_m"].iloc[-4:].sum() / res["revenue_usd_m"].iloc[-4:].sum())


def test_cumulative_curves_end_at_annual_totals(orders):
    as_of = as_of_date(orders)
    c = outlook.cumulative_curves(orders, as_of)
    for year in (2022, 2024, as_of.year):
        final = c[c["year"] == year]["cum_mw"].iloc[-1]
        cut = as_of if year == as_of.year else pd.Timestamp(year, 12, 31)
        assert final == pytest.approx(orders[(orders["order_year"] == year) & (orders["order_date"] <= cut)]["size_mw"].sum())
    assert c.groupby("year")["cum_mw"].apply(lambda s: s.is_monotonic_increasing).all()


# ---------------------------------------------------------------- projection
def test_projection_structure(results):
    res = outlook.results_table(results)
    pr = outlook.project_year(results, res, lookback=8)
    sc, qo = pr["orders_scenario"], pr["quarter_orders"]
    hist = res["orders_usd_m"].iloc[-8:]
    assert qo["low"] == hist.min() and qo["high"] == hist.max() and qo["median"] == hist.median()
    assert sc["conservative"] <= sc["base"] <= sc["strong"]
    ytd = res[res["year"] == pr["year"]]
    assert pr["ytd"]["orders_usd_m"] == pytest.approx(ytd["orders_usd_m"].sum())
    assert sc["base"] == pytest.approx(pr["ytd"]["orders_usd_m"] + pr["remaining"] * qo["median"])
    g = results["guidance"]
    assert pr["ebitda_needed_remaining"] == pytest.approx(g["ebitda_usd_m"] - ytd["ebitda_usd_m"].sum())
    prev = res[res["year"] == pr["year"] - 1]
    lo, hi = g["revenue_decline_assumed"]
    assert pr["revenue_full_year"][0] == pytest.approx(prev["revenue_usd_m"].sum() * (1 + lo))
    assert pr["revenue_needed_remaining"][1] == pytest.approx(prev["revenue_usd_m"].sum() * (1 + hi) - ytd["revenue_usd_m"].sum())
    assert pr["prev_remaining"]["revenue_usd_m"] == pytest.approx(prev[prev["quarter"] > pr["quarters_reported"]]["revenue_usd_m"].sum())


# ---------------------------------------------------------------- timing
def test_quarter_end_shares_are_fractions(orders):
    by = outlook.quarter_end_share_by_year(orders)
    assert by["mw_share"].between(0, 1).all() and by["orders_share"].between(0, 1).all()
    overall = outlook.quarter_end_overall(orders)
    recent = orders[orders["order_year"] >= 2020]
    assert overall["mw_share"] == pytest.approx(recent.loc[recent["days_to_quarter_end"] < 14, "size_mw"].sum() / recent["size_mw"].sum())
    assert 0 <= overall["booked_earlier_share"] <= 1
    assert overall["booked_earlier_share"] == pytest.approx(recent.loc[recent["booked_earlier"], "size_mw"].sum() / recent["size_mw"].sum())
    assert outlook.month_in_quarter_share(orders).sum() == pytest.approx(1.0)


# ---------------------------------------------------------------- structure
def test_concentration_invariants(orders):
    c = structure.concentration_by_year(orders)
    assert ((c["top1_share"] <= c["top5_share"] + 1e-12) & (c["top5_share"] <= c["top10_share"] + 1e-12) & (c["top10_share"] <= 1 + 1e-12)).all()
    year = int(c.index[-2])
    assert c.loc[year, "orders"] == (orders["order_year"] == year).sum()
    assert c.loc[year, "mega_orders"] == ((orders["order_year"] == year) & (orders["size_mw"] >= MEGA_MW)).sum()


def test_period_compare_is_consistent(orders):
    as_of = as_of_date(orders)
    cur = (as_of - pd.DateOffset(years=1), as_of)
    prior = (as_of - pd.DateOffset(years=2), as_of - pd.DateOffset(years=1))
    p = structure.period_compare(orders, cur, prior)
    assert np.allclose(p["delta_mw"], p["mw_current"] - p["mw_prior"])
    assert "Unknown" not in set(p.index)
    brute = orders[(orders["order_date"] > cur[0]) & (orders["order_date"] <= cur[1]) & ~orders["multi_country"]]["size_mw"].sum()
    assert p["mw_current"].sum() == pytest.approx(brute)


def test_account_radar_rules(orders):
    as_of = as_of_date(orders)
    r = structure.account_radar(orders, as_of)
    named = orders[orders["customer"] != UNKNOWN]
    assert r["lifetime_mw"].sum() == pytest.approx(named["size_mw"].sum())
    assert (r.loc[r["days_since_last"] <= 365, "status"] == "Active").all()
    assert (r.loc[r["days_since_last"] > 730, "status"] == "Dormant").all()
    assert (r.loc[~r["repeat_buyer"], "order_dates"] == 1).all()
    assert r["customer"].is_unique  # spellings are grouped, so each account appears once
    assert r["customer"].str.lower().str.strip().is_unique  # no two spellings differ only by case or spacing
    assert (orders["customer"].nunique() <= orders["customer_raw"].nunique())


def test_market_status_covers_all_countries(orders):
    m = structure.market_status(orders)
    assert set(m["status"]) <= {"Active", "Cooling", "Dormant", "Emerging"}
    assert m["lifetime_mw"].sum() == pytest.approx(orders[~orders["multi_country"]]["size_mw"].sum())


def test_mix_and_series_shares(platforms):
    mix = structure.mix_by_year(platforms)
    assert mix["large_turbine_share"].dropna().between(0, 1).all()
    assert np.allclose(structure.rating_bucket_mix(platforms).sum(axis=1), 1.0)
    sm = structure.series_mix(platforms)
    assert np.allclose(sm.sum(axis=1), 1.0)
    year = int(mix.index[-2])
    known = platforms[(platforms["order_year"] == year) & (platforms["slot_mw"] > 0) & (platforms["family"] != UNKNOWN)]
    big = known.loc[known["mw_rating"] >= 5.5, "slot_mw"].sum()
    assert mix.loc[year, "large_turbine_share"] == pytest.approx(big / known["slot_mw"].sum())
    pm = structure.platform_mw(platforms, year, top=50)
    assert "GE6-164" not in pm.index and "GE6.0-164" not in pm.index  # variants are one platform family


def test_entity_profile_matches_direct_sums(orders, platforms):
    country = orders.loc[~orders["multi_country"], "country_clean"].value_counts().index[0]
    prof = structure.entity_profile(orders, platforms, "Country", country)
    sub = orders[orders["country_clean"] == country]
    assert prof["mw"] == pytest.approx(sub["size_mw"].sum()) and prof["orders"] == len(sub)
    assert 0 <= prof["large_turbine_share"] <= 1 or np.isnan(prof["large_turbine_share"])
    fam = platforms["family"].value_counts().index[0]
    pf = structure.entity_profile(orders, platforms, "Platform family", fam)
    assert pf["mw"] == pytest.approx(platforms.loc[platforms["family"] == fam, "slot_mw"].sum())
    assert structure.entity_yearly(orders, platforms, "Country", country).sum() == pytest.approx(prof["mw"])


# ---------------------------------------------------------------- economics
def test_economics_table_definitions(raw, results, intel):
    res = outlook.results_table(results)
    yt = outlook.year_table(res, results)
    t = economics.economics_table(raw["economy"], yt)
    y = int(t["revenue"].dropna().index.max())
    assert t.loc[y, "book_to_bill"] == pytest.approx(t.loc[y, "orders"] / t.loc[y, "revenue"])
    assert t.loc[y, "services_share"] == pytest.approx(t.loc[y, "revenue_services"] / t.loc[y, "revenue"])
    assert t.loc[y, "backlog_cover_years"] == pytest.approx(t.loc[y, "backlog"] / t.loc[y, "revenue"])
    assert t.loc[y, "ebitda_margin"] == pytest.approx(t.loc[y, "ebitda"] / t.loc[y, "revenue"])
    assert t.loc[y, "revenue_growth"] == pytest.approx(t.loc[y, "revenue"] / t.loc[y - 1, "revenue"] - 1)
    # data sanity: onshore + offshore + LM Wind Power revenue add up to the workbook's total (within rounding)
    seg = t.loc[y, ["revenue_onshore", "revenue_offshore", "revenue_lm"]].sum()
    assert seg >= t.loc[y, "revenue"] * 0.97 or t.loc[y, "revenue_equipment"] + t.loc[y, "revenue_services"] == pytest.approx(t.loc[y, "revenue"], rel=0.02)
    assert intel.econ.index.equals(t.index)


# ---------------------------------------------------------------- quality + insights
def test_quality_flags_known_issues(intel):
    checks = " | ".join(intel.anomalies["Check"])
    assert "Announced MW covers a shrinking part of bookings" in checks
    assert "Bookings announced without detail" in checks
    assert "Quarterly sums vs stated" in checks
    assert set(intel.anomalies["Severity"]) <= {"High", "Medium", "Low", "OK", "Info"}
    assert intel.anomalies["Affected"].map(type).eq(str).all()  # Arrow-safe column
    stated = intel.anomalies[intel.anomalies["Check"].str.startswith("Quarterly sums vs stated")]
    assert (stated["Severity"] == "OK").all()  # data sanity: the quarters add up to the stated annual totals


def test_completeness_is_bounded(intel):
    assert intel.completeness.stack().between(0, 1).all()
    assert intel.completeness["Country known"].min() == 1.0  # data sanity


def test_anomaly_rows_are_subsets_of_orders(orders):
    for name, frame in quality.anomaly_rows(orders).items():
        if "order_id" in frame:
            assert set(frame["order_id"]) <= set(orders["order_id"]), name


def test_insights_generate_and_are_clean(intel):
    items = insights.generate_insights(intel)
    assert len(items) >= 8
    keys = [i.key for i in items]
    assert len(keys) == len(set(keys))
    for i in items:
        text = f"{i.headline} {i.detail} {i.metric} {i.metric_label}"
        assert i.headline and i.detail and i.metric and i.where
        assert not re.search(r"(\bnan\b|\bNaN\b|\bNone\b|\binf\b|n/a)", text), (i.key, text)
        assert i.tone in {"up", "down", "warn", "neutral"}
    assert insights.watchlist(intel)


def test_each_insight_builder_runs_without_swallowed_errors(intel):
    """generate_insights() hides exceptions; call each builder directly so a regression cannot hide."""
    for build in insights.INSIGHT_BUILDERS:
        assert build(intel) is not None, build.__name__


def test_money_formatters():
    assert insights.usd(-275) == "-USD 275m" and insights.usd(257) == "USD 257m" and insights.usd(7400) == "USD 7.4bn"
    assert insights.pc(0.153) == "15%" and insights.pc(0.153, signed=True) == "+15%"
