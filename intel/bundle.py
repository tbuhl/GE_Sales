"""One-shot computation of every analytic the intelligence tabs need (so each rerun is cheap)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import economics, outlook, quality, structure
from .common import as_of_date, load_quarterly_results, load_supplement_meta, prepare_orders, prepare_platforms


@dataclass
class Intel:
    """Precomputed analytics for the full (unfiltered) dataset."""

    as_of: pd.Timestamp
    orders: pd.DataFrame
    platforms: pd.DataFrame
    economy: pd.DataFrame
    results: dict[str, Any] | None
    res: pd.DataFrame
    yt: pd.DataFrame
    halves: pd.DataFrame
    ann_q: pd.DataFrame
    ann_y: pd.DataFrame
    ytd: pd.DataFrame
    ltm: dict[str, Any]
    projection: dict[str, Any]
    qe_overall: dict[str, float]
    qe_by_year: pd.DataFrame
    month_share: pd.Series
    concentration: pd.DataFrame
    mix: pd.DataFrame
    series: pd.DataFrame
    rating_mix: pd.DataFrame
    leading: pd.DataFrame
    radar: pd.DataFrame
    markets: pd.DataFrame
    econ: pd.DataFrame
    anomalies: pd.DataFrame
    completeness: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)


def build_intel(
    economy: pd.DataFrame,
    orders: pd.DataFrame,
    platforms: pd.DataFrame,
    stock: pd.DataFrame | None = None,
    market: pd.DataFrame | None = None,
    results: dict[str, Any] | None = None,
) -> Intel:
    o = prepare_orders(orders)
    p = prepare_platforms(platforms)
    as_of = as_of_date(o)
    if results is None:
        results = load_quarterly_results()
    res = outlook.results_table(results)
    yt = outlook.year_table(res, results)
    meta = load_supplement_meta()
    ann_y = outlook.announced_by_year(o, yt, meta)
    return Intel(
        as_of=as_of,
        orders=o,
        platforms=p,
        economy=economy,
        results=results,
        res=res,
        yt=yt,
        halves=outlook.half_year_table(res),
        ann_q=outlook.announced_by_booking_quarter(o, res),
        ann_y=ann_y,
        ytd=outlook.ytd_table(o, as_of),
        ltm=outlook.ltm_summary(o, res, as_of),
        projection=outlook.project_year(results, res),
        qe_overall=outlook.quarter_end_overall(o),
        qe_by_year=outlook.quarter_end_share_by_year(o),
        month_share=outlook.month_in_quarter_share(o),
        concentration=structure.concentration_by_year(o, as_of),
        mix=structure.mix_by_year(p),
        series=structure.series_mix(p),
        rating_mix=structure.rating_bucket_mix(p),
        leading=structure.leading_platforms(p),
        radar=structure.account_radar(o, as_of),
        markets=structure.market_status(o, as_of),
        econ=economics.economics_table(economy, yt, market),
        anomalies=quality.anomaly_summary(o, p, res, results, economy, yt, ann_y, stock, market, meta),
        completeness=quality.completeness_by_year(o),
        meta=meta,
    )
