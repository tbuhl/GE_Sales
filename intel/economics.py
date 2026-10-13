"""Company economics by calendar year (USD m) from the workbook's economy sheet: mix, margins, backlog cover, installed base."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import eco_series, eco_wide


def economics_table(economy: pd.DataFrame, yt: pd.DataFrame | None = None, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Annual KPIs (index = calendar year). ``yt`` is the year table built from the quarterly releases (used to cross-check orders)."""
    wide = eco_wide(economy)
    if wide.empty:
        return pd.DataFrame()

    def s(*names: str) -> pd.Series:
        return eco_series(wide, *names)

    t = pd.DataFrame(index=wide.index)
    t["orders"] = s("orders_order_intake_USD_m")
    t["revenue"] = s("revenue_USD_m")
    t["revenue_equipment"] = s("revenue_equipment_USD_m")
    t["revenue_services"] = s("revenue_services_USD_m")
    t["revenue_onshore"] = s("revenue_onshore_USD_m")
    t["revenue_offshore"] = s("revenue_offshore_USD_m")
    t["revenue_lm"] = s("revenue_lm_wind_power_USD_m")
    t["ebitda"] = s("segment_EBITDA_USD_m")
    t["ebitda_margin_pct"] = s("segment_EBITDA_margin_pct")
    t["cost_of_revenue"] = s("cost_of_revenues_USD_m")
    t["backlog"] = s("total order backlog USD_m")
    t["installed_turbines"] = s("installed_base_wind_turbines_units")
    t["services_turbines"] = s("services_agreements_covered_turbines_units")
    t["ebitda_margin"] = t["ebitda"] / t["revenue"]
    t["gross_margin"] = (t["revenue"] - t["cost_of_revenue"]) / t["revenue"]
    t["services_share"] = t["revenue_services"] / t["revenue"]
    t["offshore_share"] = t["revenue_offshore"] / t["revenue"]
    t["book_to_bill"] = t["orders"] / t["revenue"]
    t["backlog_cover_years"] = t["backlog"] / t["revenue"]
    t["revenue_growth"] = t["revenue"].pct_change(fill_method=None)
    t["services_growth"] = t["revenue_services"].pct_change(fill_method=None)
    t["equipment_growth"] = t["revenue_equipment"].pct_change(fill_method=None)
    t["services_coverage"] = t["services_turbines"] / t["installed_turbines"]
    t["services_revenue_per_covered_turbine_k"] = t["revenue_services"] * 1000.0 / t["services_turbines"]
    if yt is not None and not yt.empty:
        t["orders_quarterly_sum"] = yt["orders_usd_m"].reindex(t.index)
    if market is not None and not market.empty:
        m = market.copy()
        m["year"] = pd.to_datetime(m["date"]).dt.year
        t["steel_usd_per_ton"] = m.groupby("year")["steel_hrc_usd_per_short_ton"].mean().reindex(t.index)
    return t
