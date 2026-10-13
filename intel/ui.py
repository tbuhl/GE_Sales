"""Streamlit layout for the Executive Briefing and the Intelligence Lab."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from . import charts_outlook as co
from . import charts_structure as cs
from . import quality, structure
from .bundle import Intel, build_intel
from .common import GIGA_MW, MEGA_MW, UNKNOWN, load_quarterly_results, prepare_orders, prepare_platforms, reported_results_path, supplement_path
from .insights import generate_insights, gw, mwf, pc, usd, watchlist
from .theme import finding_card_html, inject_css, show, table_view, tokens

LAB_SECTIONS = ["Momentum & Outlook", "Market Structure", "Accounts", "Compare Lab", "Data Quality"]


def _mtime(path) -> str:
    return f"{path.stat().st_mtime_ns}" if path.exists() else "missing"


def _data_signature(economy, orders, platforms, stock, market) -> str:
    return "|".join(
        [
            str(len(orders)), str(orders["order_date"].max()), f"{orders['size_mw'].sum():.2f}", str(len(platforms)), f"{platforms['slot_mw'].sum():.2f}",
            str(len(economy)), f"{economy['value'].sum():.2f}", str(None if stock is None else stock.shape), str(None if market is None else market.shape),
            _mtime(reported_results_path()), _mtime(supplement_path()),
        ]
    )


@st.cache_data(show_spinner="Computing the intelligence layer...")
def _cached_intel(signature: str, _economy, _orders, _platforms, _stock, _market) -> Intel:  # noqa: ARG001 - signature drives the cache
    return build_intel(_economy, _orders, _platforms, _stock, _market, load_quarterly_results())


def get_intel(economy, orders, platforms, stock=None, market=None) -> Intel:
    return _cached_intel(_data_signature(economy, orders, platforms, stock, market), economy, orders, platforms, stock, market)


def _scope_note(text: str) -> None:
    st.markdown(f'<div class="intel-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def _csv_button(df: pd.DataFrame, label: str, name: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv", key=key)


# ======================================================================================
# Executive Briefing
# ======================================================================================
def render_executive_briefing(intel: Intel) -> None:
    inject_css()
    t = tokens()
    res = intel.res
    pr = intel.projection
    r = intel.ltm.get("reported")
    last = res.iloc[-1] if len(res) else None

    st.markdown(
        f'<div class="intel-hero"><h2>Executive Briefing</h2>'
        f'<p class="intel-sub">GE Vernova\'s Wind segment. Quarterly results run to {last["label"] if last is not None else "n/a"} (earnings releases, USD); the latest announced order is '
        f"{intel.as_of:%d %B %Y}. Every number is computed from the data - sidebar filters apply to the other tabs, not to this one.</p></div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(6)
    if last is not None and r:
        cols[0].metric(f"Wind orders, {last['label']}", usd(last["orders_usd_m"]), f"{last['orders_usd_m_yoy']:+.0%} vs a year earlier")
        cols[1].metric(f"Wind revenue, {last['label']}", usd(last["revenue_usd_m"]), f"{last['revenue_usd_m_yoy']:+.0%} vs a year earlier")
        cols[2].metric(f"Segment EBITDA, {last['label']}", usd(last["ebitda_usd_m"]), f"margin {last['ebitda_margin']:.1%}", delta_color="off", delta_arrow="off")
        cols[3].metric("Last 4 quarters: orders / revenue", f"{usd(r['orders'])} / {usd(r['revenue'])}", f"book-to-bill {r['book_to_bill']:.2f}", delta_color="off", delta_arrow="off")
    if pr and "ebitda_needed_remaining" in pr:
        cols[4].metric(f"H2 {pr['year']} EBITDA needed for guidance", usd(pr["ebitda_needed_remaining"]), f"H1: {usd(pr['ytd']['ebitda_usd_m'])}; guided year ~{usd(pr['guidance']['ebitda_usd_m'])}", delta_color="off", delta_arrow="off")
    e = intel.econ.dropna(subset=["backlog"])
    if len(e):
        row = e.iloc[-1]
        cols[5].metric(f"Wind backlog, end-{int(e.index[-1])}", usd(row["backlog"]), f"{row['backlog_cover_years']:.1f} yrs of revenue", delta_color="off", delta_arrow="off")

    c1, c2 = st.columns(2)
    with c1:
        show(co.fig_orders_revenue(intel), "brief_or")
    with c2:
        show(co.fig_ebitda(intel), "brief_eb")
    c3, c4 = st.columns(2)
    with c3:
        show(co.fig_halves(intel), "brief_halves")
    with c4:
        show(co.fig_year_orders(intel), "brief_year")
    st.caption(
        "Guidance (Wind): organic revenue down low double digits and about USD 400m of segment EBITDA losses for 2026; 'low double digits' is read as a 10-13% decline for the arithmetic. "
        "The scenario for orders uses the lowest, median and highest of the last 8 quarters; it is a range, not a forecast."
    )

    st.subheader("Key findings")
    items = generate_insights(intel)
    if items:
        st.markdown('<div class="intel-grid">' + "".join(finding_card_html(i, t) for i in items) + "</div>", unsafe_allow_html=True)
    else:
        st.info("Not enough data to generate findings.")
    st.subheader("What to watch next")
    wl = watchlist(intel)
    if wl:
        st.markdown('<ul class="intel-watch">' + "".join(f"<li>{html.escape(w)}</li>" for w in wl) + "</ul>", unsafe_allow_html=True)
    with st.expander("How these findings are generated"):
        st.markdown(
            "- Findings are rule-based: each template pulls its numbers from the same tables used on the Intelligence Lab tab, so every claim can be traced to a chart there.\n"
            "- **Reported results** are the Wind segment in GE Vernova's quarterly earnings releases (USD m); the workbook adds annual revenue mix, backlog and installed base.\n"
            "- **Announced MW** is the order list (workbook plus press releases). Where a release says the order was booked in an earlier quarter, that booking quarter is used.\n"
            "- **Guidance arithmetic** shows what the rest of the year must deliver for the stated outlook; it is not a forecast."
        )


# ======================================================================================
# Intelligence Lab
# ======================================================================================
def render_intelligence_lab(intel: Intel, orders_f: pd.DataFrame, platforms_f: pd.DataFrame) -> None:
    inject_css()
    st.markdown(
        '<div class="intel-hero"><h2>Intelligence Lab</h2><p class="intel-sub">Deeper analytics from reported Wind results, the announced order list and the workbook economics.</p></div>',
        unsafe_allow_html=True,
    )
    if hasattr(st, "segmented_control"):
        choice = st.segmented_control("Section", LAB_SECTIONS, default=LAB_SECTIONS[0], key="intel_lab_section", label_visibility="collapsed")
    else:  # older Streamlit
        choice = st.radio("Section", LAB_SECTIONS, horizontal=True, key="intel_lab_section", label_visibility="collapsed")
    choice = choice or LAB_SECTIONS[0]
    if choice == "Momentum & Outlook":
        _section_momentum(intel)
    elif choice == "Market Structure":
        _section_structure(intel, orders_f, platforms_f)
    elif choice == "Accounts":
        _section_accounts(intel, orders_f)
    elif choice == "Compare Lab":
        _section_compare(intel)
    else:
        _section_quality(intel)


# --------------------------------------------------------------------------------------
def _section_momentum(intel: Intel) -> None:
    _scope_note("Company-level section: uses the full dataset and ignores the sidebar filters.")
    pr = intel.projection
    res = intel.res

    st.subheader("Quarterly results")
    c1, c2 = st.columns(2)
    with c1:
        show(co.fig_orders_revenue(intel), "lab_or")
    with c2:
        show(co.fig_ebitda(intel), "lab_eb")
    cols = ["label", "orders_usd_m", "revenue_usd_m", "ebitda_usd_m", "ebitda_margin", "book_to_bill", "orders_usd_m_yoy", "revenue_usd_m_yoy", "ltm_orders_usd_m", "ltm_revenue_usd_m", "ltm_ebitda_usd_m", "ltm_book_to_bill"]
    table_view(res[cols].rename(columns={"label": "Quarter"}).set_index("Quarter").round(3), "Data table: quarterly Wind results (USD m)")
    notes = res[res["note"].astype(str).str.len() > 0][["label", "note"]].rename(columns={"label": "Quarter", "note": "What drove it (from the release)"})
    if len(notes):
        with st.expander("What drove each quarter"):
            st.dataframe(notes, width="stretch", hide_index=True)

    st.subheader("Guidance arithmetic and orders scenario")
    c3, c4 = st.columns(2)
    with c3:
        show(co.fig_halves(intel), "lab_halves")
    with c4:
        show(co.fig_year_orders(intel), "lab_year")
    if pr:
        with st.expander("Working", expanded=True):
            sc, qo, g, y = pr["orders_scenario"], pr["quarter_orders"], pr["guidance"], pr["year"]
            rows = [
                (f"{y} reported to date ({pr['label']})", f"orders {usd(pr['ytd']['orders_usd_m'])}; revenue {usd(pr['ytd']['revenue_usd_m'])}; EBITDA {usd(pr['ytd']['ebitda_usd_m'])}",
                 f"same period {y - 1}: {usd(pr['prev_ytd']['orders_usd_m'])}; {usd(pr['prev_ytd']['revenue_usd_m'])}; {usd(pr['prev_ytd']['ebitda_usd_m'])}"),
                (f"Guided {y} EBITDA", usd(g["ebitda_usd_m"]), g.get("as_of", "")),
                ("EBITDA needed in the rest of the year", usd(pr["ebitda_needed_remaining"]), f"same period last year: {usd(pr['ebitda_prev_remaining'])}"),
                (f"Revenue needed in the rest of the year (assumed {abs(g['revenue_decline_assumed'][1]):.0%}-{abs(g['revenue_decline_assumed'][0]):.0%} decline)", f"{usd(pr['revenue_needed_remaining'][0])} - {usd(pr['revenue_needed_remaining'][1])}",
                 f"same period last year: {usd(pr['prev_remaining']['revenue_usd_m'])}"),
                ("Remaining quarters: low / median / high of the last 8", f"{usd(qo['low'])} / {usd(qo['median'])} / {usd(qo['high'])}", "reported Wind orders per quarter"),
                (f"{y} orders (conservative / base / strong)", f"{usd(sc['conservative'])} / {usd(sc['base'])} / {usd(sc['strong'])}", f"last year: {usd(pr['prev_full']['orders_usd_m'])}"),
            ]
            st.dataframe(pd.DataFrame(rows, columns=["Item", "Value", "Basis"]), width="stretch", hide_index=True)
            st.caption(g.get("revenue_decline_note", ""))
    table_view(intel.halves.round(3), "Data table: half-year orders, revenue and EBITDA")

    st.subheader("Announced orders against reported orders")
    show(co.fig_announced_vs_orders(intel), "lab_annvs")
    aq = intel.ann_q.copy()
    aq["usd_m_per_announced_mw"] = aq["usd_m_per_announced_mw"].round(2)
    table_view(aq, "Data table: announced MW by booking quarter against reported orders")
    table_view(intel.ann_y.round(2), "Data table: announced MW by booking year")
    st.caption(
        "Reported orders include services, offshore and orders that are never announced; the list covers individually announced onshore turbine orders only. "
        "Releases are dated by announcement, so the stated booking quarter is used where given (for example the Q4 2025 bookings announced in March-April 2026)."
    )
    show(co.fig_cumulative_curves(intel, height=470, width_hint=1150), "lab_cum")
    qe = intel.qe_overall
    st.caption(f"Timing: {pc(qe['mw_share'])} of announced MW falls in the last {qe['window_days']} days of a quarter (even spread: {pc(qe['expected_if_uniform'])}); {pc(qe['booked_earlier_share'])} was booked in an earlier quarter than announced.")

    st.subheader("Annual economics")
    if intel.econ.empty:
        st.info("Economy data not available.")
        return
    c5, c6 = st.columns(2)
    with c5:
        show(co.fig_revenue_mix(intel), "lab_revmix")
    with c6:
        show(co.fig_revenue_segments(intel), "lab_revseg")
    c7, c8 = st.columns(2)
    with c7:
        show(co.fig_backlog(intel), "lab_backlog")
    with c8:
        show(co.fig_installed_base(intel), "lab_ib")
    cols_view = ["orders", "revenue", "revenue_growth", "services_share", "services_growth", "equipment_growth", "offshore_share", "ebitda", "ebitda_margin", "gross_margin", "backlog", "backlog_cover_years", "book_to_bill", "installed_turbines", "services_coverage", "services_revenue_per_covered_turbine_k"]
    table_view(intel.econ[[c for c in cols_view if c in intel.econ]].round(3), "Data table: derived economics by year")
    st.caption("Annual figures come from the workbook (2023-2025) and the annual reports; the workbook rounds orders to USD 100m. Gross margin is revenue less cost of revenues.")


# --------------------------------------------------------------------------------------
def _section_structure(intel: Intel, orders_f: pd.DataFrame, platforms_f: pd.DataFrame) -> None:
    _scope_note("This section respects the sidebar filters. It covers announced onshore turbine orders only.")
    o = prepare_orders(orders_f)
    p = prepare_platforms(platforms_f)
    if o.empty:
        st.info("No orders after filtering.")
        return
    as_of = intel.as_of

    st.subheader("Order size and lumpiness")
    conc = structure.concentration_by_year(o, as_of)
    mode = st.radio("Size-class view", ["Absolute MW", "Share of MW"], horizontal=True, key="str_size_mode")
    c1, c2 = st.columns(2)
    with c1:
        show(cs.fig_top_share(conc), "str_top")
    with c2:
        show(cs.fig_size_class_mix(o, share=(mode == "Share of MW")), "str_size")
    st.markdown("**Largest orders in scope**")
    st.dataframe(structure.largest_orders(o, 15), width="stretch", hide_index=True)
    table_view(conc[["orders", "mw", "median_mw", "top1_share", "top5_share", "top10_share", "mega_orders", "mega_share", "giga_orders"]].round(3), "Data table: concentration by year")
    st.caption(f"Mega orders are {MEGA_MW:,.0f} MW or more; jumbo orders {GIGA_MW:,.0f} MW or more. 2024 includes the 2.4 GW US order and a 2.9 GW Australian group of orders.")

    st.subheader("Product mix")
    mix = structure.mix_by_year(p)
    if mix.empty:
        st.info("No platform-level rows for the current filters.")
    else:
        c3, c4 = st.columns(2)
        with c3:
            show(cs.fig_rating_mix(structure.rating_bucket_mix(p)), "str_ratingmix")
        with c4:
            show(cs.fig_series_mix(structure.series_mix(p)), "str_series")
        show(cs.fig_rating_rotor(mix), "str_ratingrotor")
        years = sorted(int(y) for y in p.loc[p["slot_mw"].fillna(0) > 0, "order_year"].unique())
        pick = st.selectbox("Platforms ordered in", years, index=len(years) - 1, key="str_plat_year")
        pm = structure.platform_mw(p, int(pick), 12)
        if len(pm):
            show(cs.fig_platform_mw(pm, f"Platforms by ordered MW, {pick}"), "str_platforms")
        lead = structure.leading_platforms(p, 3)
        if not lead.empty:
            lt = lead.pivot(index="year", columns="rank", values="platform").rename(columns=lambda r: f"#{r} platform")
            sh = lead.pivot(index="year", columns="rank", values="share").rename(columns=lambda r: f"#{r} share")
            tbl = pd.concat([lt, sh.map(lambda v: pc(v))], axis=1)
            tbl = tbl[[c for r in (1, 2, 3) for c in (f"#{r} platform", f"#{r} share") if c in tbl.columns]]
            st.markdown("**Leading platforms by order year** (share of platform-mapped MW)")
            st.dataframe(tbl.sort_index(ascending=False), width="stretch")
        st.caption("Platform names are normalised (GE6-164 and GE6.0-164 are one model). Series: the 5.5-6.1 MW class (158 and 164 m rotors), the 3.6-4.2 MW class, and older 1.5-3.4 MW models.")
        table_view(structure.series_mix(p).round(3), "Data table: product series share by order year")

    st.subheader("Geography")
    window = st.radio("Compare", ["Last 12 months vs the 12 before", "Last full year vs the year before"], horizontal=True, key="str_geo_window")
    if window.startswith("Last 12"):
        cur = (as_of - pd.DateOffset(years=1), as_of)
        prior = (as_of - pd.DateOffset(years=2), as_of - pd.DateOffset(years=1))
        title = f"Country movers: last 12 months vs prior 12 months (to {as_of:%d %b %Y})"
    else:
        y = as_of.year - 1
        cur = (pd.Timestamp(y - 1, 12, 31), pd.Timestamp(y, 12, 31))
        prior = (pd.Timestamp(y - 2, 12, 31), pd.Timestamp(y - 1, 12, 31))
        title = f"Country movers: {y} vs {y - 1}"
    pcmp = structure.period_compare(o, cur, prior)
    c5, c6 = st.columns([3, 2])
    with c5:
        if pcmp["delta_mw"].abs().sum() == 0:
            st.info("No country-level orders in these windows for the current filters.")
        else:
            show(cs.fig_diverging(pcmp["delta_mw"], 10, title), "str_movers")
    with c6:
        show(cs.fig_continent_mix(o), "str_continent")
    table_view(pcmp.sort_values("delta_mw", ascending=False).round(3), "Data table: country movers")
    c7, c8 = st.columns(2)
    markets = structure.market_status(o, as_of)
    with c7:
        show(cs.fig_breadth(conc), "str_breadth")
    with c8:
        if not markets.empty:
            show(cs.fig_market_status(markets, 16), "str_markets")
    if not markets.empty:
        counts = markets["status"].value_counts()
        st.caption("Market status by months since the last order: Active <= 12, Cooling 13-36, Dormant > 36; Emerging = first order within the last 5 years. " + ", ".join(f"{k}: {v}" for k, v in counts.items()) + ".")
        table_view(markets.round(1), "Data table: market lifecycle")


# --------------------------------------------------------------------------------------
def _section_accounts(intel: Intel, orders_f: pd.DataFrame) -> None:
    _scope_note("This section respects the sidebar filters. Customer spellings are grouped (for example three Greenvolt names).")
    o = prepare_orders(orders_f)
    if o.empty:
        st.info("No orders after filtering.")
        return
    radar = structure.account_radar(o, intel.as_of)
    if radar.empty:
        st.info("No disclosed customers in the filtered data.")
        return
    min_mw = st.slider("Minimum lifetime MW for an account to appear", 25, 1500, 100, step=25, key="acc_min_mw")
    key_accounts = radar[(radar["lifetime_mw"] >= min_mw) & radar["repeat_buyer"]]
    summ = structure.account_status_summary(radar, float(min_mw), repeat_only=True)
    m = st.columns(4)
    for col, status in zip(m[:3], ["Active", "Cooling", "Dormant"]):
        n = int(summ.loc[status, "accounts"]) if status in summ.index else 0
        mw = float(summ.loc[status, "lifetime_mw"]) if status in summ.index else 0.0
        col.metric(f"{status} repeat accounts", f"{n}", f"{mw / 1000:.1f} GW lifetime", delta_color="off", delta_arrow="off")
    m[3].metric("Repeat buyers", f"{int(radar['repeat_buyer'].sum())} of {len(radar)}", "named accounts", delta_color="off", delta_arrow="off")
    show(cs.fig_account_radar(radar, float(min_mw)), "acc_radar")
    st.caption("Status by time since the last announced order: Active <= 12 months, Cooling 12-24, Dormant > 24. Bubble size reflects the number of orders.")
    c1, c2 = st.columns([2, 3])
    with c1:
        show(cs.fig_diverging(radar.set_index("customer")["ltm_delta_mw"], 8, "Account movers: last 12 months vs prior 12 months", height=520), "acc_movers")
    with c2:
        st.markdown("**Re-engagement watchlist** - repeat accounts that have gone quiet")
        status_pick = st.multiselect("Status", ["Cooling", "Dormant", "Active"], default=["Cooling", "Dormant"], key="acc_status_pick")
        wl = key_accounts[key_accounts["status"].isin(status_pick)].copy().sort_values("lifetime_mw", ascending=False)
        view = wl[["customer", "status", "orders", "lifetime_mw", "last_order", "days_since_last", "median_gap_days", "top_country"]].rename(
            columns={"customer": "Customer", "status": "Status", "orders": "Orders", "lifetime_mw": "Lifetime MW", "last_order": "Last order", "days_since_last": "Days since",
                     "median_gap_days": "Typical gap (days)", "top_country": "Main country"}
        )
        st.dataframe(view.round(0), width="stretch", hide_index=True, height=470)
        _csv_button(view, "Download watchlist (CSV)", "ge_account_watchlist.csv", "acc_dl")


# --------------------------------------------------------------------------------------
def _section_compare(intel: Intel) -> None:
    _scope_note("Head-to-head comparison uses the full dataset and ignores the sidebar filters.")
    o, p, as_of = intel.orders, intel.platforms, intel.as_of
    st.subheader("Compare Lab")
    dim = st.radio("Compare", structure.COMPARE_DIMENSIONS, horizontal=True, key="cmp_dim")
    options = structure.entity_options(o, p, dim)
    if len(options) < 2:
        st.info("Not enough entities to compare.")
        return
    ca, cb = st.columns(2)
    a = ca.selectbox("Side A", options, index=0, key=f"cmp_a_{dim}")
    b = cb.selectbox("Side B", options, index=1, key=f"cmp_b_{dim}")
    pa = structure.entity_profile(o, p, dim, a, as_of)
    pb = structure.entity_profile(o, p, dim, b, as_of)

    def fmt(profile: dict) -> dict[str, str]:
        return {
            "Announced MW": f"{profile['mw']:,.0f}",
            "Share of total MW": pc(profile["share_of_total"], 1),
            "Orders": f"{profile['orders']:,}",
            "Average order (MW)": f"{profile['avg_order_mw']:,.0f}",
            "Largest order (MW)": f"{profile['largest_order_mw']:,.0f}",
            "Last 12 months (MW)": f"{profile['ltm_mw']:,.0f}",
            "Prior 12 months (MW)": f"{profile['prior_ltm_mw']:,.0f}",
            "12-month change": pc(profile["ltm_growth"], 0, signed=True),
            "Share on 5.5 MW-and-larger turbines": pc(profile["large_turbine_share"]),
            "Avg turbine rating (MW)": "n/a" if pd.isna(profile["avg_rating_mw"]) else f"{profile['avg_rating_mw']:.1f}",
            "Avg rotor (m)": "n/a" if pd.isna(profile["avg_rotor_m"]) else f"{profile['avg_rotor_m']:.0f}",
            "First order": str(profile["first_order"]),
            "Last order": str(profile["last_order"]),
            "Active order years": f"{profile['active_years']}",
            "Countries": f"{profile['countries']}",
            "Named customers": f"{profile['customers']}",
        }

    table = pd.DataFrame({a: fmt(pa), b: fmt(pb)})
    table.index.name = "Metric"
    st.dataframe(table.reset_index(), width="stretch", height=580, hide_index=True, column_config={"Metric": st.column_config.TextColumn("Metric", width="large")})
    c1, c2 = st.columns(2)
    with c1:
        show(cs.fig_compare_yearly(structure.entity_yearly(o, p, dim, a), structure.entity_yearly(o, p, dim, b)), "cmp_yearly")
    with c2:
        by_options = [x for x in ["Country", "Customer", "Platform family"] if x != dim]
        by = st.selectbox("Break down by", by_options, index=0, key=f"cmp_by_{dim}")
        show(cs.fig_compare_mix(structure.entity_breakdown(o, p, dim, a, by), structure.entity_breakdown(o, p, dim, b, by), a, b, by), "cmp_mix")
    st.caption("Platform-family comparisons count platform-slot MW; other comparisons count order MW. 'Other' groups the tail beyond the top eight.")


# --------------------------------------------------------------------------------------
SEVERITY_LABEL = {"High": "▲ High", "Medium": "● Medium", "Low": "○ Low", "OK": "✓ OK", "Info": "i Info"}


def _section_quality(intel: Intel) -> None:
    _scope_note("Data-quality section: full dataset, ignoring the sidebar filters.")
    st.subheader("Data quality and reconciliation")
    summary = intel.anomalies.copy()
    m = st.columns(4)
    m[0].metric("High-severity checks", int((summary["Severity"] == "High").sum()), delta_color="off", delta_arrow="off")
    m[1].metric("Medium", int((summary["Severity"] == "Medium").sum()), delta_color="off", delta_arrow="off")
    m[2].metric("Passed / informational", int(summary["Severity"].isin(["OK", "Info"]).sum()), delta_color="off", delta_arrow="off")
    m[3].metric("Orders last checked", str((intel.meta or {}).get("orders_checked_through", "n/a")), delta_color="off", delta_arrow="off")
    summary["Severity"] = summary["Severity"].map(lambda s: SEVERITY_LABEL.get(s, s))
    st.table(summary.set_index("Check"))
    if (intel.meta or {}).get("orders_check_note"):
        st.caption(intel.meta["orders_check_note"])
    c1, c2 = st.columns(2)
    with c1:
        show(co.fig_announced_vs_orders(intel), "dq_ann")
    with c2:
        show(cs.fig_completeness_heatmap(intel.completeness), "dq_complete")
    if intel.results:
        with st.expander("Quarterly results used"):
            st.dataframe(intel.res[["label", "orders_usd_m", "revenue_usd_m", "ebitda_usd_m"]].rename(columns={"label": "Quarter"}), width="stretch", hide_index=True)
            st.caption(f"Source: {intel.results['source']} (retrieved {intel.results['retrieved']}). {intel.results['notes']}")
    details = quality.anomaly_rows(intel.orders)
    if details:
        st.markdown("**Record-level detail**")
        for name, frame in details.items():
            with st.expander(f"{name} ({len(frame)})"):
                st.dataframe(frame, width="stretch", hide_index=True)
    with st.expander("Definitions and method notes"):
        st.markdown(
            "- **Reported results**: GE Vernova Wind segment (onshore, offshore, LM Wind Power, services), USD m, calendar quarters.\n"
            "- **Announced MW**: order sizes from the yearly sheets and press releases; where a release states the booking quarter that quarter is used, otherwise the announcement date.\n"
            f"- **Mega / jumbo orders**: {MEGA_MW:,.0f} MW and above / {GIGA_MW:,.0f} MW and above.\n"
            "- **Book-to-bill**: orders / revenue in the same period; **backlog cover**: year-end backlog / the year's revenue.\n"
            "- **Guidance arithmetic**: what the rest of the year must deliver for the stated outlook; 'low double digits' is read as a 10-13% decline.\n"
            "- **Customers**: spellings are grouped; platform names are normalised."
        )
