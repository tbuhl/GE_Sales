"""Charts for the Executive Briefing and the Momentum & Outlook section."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import outlook
from .bundle import Intel
from .common import GIGA_MW
from .theme import style_figure, tokens, year_axis


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _bar(t: dict, color: str | None = None) -> dict:
    return dict(color=color or t["blue"], line=dict(color=t["surface"], width=2))


def _line(t: dict, color: str, width: float = 2.4) -> dict:
    return dict(color=color, width=width)


def _y2(t: dict, title: str, fmt: str | None = None) -> dict:
    d = dict(overlaying="y", side="right", showgrid=False, title=dict(text=title, font=dict(color=t["muted"], size=11)))
    if fmt:
        d["tickformat"] = fmt
    return d


# --------------------------------------------------------------------------------------
# Reported Wind results
# --------------------------------------------------------------------------------------
def fig_orders_revenue(intel: Intel, height: int = 440) -> go.Figure:
    t = tokens()
    r = intel.res
    fig = go.Figure()
    fig.add_trace(go.Bar(x=r["label"], y=r["orders_usd_m"], name="Orders", marker=_bar(t), hovertemplate="%{x}: USD %{y:,.0f}m orders<extra></extra>"))
    fig.add_trace(go.Bar(x=r["label"], y=r["revenue_usd_m"], name="Revenue", marker=_bar(t, t["neutral"]), hovertemplate="%{x}: USD %{y:,.0f}m revenue<extra></extra>"))
    style_figure(fig, title="Wind segment: orders and revenue per quarter (USD m)", height=height, ytitle="USD m", hovermode="x unified")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    fig.update_xaxes(type="category", tickangle=-45)
    return fig


def fig_ebitda(intel: Intel, height: int = 400) -> go.Figure:
    t = tokens()
    r = intel.res
    colors = [t["down"] if v < 0 else t["up"] for v in r["ebitda_usd_m"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=r["label"], y=r["ebitda_usd_m"], name="Segment EBITDA (USD m)", marker=dict(color=colors, line=dict(color=t["surface"], width=2)),
                         hovertemplate="%{x}: USD %{y:,.0f}m<extra>EBITDA</extra>"))
    fig.add_trace(go.Scatter(x=r["label"], y=r["ebitda_margin"], name="Margin (right axis)", yaxis="y2", mode="lines+markers", line=_line(t, t["ink"], 2.0),
                             marker=dict(size=7, line=dict(color=t["surface"], width=2)), hovertemplate="%{x}: %{y:.1%}<extra>Margin</extra>"))
    fig.add_hline(y=0, line=dict(color=t["ink2"], width=1.2))
    style_figure(fig, title="Segment EBITDA per quarter and margin", height=height, ytitle="USD m", hovermode="x unified")
    fig.update_layout(bargap=0.3, yaxis2=_y2(t, "Margin", ".0%"))
    fig.update_xaxes(type="category", tickangle=-45)
    return fig


def fig_year_orders(intel: Intel, height: int = 440) -> go.Figure:
    """Wind orders per year with the running year and its scenario."""
    t = tokens()
    pr = intel.projection
    yt = intel.yt
    year_now = pr["year"] if pr else int(yt.index.max())
    done = yt[yt.index < year_now]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=done.index, y=done["orders_usd_m"], name="Orders", marker=_bar(t), hovertemplate="%{x}: USD %{y:,.0f}m<extra></extra>"))
    if pr:
        ytd, base = pr["ytd"]["orders_usd_m"], pr["orders_scenario"]["base"]
        sc = pr["orders_scenario"]
        fig.add_trace(go.Bar(x=[year_now], y=[ytd], name=f"{year_now} reported to date", marker=_bar(t, t["ink2"]), hovertemplate=f"{year_now}: USD " + "%{y:,.0f}m to date<extra></extra>"))
        fig.add_trace(go.Bar(x=[year_now], y=[max(base - ytd, 0)], name="Base scenario for the rest of the year",
                             marker=dict(color=rgba(t["ink2"], 0.30), line=dict(color=t["ink2"], width=1.5), pattern=dict(shape="/", fgcolor=t["ink2"], size=7)),
                             hovertemplate="Base scenario: USD %{y:,.0f}m<extra></extra>"))
        fig.add_trace(go.Scatter(x=[year_now], y=[base], mode="markers", name="Scenario range (low / base / high)",
                                 marker=dict(symbol="diamond-open", size=11, color=t["ink"], line=dict(width=2, color=t["ink"])),
                                 error_y=dict(type="data", symmetric=False, array=[sc["strong"] - base], arrayminus=[base - sc["conservative"]], color=t["ink"], thickness=1.6, width=7),
                                 hovertemplate=f"Base USD {base:,.0f}m (range {sc['conservative']:,.0f}-{sc['strong']:,.0f})<extra></extra>"))
    style_figure(fig, title="Wind orders by year (USD m)", height=height, ytitle="USD m", hovermode="x unified")
    fig.update_layout(barmode="stack", bargap=0.28)
    year_axis(fig)
    return fig


def fig_halves(intel: Intel, height: int = 400) -> go.Figure:
    t = tokens()
    h = intel.halves.reset_index()
    h["label"] = h["half"] + " " + h["year"].astype(int).astype(str)
    colors = [t["down"] if v < 0 else t["up"] for v in h["ebitda_usd_m"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=h["label"], y=h["ebitda_usd_m"], marker=dict(color=colors, line=dict(color=t["surface"], width=2)), name="Segment EBITDA",
                         customdata=h["ebitda_margin"], hovertemplate="%{x}: USD %{y:,.0f}m (%{customdata:.1%})<extra></extra>"))
    pr = intel.projection
    if pr and "ebitda_needed_remaining" in pr:
        fig.add_trace(go.Bar(x=[f"H2 {pr['year']} needed"], y=[pr["ebitda_needed_remaining"]], name="Needed for guidance",
                             marker=dict(color=rgba(t["aqua"], 0.4), line=dict(color=t["aqua"], width=1.5), pattern=dict(shape="/", fgcolor=t["aqua"], size=7)),
                             hovertemplate="H2 needed for the guided full-year loss: USD %{y:,.0f}m<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=t["ink2"], width=1.2))
    style_figure(fig, title="Segment EBITDA by half-year, and what H2 needs for guidance", height=height, ytitle="USD m", hovermode="x unified", legend=False)
    fig.update_layout(bargap=0.3)
    fig.update_xaxes(type="category")
    return fig


def fig_announced_vs_orders(intel: Intel, first_year: int = 2020, height: int = 440) -> go.Figure:
    t = tokens()
    a = intel.ann_y.loc[first_year:]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=a.index, y=a["announced_mw"], name="Announced MW (by booking year)", marker=_bar(t), hovertemplate="%{x}: %{y:,.0f} MW<extra>Announced</extra>"))
    if a["unlisted_mw"].sum() > 0:
        fig.add_trace(go.Bar(x=a.index, y=a["unlisted_mw"], name="Announced in aggregate only (no detail)", marker=dict(color=rgba(t["orange"], 0.5), line=dict(color=t["orange"], width=1.5), pattern=dict(shape="/", fgcolor=t["orange"], size=7)),
                             hovertemplate="%{x}: %{y:,.0f} MW<extra>Unlisted</extra>"))
    rep = a["reported_orders_usd_m"]
    fig.add_trace(go.Scatter(x=a.index, y=rep, name="Reported Wind orders, USD m (right axis)", yaxis="y2", mode="lines+markers", line=_line(t, t["ink"], 2.2),
                             marker=dict(size=8, symbol="diamond-open", line=dict(width=2, color=t["ink"])), hovertemplate="%{x}: USD %{y:,.0f}m<extra>Reported orders</extra>"))
    style_figure(fig, title="Announced MW against reported Wind orders", height=height, ytitle="MW", hovermode="x unified")
    fig.update_layout(barmode="stack", bargap=0.28, yaxis2=_y2(t, "USD m"))
    fig.update_yaxes(rangemode="tozero")
    year_axis(fig)
    return fig


def fig_cumulative_curves(intel: Intel, first_year: int = 2020, height: int = 470, width_hint: int = 540) -> go.Figure:
    t = tokens()
    year = intel.as_of.year
    curves = outlook.cumulative_curves(intel.orders, intel.as_of, first_year)
    past = [int(y) for y in sorted(curves["year"].unique()) if y != year]
    finals = curves[curves["year"] != year].groupby("year")["cum_mw"].max()
    label_years = set(finals.nlargest(2).index.tolist()) | {year - 1}
    fig = go.Figure()
    first_gray = True
    for y in past:
        c = curves[curves["year"] == y]
        prev = bool(y == year - 1)
        fig.add_trace(go.Scatter(
            x=c["plot_date"], y=c["cum_mw"], mode="lines", name=str(y) if prev else f"Earlier years ({first_year}-{year - 2})", legendgroup="prev" if prev else "earlier",
            showlegend=prev or first_gray, line=dict(color=t["orange"] if prev else t["neutral"], width=2.4 if prev else 1.2), opacity=1.0 if prev else 0.55,
            hovertemplate=f"{y}: " + "%{y:,.0f} MW at %{x|%d %b}<extra></extra>"))
        if not prev:
            first_gray = False
        if y in label_years:
            fig.add_annotation(x=c["plot_date"].iloc[-1], y=c["cum_mw"].iloc[-1], text=str(y), showarrow=False, xanchor="left", xshift=4, font=dict(size=10, color=t["muted"]))
    cur = curves[curves["year"] == year]
    fig.add_trace(go.Scatter(x=cur["plot_date"], y=cur["cum_mw"], mode="lines", name=str(year), line=dict(color=t["blue"], width=3.6),
                             hovertemplate=f"{year}: " + "%{y:,.0f} MW at %{x|%d %b}<extra></extra>"))
    big = intel.orders[(intel.orders["order_year"] == year) & (intel.orders["size_mw"] >= 150)]
    if len(big):
        pts = cur.set_index("date").reindex(big["order_date"].dt.normalize())
        fig.add_trace(go.Scatter(x=pts["plot_date"], y=pts["cum_mw"], mode="markers", name=f"Orders of 150 MW+ in {year}",
                                 marker=dict(size=11, color=t["blue"], line=dict(color=t["surface"], width=2.5), symbol="diamond"),
                                 text=[f"{c} - {m:,.0f} MW" for c, m in zip(big["customer"], big["size_mw"])], hovertemplate="%{text}<br>%{x|%d %b}<extra></extra>"))
    style_figure(fig, title=f"Cumulative announced MW: {year} (to {intel.as_of:%d %b}) against every year since {first_year}", height=height, ytitle="Cumulative MW", hovermode="closest", width_hint=width_hint)
    fig.update_xaxes(tickformat="%b", range=[pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-25")])
    return fig


# --------------------------------------------------------------------------------------
# Annual economics (workbook)
# --------------------------------------------------------------------------------------
def fig_revenue_mix(intel: Intel, height: int = 400) -> go.Figure:
    t = tokens()
    e = intel.econ.dropna(subset=["revenue_equipment", "revenue_services"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=e.index, y=e["revenue_equipment"], name="Equipment", marker=_bar(t), hovertemplate="%{x}: USD %{y:,.0f}m<extra>Equipment</extra>"))
    fig.add_trace(go.Bar(x=e.index, y=e["revenue_services"], name="Services", marker=_bar(t, t["orange"]), hovertemplate="%{x}: USD %{y:,.0f}m<extra>Services</extra>"))
    fig.add_trace(go.Scatter(x=e.index, y=e["services_share"], name="Services share (right axis)", yaxis="y2", mode="lines+markers", line=_line(t, t["ink"], 2.0),
                             marker=dict(size=7, line=dict(color=t["surface"], width=2)), hovertemplate="%{x}: %{y:.1%}<extra>Services share</extra>"))
    style_figure(fig, title="Wind revenue: equipment and services (USD m)", height=height, ytitle="USD m", hovermode="x unified")
    fig.update_layout(barmode="stack", bargap=0.4, yaxis2=_y2(t, "Share", ".0%"))
    year_axis(fig)
    return fig


def fig_revenue_segments(intel: Intel, height: int = 400) -> go.Figure:
    t = tokens()
    e = intel.econ.dropna(subset=["revenue_onshore"])
    fig = go.Figure()
    for col, name, color in (("revenue_onshore", "Onshore Wind", t["blue"]), ("revenue_offshore", "Offshore Wind", t["orange"]), ("revenue_lm", "LM Wind Power (blades)", t["aqua"])):
        fig.add_trace(go.Bar(x=e.index, y=e[col], name=name, marker=_bar(t, color), hovertemplate="%{x}: USD %{y:,.0f}m<extra>" + name + "</extra>"))
    style_figure(fig, title="Wind revenue by business (USD m; the three overlap with intersegment sales)", height=height, ytitle="USD m", hovermode="x unified")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06)
    year_axis(fig)
    return fig


def fig_backlog(intel: Intel, height: int = 380) -> go.Figure:
    t = tokens()
    e = intel.econ.dropna(subset=["backlog"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=e.index, y=e["backlog"], name="Backlog at year-end (USD m)", marker=_bar(t), hovertemplate="%{x}: USD %{y:,.0f}m<extra></extra>"))
    fig.add_trace(go.Scatter(x=e.index, y=e["backlog_cover_years"], name="Cover, years of revenue (right axis)", yaxis="y2", mode="lines+markers", line=_line(t, t["orange"]),
                             marker=dict(size=8, line=dict(color=t["surface"], width=2)), hovertemplate="%{x}: %{y:.1f} years<extra>Cover</extra>"))
    style_figure(fig, title="Wind backlog at year-end", height=height, ytitle="USD m", hovermode="x unified")
    fig.update_layout(bargap=0.4, yaxis2=_y2(t, "Years"))
    fig.update_yaxes(rangemode="tozero")
    year_axis(fig)
    return fig


def fig_installed_base(intel: Intel, height: int = 380) -> go.Figure:
    t = tokens()
    e = intel.econ.dropna(subset=["installed_turbines"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=e.index, y=e["installed_turbines"], name="Installed base (turbines)", marker=_bar(t), hovertemplate="%{x}: %{y:,.0f} turbines<extra></extra>"))
    fig.add_trace(go.Bar(x=e.index, y=e["services_turbines"], name="Covered by service agreements", marker=_bar(t, t["orange"]), hovertemplate="%{x}: %{y:,.0f} turbines<extra>Serviced</extra>"))
    style_figure(fig, title="Installed base and service coverage (turbines)", height=height, ytitle="Turbines", hovermode="x unified")
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.06)
    year_axis(fig)
    return fig
