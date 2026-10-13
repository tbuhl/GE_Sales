# GE Wind Sales Intelligence Dashboard

Interactive Streamlit dashboard for GE Vernova's Wind business (orders, results and economics), built on public data.

## What This App Includes
- **Economy Summary** and the existing **Year-by-Year**, **Quarterly**, **Across Years**, **Platform** and **Country** analytics on the announced order list.
- **Executive Briefing**: Wind orders, revenue and segment EBITDA from GE Vernova's quarterly earnings releases (USD), last-four-quarter book-to-bill,
  what the rest of 2026 must deliver for the Wind guidance, backlog, and ten auto-generated, number-backed findings plus a watchlist.
- **Intelligence Lab** (choose a section at the top):
  - `Momentum & Outlook`: reported results by quarter with the release commentary, half-year view, guidance arithmetic and an orders scenario,
    announced MW against reported orders (by booking quarter), cumulative curves, and annual economics (revenue mix, backlog, installed base).
  - `Market Structure`: order size and lumpiness, product mix (platform families, 5.5 MW-and-larger share), country movers, market lifecycle.
  - `Accounts`: customer radar with Active / Cooling / Dormant status and a re-engagement watchlist (customer spellings are grouped).
  - `Compare Lab`: head-to-head profiles by Country, Continent, Customer or Platform family.
  - `Data Quality`: severity-ranked checks, reconciliation against the earnings releases, completeness heatmap.
- Latest GE wind-related news and patents (WIPO PATENTSCOPE), information/disclaimer page and a dark-mode toggle.

## Data sources
- **Workbook** `GE_data.xls` (auto-detected in the current folder, the app folder or its parent): yearly order sheets and the economy sheet. Not committed.
- **Parsed cache** `data_cache/ge_parsed_data.pkl`: the workbook plus the press-release orders, so the app runs without the workbook.
- **`data/ge_quarterly_results.json`**: Wind segment orders, revenue and EBITDA (USD m) by calendar quarter from the earnings releases, annual totals,
  per-quarter commentary and the 2026 guidance. "Low double digits" revenue decline is interpreted as 10-13%; this is flagged in the app.
- **`data/ge_orders_supplement.json`**: orders announced after the workbook's last row (GE Vernova press releases), with stated booking quarters and
  `unlisted_bookings` (for example the 1.1 GW US repower booked in 2025 without detail). It also records the date orders were last checked.
- **`data/ge_stock_monthly.json`**, **`data/market_prices_monthly.json`**: GE Vernova share price (GEV, listed April 2024) and steel / copper futures (Yahoo Finance).

## Maintenance
```powershell
python tools/refresh_market_data.py    # share price, steel and copper to the latest month
python tools/rebuild_cache.py          # re-parse the workbook and merge the press-release orders (rerun after editing the supplement)
python -m pytest tests -q              # tests for the intelligence layer
```

## Run locally
```powershell
cd C:\Users\forsk\Documents\python\GE_sales
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Method notes
- **Announced MW** counts individually announced onshore turbine orders. It is not total orders: reported Wind orders also include services and
  offshore, and several bookings are never announced. The Data Quality tab shows the gap.
- Where a release states the booking quarter (for example Q4 2025 orders announced in March-April 2026) that quarter is used for comparisons with reported orders.
- The orders scenario uses the lowest, median and highest of the last eight reported quarters; it is a range, not a forecast.
- Intelligence tabs use the full dataset except the Market Structure and Accounts sections, which follow the sidebar filters.

## Notes
- `xlrd` is required because the source workbook is `.xls`.
- News/patent filters are restricted to GE + wind/turbine relevance.
- This dashboard is unofficial and provided for informational use only.
