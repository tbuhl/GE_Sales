# GE Wind Sales Intelligence Dashboard

Streamlit dashboard for GE wind-turbine order intake data (2020-2025 currently) with:

- Overall economics (when economy values are available in the workbook)
- Year-by-year and across-years analytics
- Country, platform, delivery, and correlation views
- Latest GE wind-related news (20 items)
- Latest GE wind-related patents (WIPO PATENTSCOPE)
- Information/disclaimer page and dark-mode toggle

## Data source

- Workbook: `C:\Users\forsk\Documents\python\GE_data.xls`
- Parsed cache: `data_cache/ge_parsed_data.pkl`

The app auto-detects the workbook from:

1. Current folder
2. App folder (`GE_sales`)
3. Parent folder (`C:\Users\forsk\Documents\python`)

## Run locally

```powershell
cd C:\Users\forsk\Documents\python\GE_sales
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Notes

- `xlrd` is required because the source workbook is `.xls`.
- News/patent filters are restricted to GE + wind/turbine relevance.
- This dashboard is unofficial and provided for informational use only.
