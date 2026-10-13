from __future__ import annotations

import html
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from intel.common import load_quarterly_results
from intel.ui import get_intel, render_executive_briefing, render_intelligence_lab


APP_TITLE = "GE Wind Sales Intelligence Dashboard"
DATA_DIR = Path("data")
DATA_CACHE_DIR = Path("data_cache")
PARSED_DATA_FILE = DATA_CACHE_DIR / "ge_parsed_data.pkl"
STOCK_MONTHLY_FILE = DATA_DIR / "ge_stock_monthly.json"
MARKET_MONTHLY_FILE = DATA_DIR / "market_prices_monthly.json"
SUPPLEMENT_FILE = DATA_DIR / "ge_orders_supplement.json"

NEWS_FEEDS = [
    {
        "source": "Google News (GE Vernova official)",
        "url": "https://news.google.com/rss/search?q=site%3Agevernova.com%20%22wind%22%20%22press%20release%22&hl=en-US&gl=US&ceid=US:en",
        "focus": "company",
    },
    {
        "source": "Google News (Reuters + GE Vernova wind)",
        "url": "https://news.google.com/rss/search?q=site%3Areuters.com%20%22GE%20Vernova%22%20wind&hl=en-US&gl=US&ceid=US:en",
        "focus": "financial",
    },
    {
        "source": "Google News (Bloomberg + GE Vernova wind)",
        "url": "https://news.google.com/rss/search?q=site%3Abloomberg.com%20%22GE%20Vernova%22%20wind&hl=en-US&gl=US&ceid=US:en",
        "focus": "financial",
    },
    {
        "source": "Google News (Windpower Monthly + GE Vernova)",
        "url": "https://news.google.com/rss/search?q=site%3Awindpowermonthly.com%20%22GE%20Vernova%22%20wind&hl=en-US&gl=US&ceid=US:en",
        "focus": "industry",
    },
    {
        "source": "Google News (OffshoreWIND.biz + GE Vernova)",
        "url": "https://news.google.com/rss/search?q=site%3Aoffshorewind.biz%20%22GE%20Vernova%22%20wind&hl=en-US&gl=US&ceid=US:en",
        "focus": "industry",
    },
    {
        "source": "Google News (Power Technology + GE Vernova wind)",
        "url": "https://news.google.com/rss/search?q=site%3Apower-technology.com%20%22GE%20Vernova%22%20wind&hl=en-US&gl=US&ceid=US:en",
        "focus": "technical",
    },
]

GE_KEYWORDS = (
    "ge vernova",
    "ge renewable energy",
    "general electric",
    "ge wind",
    "lm wind power",
)
WIND_TOPIC_KEYWORDS = ("wind", "turbine", "onshore", "offshore", "repower")
GE_WIND_NEWS_BLACKLIST = (
    "careers",
    "/career",
    "/jobs",
    "grid solutions",
    "gas power",
    "steam power",
    "nuclear",
    "healthcare",
    "aviation",
    "appliances",
    "stock price & latest news",
    "deals of the week",
    "wind power patents",
    "weekly roundup",
)

GE_TITLE_RE = re.compile(r"\b(?:ge\s+vernova|ge\s+renewable\s+energy|general\s+electric|lm\s+wind\s+power)\b", re.IGNORECASE)
PATENTSCOPE_SEARCH_URL = (
    "https://patentscope.wipo.int/search/en/result.jsf"
    "?query=PA%3A%22General%20Electric%22%20AND%20wind"
    "&sortOption=-DP"
)

DEFAULT_ECON_METRICS = [
    "order intake [mUSD]",
    "installed capacity [MW]",
    "sales",
    "gross revenue",
    "employees eoy",
    "staff cost [mUSD]",
]

PLATFORM_SLOT_SPECS = [
    {
        "slot": 1,
        "qty": ["#", "No_WTG_1"],
        "platform": ["platform"],
        "rotor": ["Rotor", "Rotor_1"],
        "mw": ["MW", "MW_1"],
    },
    {
        "slot": 2,
        "qty": ["#.1", "No_WTG_2"],
        "platform": ["Multi platform"],
        "rotor": ["Rotor.1", "Rotor_2"],
        "mw": ["MW.1", "MW_2"],
    },
    {
        "slot": 3,
        "qty": ["#.2", "No_WTG_3"],
        "platform": ["Multi platform.1"],
        "rotor": ["Rotor.2", "Rotor"],
        "mw": ["MW.2", "MW"],
    },
    {
        "slot": 4,
        "qty": ["#.3", "No_WTG_4"],
        "platform": ["Multi platform.2"],
        "rotor": ["Rotor.3", "Rotor.1"],
        "mw": ["MW.3", "MW.1"],
    },
]

COUNTRY_MAP = {
    "usa": "United States",
    "united states": "United States",
    "united states and canada": "United States and Canada",
    "uk": "United Kingdom",
    "uk - offshore": "United Kingdom - Offshore",
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
    "irland": "Ireland",
    "spanien": "Spain",
    "mongoliet": "Mongolia",
    "den dominikanske republik": "Dominican Republic",
    "dominican republic": "Dominican Republic",
    "?": "Unknown",
    "no info": "Unknown",
    "notmentioned": "Unknown",
    "un disclosed": "Undisclosed",
}

GEO_COUNTRY_OVERRIDES = {
    "United States and Canada": "United States",
    "United Kingdom - Offshore": "United Kingdom",
    "Wales": "United Kingdom",
    "Unknown": None,
    "EU": None,
    "NotMentioned": None,
    "Undisclosed": None,
}

COUNTRY_ISO3_MAP = {
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Belgium": "BEL",
    "Bolivia": "BOL",
    "Brazil": "BRA",
    "Bulgaria": "BGR",
    "Canada": "CAN",
    "Cape Verde": "CPV",
    "Chile": "CHL",
    "China": "CHN",
    "Colombia": "COL",
    "Costa Rica": "CRI",
    "Croatia": "HRV",
    "Cyprus": "CYP",
    "Denmark": "DNK",
    "Dominican Republic": "DOM",
    "Egypt": "EGY",
    "El Salvador": "SLV",
    "Estonia": "EST",
    "Finland": "FIN",
    "France": "FRA",
    "Germany": "DEU",
    "Greece": "GRC",
    "Guatemala": "GTM",
    "Honduras": "HND",
    "Hungary": "HUN",
    "India": "IND",
    "Ireland": "IRL",
    "Italy": "ITA",
    "Jamaica": "JAM",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Kazakhstan": "KAZ",
    "Kenya": "KEN",
    "Latvia": "LVA",
    "Lithuania": "LTU",
    "Mexico": "MEX",
    "Mongolia": "MNG",
    "Morocco": "MAR",
    "Netherlands": "NLD",
    "New Zealand": "NZL",
    "Nicaragua": "NIC",
    "Norway": "NOR",
    "Pakistan": "PAK",
    "Panama": "PAN",
    "Peru": "PER",
    "Philippines": "PHL",
    "Poland": "POL",
    "Portugal": "PRT",
    "Puerto Rico": "PRI",
    "Romania": "ROU",
    "Russia": "RUS",
    "Saudi Arabia": "SAU",
    "Senegal": "SEN",
    "Serbia": "SRB",
    "South Africa": "ZAF",
    "South Korea": "KOR",
    "Spain": "ESP",
    "Sri Lanka": "LKA",
    "Sweden": "SWE",
    "Taiwan": "TWN",
    "Thailand": "THA",
    "Turkey": "TUR",
    "Ukraine": "UKR",
    "United Kingdom": "GBR",
    "United States": "USA",
    "Uruguay": "URY",
    "Vietnam": "VNM",
}

COUNTRY_REGION_CONTINENT = {
    "Argentina": ("South America", "Americas"),
    "Australia": ("Oceania", "Oceania"),
    "Austria": ("Western Europe", "Europe"),
    "Belgium": ("Western Europe", "Europe"),
    "Bolivia": ("South America", "Americas"),
    "Brazil": ("South America", "Americas"),
    "Bulgaria": ("Eastern Europe", "Europe"),
    "Canada": ("North America", "Americas"),
    "Cape Verde": ("West Africa", "Africa"),
    "Chile": ("South America", "Americas"),
    "China": ("East Asia", "Asia"),
    "Colombia": ("South America", "Americas"),
    "Costa Rica": ("Central America", "Americas"),
    "Croatia": ("Southern Europe", "Europe"),
    "Cyprus": ("Southern Europe", "Europe"),
    "Denmark": ("Northern Europe", "Europe"),
    "Dominican Republic": ("Caribbean", "Americas"),
    "Egypt": ("North Africa", "Africa"),
    "El Salvador": ("Central America", "Americas"),
    "Estonia": ("Northern Europe", "Europe"),
    "EU": ("Europe", "Europe"),
    "Finland": ("Northern Europe", "Europe"),
    "France": ("Western Europe", "Europe"),
    "Germany": ("Western Europe", "Europe"),
    "Greece": ("Southern Europe", "Europe"),
    "Guatemala": ("Central America", "Americas"),
    "Honduras": ("Central America", "Americas"),
    "Hungary": ("Eastern Europe", "Europe"),
    "India": ("South Asia", "Asia"),
    "Ireland": ("Northern Europe", "Europe"),
    "Italy": ("Southern Europe", "Europe"),
    "Jamaica": ("Caribbean", "Americas"),
    "Japan": ("East Asia", "Asia"),
    "Jordan": ("Middle East", "Asia"),
    "Kazakhstan": ("Central Asia", "Asia"),
    "Kenya": ("East Africa", "Africa"),
    "Latvia": ("Northern Europe", "Europe"),
    "Lithuania": ("Northern Europe", "Europe"),
    "Mexico": ("North America", "Americas"),
    "Mongolia": ("East Asia", "Asia"),
    "Morocco": ("North Africa", "Africa"),
    "Netherlands": ("Western Europe", "Europe"),
    "New Zealand": ("Oceania", "Oceania"),
    "Nicaragua": ("Central America", "Americas"),
    "Norway": ("Northern Europe", "Europe"),
    "Pakistan": ("South Asia", "Asia"),
    "Panama": ("Central America", "Americas"),
    "Peru": ("South America", "Americas"),
    "Philippines": ("Southeast Asia", "Asia"),
    "Poland": ("Eastern Europe", "Europe"),
    "Portugal": ("Southern Europe", "Europe"),
    "Puerto Rico": ("Caribbean", "Americas"),
    "Romania": ("Eastern Europe", "Europe"),
    "Russia": ("Eastern Europe", "Europe"),
    "Saudi Arabia": ("Middle East", "Asia"),
    "Senegal": ("West Africa", "Africa"),
    "Serbia": ("Southern Europe", "Europe"),
    "South Africa": ("Southern Africa", "Africa"),
    "South Korea": ("East Asia", "Asia"),
    "Spain": ("Southern Europe", "Europe"),
    "Sri Lanka": ("South Asia", "Asia"),
    "Sweden": ("Northern Europe", "Europe"),
    "Taiwan": ("East Asia", "Asia"),
    "Thailand": ("Southeast Asia", "Asia"),
    "Turkey": ("Middle East", "Asia"),
    "Ukraine": ("Eastern Europe", "Europe"),
    "United Kingdom": ("Northern Europe", "Europe"),
    "United Kingdom - Offshore": ("Northern Europe", "Europe"),
    "United States": ("North America", "Americas"),
    "United States and Canada": ("North America", "Americas"),
    "Unknown": ("Unknown", "Unknown"),
    "Undisclosed": ("Unknown", "Unknown"),
    "Uruguay": ("South America", "Americas"),
    "Vietnam": ("Southeast Asia", "Asia"),
    "Wales": ("Northern Europe", "Europe"),
}

COUNTRY_REGION_CONTINENT_LOWER = {k.lower(): v for k, v in COUNTRY_REGION_CONTINENT.items()}

SUMMARY_LABELS = {
    "# mw",
    "# of sites",
    "# aom5000 energy based yield",
    "# aom4000 time based yield",
    "% aom5000 of confirmed",
    "% of confirmed energy yield sc",
    "services counters",
    "market share",
    "country",
    "type of contract",
    "potential turbines",
    "total # blades",
    "total # turbines incl undisclosed",
    "w. average rotor dia / #",
    "w. average rotor dia / mw",
    "w. average service contract",
    "weighted average",
    "counters",
    "turbines",
    "sales",
    "1 qtr",
    "2 qtr",
    "3 qtr",
    "4 qtr",
    "0-5",
    "5-10",
    "10-15",
    "15-20",
    "20+",
}

SVC_UNKNOWN_VALUES = {"?", "nan", "none", "no info", "not mentioned", "unknown", "0"}
CUSTOMER_UNKNOWN_VALUES = {"?", "nan", "none", "no info", "not mentioned", "unknown"}


def apply_page_style(dark_mode: bool) -> None:
    if dark_mode:
        text_color = "#f5f7fb"
        muted_color = "#a6b3c3"
        bg_main = (
            "radial-gradient(1200px 620px at 96% -12%, rgba(250, 204, 21, 0.16) 0%, rgba(250, 204, 21, 0) 52%),"
            "radial-gradient(900px 520px at -4% 18%, rgba(16, 185, 129, 0.18) 0%, rgba(16, 185, 129, 0) 48%),"
            "linear-gradient(180deg, #090d16 0%, #0d1424 100%)"
        )
        bg_sidebar = "linear-gradient(180deg, rgba(11, 17, 31, 0.95) 0%, rgba(8, 12, 22, 0.98) 100%)"
        header_bg = "rgba(9, 14, 26, 0.82)"
        header_border = "rgba(250, 204, 21, 0.20)"
        card_bg = "rgba(17, 24, 39, 0.62)"
        card_border = "rgba(148, 163, 184, 0.28)"
        card_shadow = "0 24px 56px rgba(0, 0, 0, 0.48), 0 8px 18px rgba(0, 0, 0, 0.25)"
        card_glow = "rgba(250, 204, 21, 0.26)"
        frame_border = "rgba(148, 163, 184, 0.25)"
        metric_label_color = "#d5deea"
        metric_value_color = "#ffffff"
        metric_delta_color = "#f8fafc"
        tab_bg = "rgba(17, 24, 39, 0.66)"
        tab_active_bg = "linear-gradient(120deg, rgba(234, 179, 8, 0.30), rgba(16, 185, 129, 0.28))"
        tab_hover_bg = "rgba(30, 41, 59, 0.80)"
        tab_border = "rgba(148, 163, 184, 0.25)"
        tab_active_border = "rgba(250, 204, 21, 0.40)"
        tab_text_color = "#d7e0ec"
        tab_active_text_color = "#ffffff"
        input_bg = "rgba(15, 23, 42, 0.78)"
        input_border = "rgba(148, 163, 184, 0.30)"
        input_focus = "rgba(250, 204, 21, 0.42)"
        accent_start = "#facc15"
        accent_end = "#10b981"
        link_color = "#a7f3d0"
        link_hover = "#fde047"
        chart_wrap_bg = "rgba(17, 24, 39, 0.56)"
        chart_wrap_border = "rgba(148, 163, 184, 0.25)"
    else:
        text_color = "#0f172a"
        muted_color = "#475569"
        bg_main = (
            "radial-gradient(1200px 620px at 94% -10%, rgba(250, 204, 21, 0.28) 0%, rgba(250, 204, 21, 0) 50%),"
            "radial-gradient(1000px 540px at -8% 20%, rgba(52, 211, 153, 0.24) 0%, rgba(52, 211, 153, 0) 46%),"
            "linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)"
        )
        bg_sidebar = "linear-gradient(180deg, rgba(255, 255, 255, 0.88) 0%, rgba(248, 250, 252, 0.94) 100%)"
        header_bg = "rgba(255, 255, 255, 0.78)"
        header_border = "rgba(15, 23, 42, 0.10)"
        card_bg = "rgba(255, 255, 255, 0.66)"
        card_border = "rgba(15, 23, 42, 0.10)"
        card_shadow = "0 26px 52px rgba(2, 6, 23, 0.12), 0 8px 18px rgba(2, 6, 23, 0.08)"
        card_glow = "rgba(16, 185, 129, 0.20)"
        frame_border = "rgba(15, 23, 42, 0.11)"
        metric_label_color = "#334155"
        metric_value_color = "#0f172a"
        metric_delta_color = "#0f172a"
        tab_bg = "rgba(255, 255, 255, 0.72)"
        tab_active_bg = "linear-gradient(120deg, rgba(254, 240, 138, 0.74), rgba(167, 243, 208, 0.72))"
        tab_hover_bg = "rgba(255, 255, 255, 0.90)"
        tab_border = "rgba(15, 23, 42, 0.12)"
        tab_active_border = "rgba(16, 185, 129, 0.44)"
        tab_text_color = "#334155"
        tab_active_text_color = "#0f172a"
        input_bg = "rgba(255, 255, 255, 0.80)"
        input_border = "rgba(15, 23, 42, 0.14)"
        input_focus = "rgba(16, 185, 129, 0.36)"
        accent_start = "#eab308"
        accent_end = "#10b981"
        link_color = "#0f766e"
        link_hover = "#a16207"
        chart_wrap_bg = "rgba(255, 255, 255, 0.58)"
        chart_wrap_border = "rgba(15, 23, 42, 0.11)"

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,300..700,0..1,-50..200');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');

html, body, .stApp {{
  font-family: 'Nunito Sans', sans-serif;
  color: {text_color};
}}

.stApp {{
  background: {bg_main};
  background-attachment: fixed;
}}

.block-container {{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}}

h1, h2, h3, h4, h5, h6 {{
  letter-spacing: -0.01em;
  font-weight: 700;
}}

h1 {{
  font-weight: 800;
  background-image: linear-gradient(110deg, {accent_start} 0%, {accent_end} 56%, {text_color} 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}}

p, span, label, li {{
  color: {text_color};
}}

a {{
  color: {link_color};
  text-decoration-color: {link_color};
}}

a:hover {{
  color: {link_hover};
  text-decoration-color: {link_hover};
}}

section[data-testid="stSidebar"] {{
  background: {bg_sidebar};
  border-right: 1px solid {frame_border};
  backdrop-filter: blur(9px);
}}

section[data-testid="stSidebar"] * {{
  color: {text_color};
}}

section[data-testid="stSidebar"] > div:first-child {{
  height: 100vh;
}}

section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {{
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 100%;
}}

[data-testid="stHeader"],
[data-testid="stToolbar"] {{
  background: {header_bg} !important;
  border-bottom: 1px solid {header_border} !important;
  backdrop-filter: blur(12px);
}}

[data-testid="stDecoration"] {{
  background: transparent !important;
}}

/* Restore Streamlit/Material icon rendering (prevents ligature text such as keyboard_double_arrow_right) */
[class*="material-icons"],
[class*="material-symbols"],
[data-testid="stHeader"] button span,
[data-testid="stToolbar"] button span,
[data-testid="collapsedControl"] span,
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarNav"] button span,
button[aria-label*="sidebar"] span,
button[aria-label*="Sidebar"] span,
button[title*="sidebar"] span,
button[title*="Sidebar"] span {{
  font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
  font-feature-settings: "liga" 1, "calt" 1;
  -webkit-font-feature-settings: "liga" 1, "calt" 1;
  font-weight: normal;
  font-style: normal;
  letter-spacing: normal;
  text-transform: none;
}}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="input"] > div,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {{
  background: {input_bg};
  border: 1px solid {input_border};
  color: {text_color};
  border-radius: 10px;
}}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within,
section[data-testid="stSidebar"] div[data-baseweb="input"] > div:focus-within,
section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input:focus,
section[data-testid="stSidebar"] div[data-testid="stTextInput"] input:focus {{
  border-color: {input_focus} !important;
  box-shadow: 0 0 0 2px {input_focus} !important;
}}

div[data-testid="stAlert"],
div[data-testid="stDataFrame"] {{
  background: radial-gradient(65% 80% at 50% 0%, {card_glow} 0%, rgba(255,255,255,0) 100%), {card_bg};
  border: 1px solid {card_border};
  border-radius: 14px;
  box-shadow: {card_shadow};
  backdrop-filter: blur(8px);
}}

div[data-testid="stMetric"] {{
  background: radial-gradient(65% 80% at 50% 0%, {card_glow} 0%, rgba(255,255,255,0) 100%), {card_bg};
  border: 1px solid {card_border};
  border-radius: 14px;
  padding: 0.45rem 0.82rem;
  box-shadow: {card_shadow};
  backdrop-filter: blur(8px);
}}

div[data-testid="stMetric"] div[data-testid="stMetricLabel"] {{
  color: {metric_label_color} !important;
  font-weight: 700 !important;
}}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
  color: {metric_value_color} !important;
  letter-spacing: -0.01em;
}}

.st-key-header_kpis div[data-testid="stMetricValue"] {{
  font-size: 1.5rem;
}}

.st-key-header_kpis div[data-testid="stMetricValue"] > div,
.st-key-header_kpis div[data-testid="stMetricDelta"] > div {{
  overflow: visible !important;
  text-overflow: clip !important;
}}

.st-key-header_kpis div[data-testid="stMetricDelta"] > div {{
  white-space: normal !important;
}}

div[data-testid="stMetricDelta"] {{
  color: {metric_delta_color} !important;
  font-size: 0.72rem !important;
  line-height: 1.05 !important;
}}

div[data-testid="stPlotlyChart"] {{
  background: radial-gradient(65% 80% at 50% 0%, {card_glow} 0%, rgba(255,255,255,0) 100%), {chart_wrap_bg};
  border: 1px solid {chart_wrap_border};
  border-radius: 16px;
  padding: 0.35rem 0.45rem 0.2rem 0.45rem;
  box-shadow: {card_shadow};
  backdrop-filter: blur(8px);
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 0.4rem;
  padding-bottom: 0.1rem;
}}

.stTabs [data-baseweb="tab"] {{
  background: {tab_bg};
  border: 1px solid {tab_border};
  color: {tab_text_color} !important;
  border-radius: 11px 11px 0 0;
  padding-top: 0.42rem;
  padding-bottom: 0.48rem;
  transition: all 0.18s ease;
}}

.stTabs [data-baseweb="tab"]:hover {{
  background: {tab_hover_bg};
  border-color: {tab_active_border};
  transform: translateY(-1px);
}}

.stTabs [data-baseweb="tab"][aria-selected="true"] {{
  background: {tab_active_bg} !important;
  color: {tab_active_text_color} !important;
  border-color: {tab_active_border} !important;
  box-shadow: 0 10px 24px rgba(2, 6, 23, 0.18);
}}

div[data-testid="stVerticalBlock"] div[data-testid="stDataFrame"] {{
  border: 1px solid {frame_border};
  border-radius: 14px;
}}

[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span {{
  color: {muted_color} !important;
}}

.sidebar-footer {{
  padding: 0.35rem 0.45rem 0.05rem 0.45rem;
  border-top: 1px solid {frame_border};
  font-size: 0.62rem;
  line-height: 1.2;
  opacity: 0.8;
}}

.sidebar-footer-wrap {{
  margin-top: auto;
  margin-left: 0.15rem;
  margin-right: 0.15rem;
  margin-bottom: 0.2rem;
  background: radial-gradient(65% 80% at 50% 0%, {card_glow} 0%, rgba(255,255,255,0) 100%), {card_bg};
  border: 1px solid {card_border};
  border-radius: 10px;
  box-shadow: {card_shadow};
  backdrop-filter: blur(8px);
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def plotly_template() -> str:
    return "plotly_dark" if st.session_state.get("dark_mode", False) else "plotly_white"


def normalize_col_name(name: Any) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def normalize_metric_name(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    # Normalize different dash encodings to plain ASCII dash
    text = text.replace("–", "-").replace("—", "-").replace("â€“", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).replace("\xa0", " ").replace("\u202f", " ").strip()
    if not text:
        return None
    if text.lower() in {"nan", "none", "nat"}:
        return None
    return text


def to_float(value: Any) -> float:
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if not math.isnan(float(value)) else np.nan

    text = clean_text(value)
    if text is None:
        return np.nan

    compact = text.replace(" ", "").replace("%", "").replace(",", ".")
    if compact in {"?", "-", "--"}:
        return np.nan
    try:
        return float(compact)
    except ValueError:
        return np.nan


def parse_mixed_date(value: Any) -> pd.Timestamp | pd.NaT:
    if isinstance(value, pd.Timestamp):
        return value.normalize()
    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()

    text = clean_text(value)
    if text is None:
        return pd.NaT

    compact = re.sub(r"\s+", " ", text).strip()
    numeric_compact = compact.replace(" ", "")

    if re.fullmatch(r"[+-]?\d+(\.\d+)?", numeric_compact):
        num = float(numeric_compact)
        if 20_000 <= num <= 70_000:
            ts = pd.Timestamp("1899-12-30") + pd.to_timedelta(num, unit="D")
            if 1990 <= ts.year <= 2035:
                return ts.normalize()
        return pd.NaT

    fix_year = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./-](\d{5})(?:\s+.*)?", compact)
    if fix_year:
        day, month, year = fix_year.groups()
        compact = f"{day}.{month}.{year[-4:]}"

    # Handle ISO-like date-time first, then flexible day-first fallbacks.
    ts = pd.to_datetime(compact, errors="coerce", format="%Y-%m-%d %H:%M:%S")
    if pd.isna(ts):
        ts = pd.to_datetime(compact, errors="coerce", format="%Y-%m-%d")
    if pd.isna(ts):
        ts = pd.to_datetime(compact, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        return pd.NaT
    if ts.year < 1990 or ts.year > 2035:
        return pd.NaT
    return pd.Timestamp(ts).normalize()


def normalize_country(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return "Unknown"
    norm = re.sub(r"\s+", " ", text).strip().lower()
    if norm in COUNTRY_MAP:
        return COUNTRY_MAP[norm]
    if norm in SUMMARY_LABELS:
        return "Unknown"
    if norm.startswith("#"):
        return "Unknown"
    return text


def map_country_for_geo(country: Any) -> str | None:
    text = clean_text(country)
    if text is None:
        return None
    text = GEO_COUNTRY_OVERRIDES.get(text, text)
    if text is None:
        return None
    return COUNTRY_ISO3_MAP.get(text)


def country_region_continent(country: Any) -> tuple[str, str]:
    text = clean_text(country)
    if text is None:
        return "Unknown", "Unknown"
    mapped = COUNTRY_REGION_CONTINENT.get(text)
    if mapped is not None:
        return mapped
    return COUNTRY_REGION_CONTINENT_LOWER.get(text.lower(), ("Unknown", "Unknown"))


def apply_region_continent_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "country" not in df.columns:
        return df

    out = df.copy()
    rc = out["country"].apply(country_region_continent)
    out["region"] = rc.apply(lambda t: t[0])
    out["continent"] = rc.apply(lambda t: t[1])
    return out


def render_sidebar_footer() -> None:
    st.sidebar.markdown(
        """
<div class="sidebar-footer-wrap">
  <div class="sidebar-footer">
  Unofficial dashboard - not affiliated with or endorsed by GE Vernova or General Electric. Data compiled manually from public reports/announcements; verify against original sources. Provided "as is" without warranty.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def normalize_service_scheme(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return "Unknown"
    norm = re.sub(r"\s+", " ", text).strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", norm)

    if norm in SVC_UNKNOWN_VALUES:
        return "Unknown"
    if "5000" in compact:
        return "AOM5000"
    if "4000" in compact:
        return "AOM4000"
    return "Unknown"


def normalize_customer(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return "Unknown"
    norm = text.strip().lower()
    if norm in CUSTOMER_UNKNOWN_VALUES:
        return "Unknown"
    return text


def normalize_platform(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return "Unknown"
    text = text.upper().replace(",", ".")
    text = re.sub(r"\s+", "", text)
    if text in {"?", "NAN", "NONE"}:
        return "Unknown"
    return text


def parse_platform_specs(platform: str) -> tuple[float, float]:
    if platform == "Unknown":
        return np.nan, np.nan

    p = platform.upper().replace(" ", "")
    full = re.search(r"V(\d{2,3})-([0-9]+(?:\.[0-9]+)?)(MW|KW)", p)
    if full:
        rotor = float(full.group(1))
        power = float(full.group(2))
        if full.group(3) == "KW":
            power = power / 1000.0
        return rotor, power

    mw_only = re.search(r"([0-9]+(?:\.[0-9]+)?)(MW|KW)", p)
    if mw_only:
        power = float(mw_only.group(1))
        if mw_only.group(2) == "KW":
            power = power / 1000.0
        return np.nan, power

    return np.nan, np.nan


def first_text(row: pd.Series, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in row.index:
            val = clean_text(row[col])
            if val is not None:
                return val
    return None


def first_float(row: pd.Series, candidates: list[str]) -> float:
    for col in candidates:
        if col in row.index:
            val = to_float(row[col])
            if not math.isnan(val):
                return val
    return np.nan


def quarter_end_date(year: int, quarter: int) -> pd.Timestamp:
    month = quarter * 3
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)


def parse_quarter_label(value: Any) -> int | None:
    text = clean_text(value)
    if text is None:
        return None
    token = re.sub(r"[^a-z0-9]", "", text.lower())
    match = re.fullmatch(r"q([1-4])", token)
    if match:
        return int(match.group(1))
    return None


def extract_unannounced_quarters(raw: pd.DataFrame, sheet_name: str, sheet_year: int) -> list[dict[str, Any]]:
    cols_norm = {normalize_col_name(c): c for c in raw.columns}
    size_col = cols_norm.get("size [mw]")
    if size_col is None and len(raw.columns) > 3:
        size_col = raw.columns[3]

    label_cols: list[str] = []
    for candidate in ["date", "country"]:
        col = cols_norm.get(candidate)
        if col is not None:
            label_cols.append(col)
    for idx in [0, 1, 2]:
        if idx < len(raw.columns):
            col = raw.columns[idx]
            if col not in label_cols:
                label_cols.append(col)

    rows: list[dict[str, Any]] = []
    for i in range(len(raw)):
        row_values = raw.iloc[i].tolist()
        texts = [clean_text(v) for v in row_values]
        normalized = [re.sub(r"\s+", " ", t).strip().lower() for t in texts if t is not None]
        if not any(t.startswith("unannounced") or t.startswith("unannounce") for t in normalized):
            continue

        for j in range(i + 1, min(i + 10, len(raw))):
            row_j = raw.iloc[j]
            quarter = None
            for col in label_cols:
                quarter = parse_quarter_label(row_j.get(col))
                if quarter is not None:
                    break
            if quarter is None:
                for val in row_j.tolist():
                    quarter = parse_quarter_label(val)
                    if quarter is not None:
                        break
            if quarter is None:
                continue

            mw = to_float(row_j.get(size_col)) if size_col is not None else np.nan
            if math.isnan(mw):
                for val in row_j.tolist():
                    num = to_float(val)
                    if not math.isnan(num):
                        mw = num
                        break
            if math.isnan(mw):
                continue

            rows.append(
                {
                    "sheet_name": sheet_name,
                    "sheet_year": sheet_year,
                    "quarter": int(quarter),
                    "unannounced_mw": float(mw),
                    "source_row": int(j),
                }
            )

        break

    return rows


def extract_order_block(df: pd.DataFrame) -> pd.DataFrame:
    cols_norm = {normalize_col_name(c): c for c in df.columns}
    country_check_col = cols_norm.get("country check") or cols_norm.get("check country")
    platform_check_col = cols_norm.get("platform check") or cols_norm.get("check platform")

    core_candidates = [
        "date",
        "country",
        "size [mw]",
        "service scheme",
        "service time [years]",
        "service time  [years]",
        "customer",
        "#",
        "no_wtg_1",
        "platform",
        "multi platform",
        "multi platform.1",
        "multi platform.2",
    ]
    core_cols = [cols_norm[c] for c in core_candidates if c in cols_norm]
    if not core_cols:
        return df.iloc[0:0].copy()

    if country_check_col or platform_check_col:
        keep = pd.Series(False, index=df.index)
        if country_check_col:
            keep |= df[country_check_col].notna()
        if platform_check_col:
            keep |= df[platform_check_col].notna()
        block = df.loc[keep].copy()
    else:
        nonempty = df[core_cols].notna().any(axis=1).to_numpy()
        if not nonempty.any():
            return df.iloc[0:0].copy()
        start = int(np.argmax(nonempty))
        run = 0
        cutoff = len(df)
        for idx in range(start, len(df)):
            if nonempty[idx]:
                run = 0
            else:
                run += 1
                if run >= 5:
                    cutoff = idx - run + 1
                    break
        block = df.iloc[start:cutoff].copy()

    block = block[block[core_cols].notna().any(axis=1)]
    block.reset_index(drop=True, inplace=True)
    return block


def parse_oi_sheet_year(sheet_name: Any) -> int | None:
    text = clean_text(sheet_name)
    if text is None:
        return None
    m = re.fullmatch(r"(?:OI[\s_-]*)?(\d{4})", text, flags=re.IGNORECASE)
    if m is None:
        return None
    year = int(m.group(1))
    if 1900 <= year <= 2100:
        return year
    return None


def normalize_sheet_key(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def resolve_sheet_name(sheet_names: list[str], target_name: str) -> str | None:
    target_clean = clean_text(target_name)
    if target_clean is None:
        return None

    for name in sheet_names:
        if clean_text(name) == target_clean:
            return str(name)

    target_casefold = target_clean.casefold()
    for name in sheet_names:
        name_clean = clean_text(name)
        if name_clean is not None and name_clean.casefold() == target_casefold:
            return str(name)

    target_key = normalize_sheet_key(target_clean)
    for name in sheet_names:
        if normalize_sheet_key(name) == target_key:
            return str(name)

    return None


def parse_economy_sheet(workbook: Path) -> pd.DataFrame:
    try:
        sheet_names = pd.ExcelFile(workbook).sheet_names
    except Exception as exc:
        raise ValueError(f"Could not open workbook '{workbook.name}': {exc}") from exc

    economy_sheet = (
        resolve_sheet_name(sheet_names, "economy")
        or resolve_sheet_name(sheet_names, "GE Economy")
        or resolve_sheet_name(sheet_names, "GE Wind Economy")
    )
    if economy_sheet is None:
        available = ", ".join(sheet_names[:12]) if sheet_names else "(none)"
        raise ValueError(
            f"Workbook '{workbook.name}' does not contain an 'economy' sheet. "
            f"Available sheets: {available}"
        )

    economy = pd.read_excel(workbook, sheet_name=economy_sheet, dtype=object)
    if economy.empty:
        return pd.DataFrame(columns=["metric", "year", "value"])

    year_map: dict[Any, int] = {}

    # Detect year columns from top rows and/or header labels.
    probe_rows = min(5, len(economy))
    for col in economy.columns:
        year = np.nan
        for idx in range(probe_rows):
            year = to_float(economy.iloc[idx].get(col))
            if not math.isnan(year):
                break
        if math.isnan(year):
            if isinstance(col, (int, np.integer)):
                year = int(col)
            elif isinstance(col, (float, np.floating)) and float(col).is_integer():
                year = int(col)
            else:
                continue
        if float(year).is_integer() and 1900 <= int(year) <= 2100:
            year_map[col] = int(year)

    if not year_map:
        return pd.DataFrame(columns=["metric", "year", "value"])

    # Find likely metric-name column.
    metric_col = economy.columns[0]
    metric_score = -1
    for col in economy.columns:
        score = 0
        for value in economy[col].tolist():
            text = clean_text(value)
            if text is None:
                continue
            if re.fullmatch(r"[\d\.,\-\+%]+", text):
                continue
            t = text.lower()
            score += 1
            if any(k in t for k in ("order", "sales", "revenue", "gross", "staff", "employ", "capacity", "intake", "backlog")):
                score += 2
        if score > metric_score:
            metric_score = score
            metric_col = col

    rows: list[dict[str, Any]] = []
    for _, row in economy.iterrows():
        metric = normalize_metric_name(row.get(metric_col))
        if metric is None:
            continue

        metric_norm = metric.strip()
        if re.fullmatch(r"\d{2}q[1-4]", metric_norm, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\d{4}", metric_norm):
            continue
        if metric_norm.lower() in {"in %", "q1", "q2", "q3", "q4"}:
            continue

        values = {}
        for source_col, year in year_map.items():
            value = to_float(row.get(source_col))
            if not math.isnan(value):
                values[year] = value

        if len(values) < 1:
            continue

        rec = {"metric": metric_norm}
        rec.update(values)
        rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=["metric", "year", "value"])

    econ = pd.DataFrame(rows)
    value_cols = [c for c in econ.columns if isinstance(c, int)]
    long = econ.melt(id_vars="metric", value_vars=value_cols, var_name="year", value_name="value")
    long = long.dropna(subset=["value"]).copy()
    long["year"] = long["year"].astype(int)
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    long = long.groupby(["metric", "year"], as_index=False)["value"].mean()
    return long


def parse_oi_sheets(workbook: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xl = pd.ExcelFile(workbook)
    oi_sheets = [
        (s, parse_oi_sheet_year(s))
        for s in xl.sheet_names
    ]
    oi_sheets = [(s, y) for s, y in oi_sheets if y is not None]
    oi_sheets = sorted(oi_sheets, key=lambda pair: pair[1])

    order_rows: list[dict[str, Any]] = []
    platform_rows: list[dict[str, Any]] = []
    unannounced_rows: list[dict[str, Any]] = []

    for sheet, year in oi_sheets:
        raw = pd.read_excel(workbook, sheet_name=sheet, dtype=object)
        raw.columns = [str(c).strip() for c in raw.columns]
        unannounced_rows.extend(extract_unannounced_quarters(raw, sheet, year))
        block = extract_order_block(raw)

        for row_idx, row in block.iterrows():
            order_id = f"{year}_{row_idx:04d}"
            order_date_raw = row.get("date") if "date" in row.index else row.get("Date")
            order_date = parse_mixed_date(order_date_raw)
            if pd.isna(order_date):
                order_date = parse_mixed_date(first_text(row, ["date", "Date"]))
            size_mw = first_float(row, ["size [MW]", "size[MW]", "size mw", "size"])
            platform_raw = first_text(row, ["platform", "Platform", "turbine platform"])
            qty = first_float(row, ["#", "No_WTG_1", "no_wtg", "turbines"])
            rotor_raw = first_float(row, ["Rotor", "rotor"])
            mw_raw = first_float(row, ["MW", "mw"])
            service_time_years = first_float(row, ["Service time  [years]", "Service time [years]", "service time [years]"])
            service_scheme = normalize_service_scheme(first_text(row, ["Service scheme", "service scheme", "special mode"]))

            delivery_q = first_float(row, ["delivery q", "delivery_q", "operation q", "oparetion q"])
            delivery_year = first_float(row, ["delivery y", "delivery_year", "operation y", "oparetion y"])
            delivery_days = first_float(row, ["Delivery days", "delivery days", "delivery_days"])
            if math.isnan(delivery_q) or delivery_q not in {1.0, 2.0, 3.0, 4.0}:
                delivery_q = np.nan
            if math.isnan(delivery_year):
                delivery_year = np.nan
            elif delivery_year < 1990 or delivery_year > 2035:
                delivery_year = np.nan

            if math.isnan(delivery_days) and pd.notna(order_date) and not math.isnan(delivery_q) and not math.isnan(delivery_year):
                q_end = quarter_end_date(int(delivery_year), int(delivery_q))
                delivery_days = float((q_end - order_date).days)

            if math.isnan(size_mw) and math.isnan(qty) and platform_raw is None:
                continue

            platform = normalize_platform(platform_raw)
            rotor_guess, mw_guess = parse_platform_specs(platform)
            rotor = rotor_raw if not math.isnan(rotor_raw) else rotor_guess
            mw_rating = mw_raw if not math.isnan(mw_raw) else mw_guess

            if math.isnan(qty) and not math.isnan(size_mw) and not math.isnan(mw_rating) and mw_rating > 0:
                qty = size_mw / mw_rating

            slot_mw = size_mw
            if math.isnan(slot_mw) and (not math.isnan(qty)) and (not math.isnan(mw_rating)):
                slot_mw = qty * mw_rating
            if math.isnan(size_mw) and not math.isnan(slot_mw):
                size_mw = slot_mw

            country = normalize_country(row.get("country"))
            region, continent = country_region_continent(country)
            customer = normalize_customer(row.get("customer"))

            order_data = {
                "order_id": order_id,
                "sheet_name": sheet,
                "sheet_year": year,
                "order_date": order_date,
                "order_year": int(order_date.year) if pd.notna(order_date) else year,
                "order_quarter": int(order_date.quarter) if pd.notna(order_date) else np.nan,
                "country": country,
                "region": region,
                "continent": continent,
                "service_scheme": service_scheme,
                "service_time_years": service_time_years,
                "customer": customer,
                "size_mw": size_mw,
                "delivery_q": int(delivery_q) if not math.isnan(delivery_q) else np.nan,
                "delivery_year": int(delivery_year) if not math.isnan(delivery_year) else np.nan,
                "delivery_days": delivery_days,
            }

            order_data["primary_platform"] = platform
            platform_rows.append(
                {
                    "order_id": order_id,
                    "sheet_year": year,
                    "order_year": int(order_data["order_year"]),
                    "order_date": order_data["order_date"],
                    "country": country,
                    "region": region,
                    "continent": continent,
                    "service_scheme": service_scheme,
                    "service_time_years": service_time_years,
                    "customer": customer,
                    "delivery_q": order_data["delivery_q"],
                    "delivery_year": order_data["delivery_year"],
                    "delivery_days": order_data["delivery_days"],
                    "order_size_mw": size_mw,
                    "slot": 1,
                    "platform": platform,
                    "turbines_qty": qty,
                    "rotor_m": rotor,
                    "mw_rating": mw_rating,
                    "slot_mw": slot_mw,
                }
            )

            order_rows.append(order_data)

    orders = pd.DataFrame(order_rows)
    platforms = pd.DataFrame(platform_rows)
    orders = apply_region_continent_columns(orders)
    platforms = apply_region_continent_columns(platforms)
    unannounced = pd.DataFrame(unannounced_rows)
    if orders.empty:
        if unannounced.empty:
            unannounced = pd.DataFrame(columns=["sheet_name", "sheet_year", "quarter", "unannounced_mw", "source_row"])
        return orders, platforms, unannounced

    for col in ["size_mw", "service_time_years", "delivery_days"]:
        orders[col] = pd.to_numeric(orders[col], errors="coerce")

    for col in ["turbines_qty", "rotor_m", "mw_rating", "slot_mw", "order_size_mw"]:
        if col in platforms.columns:
            platforms[col] = pd.to_numeric(platforms[col], errors="coerce")

    valid_ids = set(orders.loc[orders["size_mw"] > 0, "order_id"])
    valid_ids.update(platforms.loc[platforms["slot_mw"] > 0, "order_id"])
    orders = orders[orders["order_id"].isin(valid_ids)].copy()
    platforms = platforms[platforms["order_id"].isin(valid_ids)].copy()

    order_platform_mw = platforms.groupby("order_id")["slot_mw"].sum(min_count=1)
    order_turbines = platforms.groupby("order_id")["turbines_qty"].sum(min_count=1)
    orders["platform_mw_sum"] = orders["order_id"].map(order_platform_mw)
    orders["turbines_qty_sum"] = orders["order_id"].map(order_turbines)
    orders["mw_gap"] = orders["size_mw"] - orders["platform_mw_sum"]

    orders["delivery_time_years"] = orders["delivery_days"] / 365.25
    platforms["delivery_time_years"] = platforms["delivery_days"] / 365.25

    if unannounced.empty:
        unannounced = pd.DataFrame(columns=["sheet_name", "sheet_year", "quarter", "unannounced_mw", "source_row"])
    else:
        unannounced["sheet_year"] = pd.to_numeric(unannounced["sheet_year"], errors="coerce").astype("Int64")
        unannounced["quarter"] = pd.to_numeric(unannounced["quarter"], errors="coerce").astype("Int64")
        unannounced["unannounced_mw"] = pd.to_numeric(unannounced["unannounced_mw"], errors="coerce")
        unannounced = unannounced.dropna(subset=["sheet_year", "quarter", "unannounced_mw"]).copy()
        unannounced["sheet_year"] = unannounced["sheet_year"].astype(int)
        unannounced["quarter"] = unannounced["quarter"].astype(int)
        unannounced = (
            unannounced.groupby(["sheet_name", "sheet_year", "quarter"], as_index=False)["unannounced_mw"].sum()
        )

    return orders, platforms, unannounced


def find_default_workbook() -> Path | None:
    app_dir = Path(__file__).resolve().parent
    parent_dir = app_dir.parent
    search_roots = [Path("."), app_dir, parent_dir]
    preferred_names = [
        "GE_data.xls",
        "GE_data.xlsx",
        "ge_data.xls",
        "ge_data.xlsx",
    ]

    for root in search_roots:
        for name in preferred_names:
            workbook = root / name
            if not workbook.exists() or workbook.name.startswith("~$"):
                continue
            try:
                sheet_names = pd.ExcelFile(workbook).sheet_names
            except Exception:
                continue
            if resolve_sheet_name(sheet_names, "economy") is not None or any(
                parse_oi_sheet_year(s) is not None for s in sheet_names
            ):
                return workbook

    patterns = ("*.xlsx", "*.xlsm", "*.xls")
    files: list[Path] = []
    seen: set[str] = set()
    for root in search_roots:
        for pattern in patterns:
            for file_path in sorted(root.glob(pattern)):
                if file_path.name.startswith("~$"):
                    continue
                key = str(file_path.resolve()).lower()
                if key in seen:
                    continue
                seen.add(key)
                files.append(file_path)
    if not files:
        return None

    def file_priority(path: Path) -> tuple[int, float, str]:
        name = path.name.lower()
        if "ge" in name:
            rank = 0
        elif any(token in name for token in ("vestas", "nordex", "suzlon", "sgre", "siemens")):
            rank = 2
        else:
            rank = 1
        try:
            mtime = path.stat().st_mtime
        except Exception:
            mtime = 0.0
        return (rank, -mtime, name)

    for workbook in sorted(files, key=file_priority):
        try:
            sheet_names = pd.ExcelFile(workbook).sheet_names
        except Exception:
            continue
        if resolve_sheet_name(sheet_names, "economy") is not None or any(
            parse_oi_sheet_year(s) is not None for s in sheet_names
        ):
            return workbook

    return None


def empty_stock_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])


def parse_stock_monthly_json(stock_path: Path) -> pd.DataFrame:
    if not stock_path.exists():
        return empty_stock_frame()

    try:
        raw = pd.read_json(stock_path)
    except Exception:
        return empty_stock_frame()

    if raw.empty:
        return empty_stock_frame()

    rename_map: dict[str, str] = {}
    for col in raw.columns:
        key = str(col).strip().lower()
        if "date" in key:
            rename_map[col] = "date"
        elif "adj" in key and "close" in key:
            rename_map[col] = "adj_close"
        elif "open" in key:
            rename_map[col] = "open"
        elif "high" in key:
            rename_map[col] = "high"
        elif "low" in key:
            rename_map[col] = "low"
        elif "close" in key:
            rename_map[col] = "close"
        elif "volume" in key:
            rename_map[col] = "volume"

    stock = raw.rename(columns=rename_map).copy()
    required_cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    for col in required_cols:
        if col not in stock.columns:
            stock[col] = np.nan
    stock = stock[required_cols].copy()

    stock["date"] = pd.to_datetime(stock["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        stock[col] = pd.to_numeric(stock[col], errors="coerce")

    stock = stock.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    stock = stock.sort_values("date")
    stock["date"] = stock["date"].dt.normalize()
    stock["volume"] = stock["volume"].fillna(0.0)
    return stock


def parse_market_monthly_json(market_path: Path) -> pd.DataFrame:
    if not market_path.exists():
        return pd.DataFrame(columns=["date", "steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"])

    try:
        raw = pd.read_json(market_path)
    except Exception:
        return pd.DataFrame(columns=["date", "steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"])

    if raw.empty:
        return pd.DataFrame(columns=["date", "steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"])

    market = raw.copy()
    if "date" not in market.columns:
        for c in market.columns:
            if "date" in str(c).lower():
                market = market.rename(columns={c: "date"})
                break
    if "date" not in market.columns:
        return pd.DataFrame(columns=["date", "steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"])

    for col in ["steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"]:
        if col not in market.columns:
            market[col] = np.nan
        market[col] = pd.to_numeric(market[col], errors="coerce")

    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market = market.dropna(subset=["date"]).sort_values("date")
    return market[["date", "steel_hrc_usd_per_short_ton", "copper_hg_usd_per_lb"]].copy()


def merge_press_orders(orders: pd.DataFrame, platforms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append the orders announced after the workbook's last row (data/ge_orders_supplement.json).

    Rows already in the workbook (same date and size) are skipped. ``booked_year`` / ``booked_q`` hold the booking quarter
    when the press release states it (announcements can trail the booking by months).
    """
    if not SUPPLEMENT_FILE.exists() or orders.empty:
        return orders, platforms
    try:
        rows = json.loads(SUPPLEMENT_FILE.read_text(encoding="utf-8")).get("orders", [])
    except (OSError, json.JSONDecodeError):
        return orders, platforms
    existing = {(pd.Timestamp(d).normalize(), round(float(m), 1)) for d, m in zip(orders["order_date"], orders["size_mw"]) if pd.notna(d)}
    counters: dict[int, int] = {}
    for oid in orders["order_id"]:
        year, n = str(oid).split("_")
        counters[int(year)] = max(counters.get(int(year), -1), int(n))
    new_orders, new_platforms = [], []
    for r in sorted(rows, key=lambda x: x["date"]):
        date = pd.Timestamp(r["date"]).normalize()
        size = float(r["size_mw"])
        if (date, round(size, 1)) in existing:
            continue
        year = int(date.year)
        counters[year] = counters.get(year, -1) + 1
        order_id = f"{year}_{counters[year]:04d}"
        country = normalize_country(r.get("country"))
        region, continent = country_region_continent(country)
        platform = normalize_platform(r.get("platform"))
        qty = float(r["turbines"]) if r.get("turbines") else np.nan
        rating = float(r["mw_rating"]) if r.get("mw_rating") else np.nan
        rotor = float(r["rotor_m"]) if r.get("rotor_m") else np.nan
        svc = float(r["service_years"]) if r.get("service_years") else np.nan
        booked = str(r.get("booked") or "")
        common = {
            "country": country, "region": region, "continent": continent, "service_scheme": "Unknown", "service_time_years": svc,
            "customer": normalize_customer(r.get("customer")), "delivery_q": np.nan, "delivery_year": np.nan, "delivery_days": np.nan,
        }
        new_orders.append(
            {
                "order_id": order_id, "sheet_name": "press releases", "sheet_year": year, "order_date": date, "order_year": year,
                "order_quarter": int(date.quarter), **common, "size_mw": size, "primary_platform": platform,
                "platform_mw_sum": size if platform != "Unknown" else np.nan, "turbines_qty_sum": qty,
                "mw_gap": 0.0 if platform != "Unknown" else np.nan, "delivery_time_years": np.nan,
                "booked_year": float(booked[:4]) if booked else np.nan, "booked_q": float(booked[-1]) if booked else np.nan,
            }
        )
        new_platforms.append(
            {
                "order_id": order_id, "sheet_year": year, "order_year": year, "order_date": date, **common, "order_size_mw": size,
                "slot": 1, "platform": platform, "turbines_qty": qty, "rotor_m": rotor, "mw_rating": rating,
                "slot_mw": size, "delivery_time_years": np.nan,
            }
        )
    if not new_orders:
        return orders, platforms
    return pd.concat([orders, pd.DataFrame(new_orders)], ignore_index=True), pd.concat([platforms, pd.DataFrame(new_platforms)], ignore_index=True)


@st.cache_data(show_spinner=False)
def load_data(cache_key: str, workbook_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache_file = PARSED_DATA_FILE
    if cache_file.exists():
        try:
            payload = pd.read_pickle(cache_file)
            if isinstance(payload, dict) and payload.get("cache_key") == cache_key:
                return (
                    payload.get("economy", pd.DataFrame(columns=["metric", "year", "value"])),
                    payload.get("orders", pd.DataFrame()),
                    payload.get("platforms", pd.DataFrame()),
                    payload.get("unannounced", pd.DataFrame()),
                )
        except Exception:
            pass

    workbook = Path(workbook_path)
    economy = parse_economy_sheet(workbook)
    orders, platforms, unannounced = parse_oi_sheets(workbook)
    orders, platforms = merge_press_orders(orders, platforms)

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_key": cache_key,
            "workbook_path": str(workbook.resolve()),
            "generated_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "economy": economy,
            "orders": orders,
            "platforms": platforms,
            "unannounced": unannounced,
        }
        pd.to_pickle(payload, cache_file)
    except Exception:
        pass

    return economy, orders, platforms, unannounced


@st.cache_data(show_spinner=False)
def load_stock_data(cache_key: str, stock_path: str) -> pd.DataFrame:
    _ = cache_key
    return parse_stock_monthly_json(Path(stock_path))


@st.cache_data(show_spinner=False)
def load_market_data(cache_key: str, market_path: str) -> pd.DataFrame:
    _ = cache_key
    return parse_market_monthly_json(Path(market_path))


def load_data_from_cache_only() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if not PARSED_DATA_FILE.exists():
        return None
    try:
        payload = pd.read_pickle(PARSED_DATA_FILE)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return (
        payload.get("economy", pd.DataFrame(columns=["metric", "year", "value"])),
        payload.get("orders", pd.DataFrame()),
        payload.get("platforms", pd.DataFrame()),
        payload.get("unannounced", pd.DataFrame()),
    )


def parse_news_datetime(value: Any) -> pd.Timestamp | pd.NaT:
    text = clean_text(value)
    if text is None:
        return pd.NaT
    try:
        dt = parsedate_to_datetime(text)
        if dt is not None:
            return pd.Timestamp(dt).tz_convert("UTC").tz_localize(None) if dt.tzinfo else pd.Timestamp(dt)
    except Exception:
        pass
    ts = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert("UTC").tz_localize(None)


def normalize_news_title(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_news_summary(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def short_summary(value: Any, max_chars: int = 320) -> str | None:
    text = normalize_news_summary(value)
    if text is None:
        return None
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    if not clipped:
        clipped = text[:max_chars].strip()
    return f"{clipped}..."


def is_ge_wind_relevant(
    title: str | None,
    summary: str | None,
    link: str | None,
    source_name: str | None = None,
) -> bool:
    _source_txt = (source_name or "").lower()
    full = " ".join([title or "", summary or "", link or ""]).lower()

    if any(bad in full for bad in GE_WIND_NEWS_BLACKLIST):
        return False

    title_txt = title or ""
    if not GE_TITLE_RE.search(title_txt):
        return False

    # Filter broad roundup headlines where GE is one of many names.
    title_low = title_txt.lower()
    if title_low.count("|") >= 2 and not title_low.startswith("ge"):
        return False

    if not any(k in full for k in GE_KEYWORDS):
        return False

    return any(k in full for k in WIND_TOPIC_KEYWORDS)


def ge_news_focus(news_type: Any) -> str:
    text = (clean_text(news_type) or "").lower()
    if "announcement" in text or "investor" in text:
        return "financial"
    if "press" in text:
        return "company"
    return "company"


def parse_ge_company_news_archive() -> list[dict[str, Any]]:
    rows = parse_rss_feed(
        "https://news.google.com/rss/search?q=site%3Agevernova.com%20%22wind%22%20%22press%20release%22&hl=en-US&gl=US&ceid=US:en",
        "GE Vernova (Official via Google News)",
        "company",
    )
    for row in rows:
        row["focus"] = ge_news_focus(row.get("focus"))
    return rows


def parse_rss_feed(feed_url: str, source_name: str, focus: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        resp = requests.get(feed_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception:
        return rows

    items = root.findall(".//item")
    atom_entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    for node in items:
        title = normalize_news_title(node.findtext("title"))
        link = clean_text(node.findtext("link"))
        summary = normalize_news_summary(node.findtext("description"))
        pub_raw = (
            node.findtext("pubDate")
            or node.findtext("{http://purl.org/dc/elements/1.1/}date")
            or node.findtext("date")
        )
        published = parse_news_datetime(pub_raw)
        if title is None or link is None:
            continue
        source = source_name
        if source_name.startswith("Google News") and " - " in title:
            t_main, t_src = title.rsplit(" - ", 1)
            t_main = normalize_news_title(t_main)
            t_src = clean_text(t_src)
            if t_main is not None:
                title = t_main
            if t_src is not None:
                source = t_src
        if not is_ge_wind_relevant(title, summary, link, source_name=source):
            continue
        rows.append(
            {
                "source": source,
                "focus": focus,
                "title": title,
                "link": link,
                "summary": summary,
                "published": published,
            }
        )

    for node in atom_entries:
        title = normalize_news_title(node.findtext("{http://www.w3.org/2005/Atom}title"))
        link = None
        for link_node in node.findall("{http://www.w3.org/2005/Atom}link"):
            href = clean_text(link_node.attrib.get("href"))
            rel = clean_text(link_node.attrib.get("rel"))
            if href is None:
                continue
            if rel in {None, "alternate"}:
                link = href
                break
            if link is None:
                link = href

        pub_raw = (
            node.findtext("{http://www.w3.org/2005/Atom}updated")
            or node.findtext("{http://www.w3.org/2005/Atom}published")
        )
        summary = normalize_news_summary(
            node.findtext("{http://www.w3.org/2005/Atom}summary")
            or node.findtext("{http://www.w3.org/2005/Atom}content")
        )
        published = parse_news_datetime(pub_raw)
        if title is None or link is None:
            continue
        source = source_name
        if source_name.startswith("Google News") and " - " in title:
            t_main, t_src = title.rsplit(" - ", 1)
            t_main = normalize_news_title(t_main)
            t_src = clean_text(t_src)
            if t_main is not None:
                title = t_main
            if t_src is not None:
                source = t_src
        if not is_ge_wind_relevant(title, summary, link, source_name=source):
            continue
        rows.append(
            {
                "source": source,
                "focus": focus,
                "title": title,
                "link": link,
                "summary": summary,
                "published": published,
            }
        )
    return rows


@st.cache_data(show_spinner=False, ttl=1800)
def load_latest_news() -> pd.DataFrame:
    rows: list[dict[str, Any]] = parse_ge_company_news_archive()
    for feed in NEWS_FEEDS:
        rows.extend(parse_rss_feed(feed["url"], feed["source"], feed["focus"]))

    if not rows:
        return pd.DataFrame(columns=["source", "focus", "title", "link", "summary", "published"])

    news = pd.DataFrame(rows)
    news = news[
        news.apply(
            lambda r: is_ge_wind_relevant(
                clean_text(r.get("title")),
                clean_text(r.get("summary")),
                clean_text(r.get("link")),
                clean_text(r.get("source")),
            ),
            axis=1,
        )
    ].copy()
    if news.empty:
        return pd.DataFrame(columns=["source", "focus", "title", "link", "summary", "published"])
    news["published"] = pd.to_datetime(news["published"], errors="coerce")
    news["title_key"] = news["title"].str.strip().str.lower()
    news["link_key"] = news["link"].str.strip().str.lower()
    news = news.sort_values(["published", "title"], ascending=[False, True])
    news = news.drop_duplicates(subset=["link_key"], keep="first")
    news = news.drop_duplicates(subset=["title_key"], keep="first")
    news = news.drop(columns=["title_key", "link_key"])

    max_per_source = 8
    selected_idx: list[int] = []
    source_counts: dict[str, int] = {}
    for idx, row in news.iterrows():
        src = str(row.get("source", "Unknown"))
        count = source_counts.get(src, 0)
        if count >= max_per_source:
            continue
        selected_idx.append(idx)
        source_counts[src] = count + 1
        if len(selected_idx) >= 20:
            break

    if len(selected_idx) < 20:
        for idx in news.index:
            if idx in selected_idx:
                continue
            selected_idx.append(idx)
            if len(selected_idx) >= 20:
                break

    return news.loc[selected_idx].copy()


@st.cache_data(show_spinner=False, ttl=43_200)
def load_latest_patents() -> pd.DataFrame:
    columns = [
        "title",
        "link",
        "publication_number",
        "publication_date",
        "filing_date",
        "priority_date",
        "inventor",
        "assignee",
        "summary",
    ]
    try:
        resp = requests.get(PATENTSCOPE_SEARCH_URL, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        page = resp.text
    except Exception:
        return pd.DataFrame(columns=columns)

    def html_text(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = re.sub(r"<[^>]+>", " ", value)
        cleaned = html.unescape(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or None

    pub_nos = re.findall(r"ps-patent-result--title--patent-number\">(.*?)</span>", page, flags=re.IGNORECASE | re.DOTALL)
    titles = re.findall(r"ps-patent-result--title--title[^>]*>(.*?)</span>\s*</span>", page, flags=re.IGNORECASE | re.DOTALL)
    pub_dates = re.findall(r"resultListTableColumnPubDate\"[^>]*>(.*?)</span>", page, flags=re.IGNORECASE | re.DOTALL)
    links = re.findall(
        r"<a href=\"([^\"]*detail\.jsf[^\"]+)\"[^>]*>\s*<span class=\"notranslate ps-patent-result--title--patent-number\">",
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )
    applicants = re.findall(r"ps-patent-result--applicant[^>]*>(.*?)\n", page, flags=re.IGNORECASE | re.DOTALL)
    inventors = re.findall(r"ps-patent-result--inventor[^>]*>(.*?)</span>", page, flags=re.IGNORECASE | re.DOTALL)
    abstracts = re.findall(r"ps-patent-result--abstract[^>]*>(.*?)</div>", page, flags=re.IGNORECASE | re.DOTALL)

    n = min(len(pub_nos), len(titles), len(pub_dates), len(links), len(applicants), len(inventors), len(abstracts))
    rows: list[dict[str, Any]] = []
    for idx in range(n):
        publication_number = html_text(pub_nos[idx])
        title = normalize_news_title(html_text(titles[idx]))
        applicant = html_text(applicants[idx])
        inventor = html_text(inventors[idx])
        summary = normalize_news_summary(html_text(abstracts[idx]))
        pub_date_raw = html_text(pub_dates[idx])
        publication_date = pd.to_datetime(pub_date_raw, format="%d.%m.%Y", errors="coerce")
        if pd.isna(publication_date):
            publication_date = pd.to_datetime(pub_date_raw, errors="coerce")

        if title is None or publication_number is None:
            continue
        applicant_l = (applicant or "").lower()
        if applicant_l and not any(token in applicant_l for token in ("general electric", "ge renewable", "ge vernova", "lm wind")):
            continue
        wind_text = " ".join([title or "", summary or ""]).lower()
        if not any(token in wind_text for token in WIND_TOPIC_KEYWORDS):
            continue

        href = html.unescape(links[idx])
        href = re.sub(r";jsessionid=[^?]+", "", href)
        link = href if href.startswith("http") else f"https://patentscope.wipo.int/search/en/{href.lstrip('/')}"

        rows.append(
            {
                "title": title,
                "link": link,
                "publication_number": publication_number,
                "publication_date": publication_date,
                "filing_date": pd.NaT,
                "priority_date": pd.NaT,
                "inventor": inventor,
                "assignee": applicant,
                "summary": summary,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    patents = pd.DataFrame(rows)
    patents = patents.drop_duplicates(subset=["publication_number", "title"], keep="first")
    patents = patents.sort_values(["publication_date", "title"], ascending=[False, True]).head(10)
    return patents


def latest_data_handled_text(
    economy: pd.DataFrame,
    orders: pd.DataFrame,
    unannounced: pd.DataFrame,
    stock_monthly: pd.DataFrame,
    market_monthly: pd.DataFrame,
    reported_label: str | None = None,
) -> str:
    economy_max_year = int(economy["year"].max()) if economy is not None and not economy.empty else None
    orders_max_date = pd.to_datetime(orders["order_date"], errors="coerce").max() if orders is not None and not orders.empty else pd.NaT
    orders_max_year = int(orders["sheet_year"].max()) if orders is not None and not orders.empty else None
    stock_max = pd.to_datetime(stock_monthly["date"], errors="coerce").max() if stock_monthly is not None and not stock_monthly.empty else pd.NaT
    market_max = pd.to_datetime(market_monthly["date"], errors="coerce").max() if market_monthly is not None and not market_monthly.empty else pd.NaT
    unann_max_year = int(unannounced["sheet_year"].max()) if unannounced is not None and not unannounced.empty else None

    orders_txt = "-"
    if pd.notna(orders_max_date):
        orders_txt = orders_max_date.strftime("%Y-%m-%d")
    elif orders_max_year is not None:
        orders_txt = str(orders_max_year)

    stock_txt = stock_max.strftime("%Y-%m-%d") if pd.notna(stock_max) else "-"
    market_txt = market_max.strftime("%Y-%m-%d") if pd.notna(market_max) else "-"
    econ_txt = str(economy_max_year) if economy_max_year is not None else "-"
    unann_txt = str(unann_max_year) if unann_max_year is not None else "-"

    return (
        f"Latest handled data -> Economy: {econ_txt}; Orders: {orders_txt}; "
        f"Quarterly Wind results: {reported_label or unann_txt}; Stock (GE Vernova): {stock_txt}; Steel/Copper: {market_txt}. "
        "Not a live feed; updates occur when source files/feeds are refreshed."
    )


def mw_fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:,.0f} MW"


def int_fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:,.0f}"


def days_fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:,.0f} days"


def top_n_by_mw(df: pd.DataFrame, col: str, value_col: str, n: int) -> list[str]:
    if df.empty:
        return []
    grouped = df.groupby(col, as_index=False)[value_col].sum()
    grouped = grouped.sort_values(value_col, ascending=False).head(n)
    return grouped[col].tolist()


def render_header_metrics(orders: pd.DataFrame, platforms: pd.DataFrame, unannounced: pd.DataFrame) -> None:
    total_orders = orders["order_id"].nunique()
    announced_mw = orders["size_mw"].sum()
    unannounced_mw = 0.0
    if unannounced is not None and not unannounced.empty and "unannounced_mw" in unannounced.columns:
        unannounced_mw = float(pd.to_numeric(unannounced["unannounced_mw"], errors="coerce").fillna(0).sum())
    total_mw = announced_mw + unannounced_mw
    total_platform_mw = platforms["slot_mw"].sum()
    avg_order_mw = orders["size_mw"].mean()
    countries = orders["country"].nunique()
    turbine_base = orders.dropna(subset=["turbines_qty_sum"]).copy()
    avg_turbines_per_order = turbine_base["turbines_qty_sum"].mean() if not turbine_base.empty else np.nan
    turbine_base_orders = turbine_base["order_id"].nunique()
    turbine_base_mw = turbine_base["size_mw"].sum()

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Orders", int_fmt(total_orders))
    m2.metric(
        "Announced MW",
        mw_fmt(total_mw),
        delta=f"largest order {orders['size_mw'].max():,.0f} MW",
        delta_color="off",
    )
    m3.metric("Platform-Mapped MW", mw_fmt(total_platform_mw))
    m4.metric(
        "Avg Order Size",
        mw_fmt(avg_order_mw),
        delta=f"based on {announced_mw / 1000.0:,.1f} GW announced",
        delta_color="off",
    )
    m5.metric(
        "Avg Turbines/Order",
        "-" if pd.isna(avg_turbines_per_order) else f"{avg_turbines_per_order:,.1f}",
        delta=f"based on {turbine_base_mw / 1000.0:,.1f} GW / {turbine_base_orders:,.0f} orders",
        delta_color="off",
    )
    m6.metric("Countries", int_fmt(countries))


def render_information_page() -> None:
    st.subheader("Information")
    st.markdown("**Data source**")
    st.write(
        "Wind-segment orders, revenue and segment EBITDA by quarter come from GE Vernova's earnings releases (`data/ge_quarterly_results.json`, USD). "
        "Orders announced after the workbook's last row come from GE Vernova press releases (`data/ge_orders_supplement.json`). "
        "The share-price series is GE Vernova (GEV), listed April 2024."
    )
    st.write("This dashboard is build on public available data.")

    st.markdown("**How to use and navigate**")
    st.write(
        "Use the sidebar filters to narrow years, countries, platforms, and minimum order size."
    )
    st.write(
        "Navigate tabs from high-level trends (Overall/Yearly/Quarterly) to deep dives (Across Years, Platform, and Country analytics)."
    )

    st.markdown("**Disclaimer**")
    st.write(
        "This dashboard is provided for informational purposes only. Data is compiled from public sources and may contain gaps, delays, or inaccuracies."
    )
    st.write(
        "Always verify critical figures with official company reporting before using the information for financial, legal, strategic, or investment decisions."
    )
    st.write(
        'The data and dashboard are provided "as is" without warranties of any kind. I make no guarantees of accuracy, completeness, or fitness for a particular purpose.'
    )
    st.write(
        "Use at your own risk. I am not responsible or liable for any losses or damages arising from use of the data or dashboard."
    )

    st.markdown("**Sources and attribution**")
    st.write(
        "All figures are compiled manually from GE Vernova / GE public investor communications (press releases/announcements) and annual reports."
    )
    st.write("Each datapoint should be verified against the original source documents.")

    st.markdown("**No affiliation / no endorsement**")
    st.write(
        "This is an unofficial, independent dashboard and is not affiliated with, endorsed by, or sponsored by GE Vernova or General Electric."
    )

    st.markdown("**IP and rights**")
    st.write(
        "The dataset in this repository is my own compilation (selection/structure) based on publicly available sources."
    )
    st.write("Underlying source documents remain the property of their respective owners.")

    st.markdown("**About me**")
    st.write(
        "Thomas Buhl is a wind-energy engineering leader with 20+ years across research and industry, including professor/department-head and director/VP roles."
    )
    st.write(
        "He combines deep technical expertise in wind turbine design and optimization with international people leadership and strategy execution."
    )


def render_latest_news() -> None:
    st.subheader("Latest News")
    st.caption("Showing the 20 latest items directly related to GE wind turbines.")
    st.caption("Sources include GE Vernova channels and selected financial and wind-industry feeds.")
    st.caption("Use links to open the original source articles.")

    news = load_latest_news()
    if news.empty:
        st.info("No news items could be loaded right now. Try again later.")
        return

    news = news.copy()
    news["published"] = pd.to_datetime(news["published"], errors="coerce")
    news["published_label"] = news["published"].dt.strftime("%Y-%m-%d %H:%M UTC")
    news["published_label"] = news["published_label"].fillna("Unknown date")

    for idx, row in enumerate(news.itertuples(index=False), start=1):
        title = clean_text(row.title) or "Untitled"
        link = clean_text(row.link) or ""
        source = clean_text(row.source) or "Unknown source"
        focus = clean_text(row.focus) or "news"
        when = clean_text(row.published_label) or "Unknown date"
        if link:
            st.markdown(f"{idx}. [{title}]({link})")
        else:
            st.write(f"{idx}. {title}")
        summary = short_summary(row.summary, max_chars=360)
        if summary is None or len(summary) < 80:
            summary = (
                f"This is a GE wind-related {focus} update from {source}. "
                "The source feed did not provide a detailed excerpt, so open the original link for full context."
            )
        st.write(summary)
        st.caption(f"{source} | {focus} | {when}")


def render_latest_patents() -> None:
    st.subheader("Latest Patents")
    st.caption("Showing 10 most recent GE-related wind patent publications from WIPO PATENTSCOPE.")
    st.caption("Inventor names are listed from filings; employment is inferred from applicant records.")

    patents = load_latest_patents()
    if patents.empty:
        st.info("No patent records could be loaded right now. Try again later.")
        return

    view = patents.copy()
    view["publication_label"] = pd.to_datetime(view["publication_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    view["filing_label"] = pd.to_datetime(view["filing_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    view["priority_label"] = pd.to_datetime(view["priority_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    view["publication_label"] = view["publication_label"].fillna("-")
    view["filing_label"] = view["filing_label"].fillna("-")
    view["priority_label"] = view["priority_label"].fillna("-")

    for idx, row in enumerate(view.itertuples(index=False), start=1):
        title = clean_text(row.title) or "Untitled patent record"
        link = clean_text(row.link) or ""
        publication_number = clean_text(row.publication_number) or "Unknown publication number"
        inventor = clean_text(row.inventor) or "Unknown inventor"
        assignee = clean_text(row.assignee) or "Unknown assignee"
        summary = short_summary(row.summary, max_chars=360)
        if summary is None:
            summary = "No abstract snippet available in the source response."

        if link:
            st.markdown(f"{idx}. [{title}]({link})")
        else:
            st.write(f"{idx}. {title}")
        st.write(summary)
        st.caption(
            f"{publication_number} | Published: {row.publication_label} | Filed: {row.filing_label} | Priority: {row.priority_label}"
        )
        st.caption(f"Inventor(s): {inventor} | Assignee: {assignee}")


def render_overall_economics(economy: pd.DataFrame, stock_monthly: pd.DataFrame, market_monthly: pd.DataFrame) -> None:
    st.subheader("Overall Economics")
    if economy.empty:
        st.info("No parsable data found in `economy` sheet.")
        return

    years = sorted(economy["year"].unique().tolist())
    y_min, y_max = int(min(years)), int(max(years))
    if y_min == y_max:
        year_range = (y_min, y_max)
        st.caption(f"Economy year range: {y_min}")
    else:
        year_range = st.slider("Economy year range", y_min, y_max, (max(y_min, 2007), y_max))
    econ = economy[economy["year"].between(year_range[0], year_range[1])].copy()
    if econ.empty:
        st.warning("No economic values available for the selected range.")
        return

    metric_options = sorted(econ["metric"].unique().tolist())

    def metric_key(text: str) -> str:
        norm = normalize_metric_name(text)
        if norm is None:
            return ""
        return re.sub(r"[^a-z0-9]+", "", norm.lower())

    def find_metric(required_tokens: list[str], any_tokens: list[str] | None = None) -> str | None:
        for metric in metric_options:
            key = metric_key(metric)
            if not all(tok in key for tok in required_tokens):
                continue
            if any_tokens is not None and not any(tok in key for tok in any_tokens):
                continue
            return metric
        return None

    def format_metric_value(metric: str, value: float) -> str:
        key = metric_key(metric)
        if "musdmw" in key:
            return f"{value:,.3f}"
        if "busd" in key:
            return f"{value:,.2f}"
        if "mw" in key and "musdmw" not in key:
            return f"{value:,.0f}"
        return f"{value:,.0f}"

    def plot_metric_group(metrics: list[str | None], title: str, y_title: str, height: int = 380) -> None:
        chosen = [m for m in metrics if m is not None]
        if not chosen:
            st.info(f"No metrics found for: {title}")
            return
        frame = econ[econ["metric"].isin(chosen)].copy()
        if frame.empty:
            st.info(f"No values found for: {title}")
            return
        fig = px.line(
            frame,
            x="year",
            y="value",
            color="metric",
            markers=True,
            template=plotly_template(),
            title=title,
            height=height,
        )
        fig.update_layout(
            yaxis_title=y_title,
            legend_title_text="",
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    revenue_metric = find_metric(["revenue"], any_tokens=["usd"]) or find_metric(["revenue"])
    gross_profit_metric = (
        find_metric(["gross", "profit"], any_tokens=["usd"])
        or find_metric(["implied", "gross", "profit"], any_tokens=["usd"])
        or find_metric(["gross", "profit"])
    )
    sales_projects_metric = (
        find_metric(["sales", "projects"], any_tokens=["usd"])
        or find_metric(["revenue", "onshore"], any_tokens=["usd"])
        or find_metric(["revenue", "equipment"], any_tokens=["usd"])
        or find_metric(["sales", "projects"])
    )

    order_intake_usd_metric = (
        find_metric(["order", "intake"], any_tokens=["usd"])
        or find_metric(["orders", "order", "intake"], any_tokens=["usd"])
    )
    order_backlog_total_usd_metric = (
        find_metric(["total", "order", "backlog"], any_tokens=["usd"])
        or find_metric(["order", "backlog", "total"], any_tokens=["usd"])
    )
    order_backlog_wt_usd_metric = find_metric(["order", "backlog", "wind"], any_tokens=["usd"])

    order_intake_mw_metric = find_metric(["order", "mw"], any_tokens=["intake", "inkake"])
    order_backlog_mw_metric = find_metric(["order", "backlog", "wind", "mw"])
    deliveries_mw_metric = find_metric(["deliveries", "mw"])

    latest_year = int(econ["year"].max())
    base_year = int(year_range[0])
    latest = econ[econ["year"] == latest_year].set_index("metric")["value"]
    baseline = econ[econ["year"] == base_year].set_index("metric")["value"]
    card_metrics = [
        revenue_metric,
        gross_profit_metric,
        sales_projects_metric,
        order_intake_usd_metric,
        order_backlog_total_usd_metric,
        order_backlog_wt_usd_metric,
    ]
    card_metrics = [m for m in card_metrics if m is not None]

    if card_metrics:
        for idx in range(0, len(card_metrics), 4):
            row_metrics = card_metrics[idx : idx + 4]
            cols = st.columns(len(row_metrics))
            for col, metric_name in zip(cols, row_metrics, strict=False):
                val = latest.get(metric_name, np.nan)
                value_text = "-"
                if pd.notna(val):
                    value_text = f"{format_metric_value(metric_name, val)} ({latest_year})"
                base_val = baseline.get(metric_name, np.nan)
                delta_text = None
                if pd.notna(val) and pd.notna(base_val):
                    delta_value = val - base_val
                    delta_text = f"{delta_value:+,.2f} ({base_year})"
                col.metric(
                    metric_name,
                    value_text,
                    delta=delta_text,
                )

    g1, g2 = st.columns(2)
    with g1:
        plot_metric_group(
            [revenue_metric, gross_profit_metric, sales_projects_metric],
            "USD Metrics (Revenue, Gross Profit, Project Sales)",
            "USD million (mUSD)",
        )
    with g2:
        plot_metric_group(
            [
                order_intake_usd_metric,
                order_backlog_total_usd_metric,
                order_backlog_wt_usd_metric,
            ],
            "USD Value Metrics (Order Intake and Backlog)",
            "USD million (mUSD)",
        )

    g3 = st.container()
    with g3:
        mw_metrics = [m for m in [order_intake_mw_metric, order_backlog_mw_metric, deliveries_mw_metric] if m is not None]
        if mw_metrics:
            plot_metric_group(
                mw_metrics,
                "MW Metrics (Order Intake, Backlog, Deliveries)",
                "MW",
            )

    st.markdown("**Custom Metric Explorer**")
    default_custom = [
        m
        for m in [
            revenue_metric,
            gross_profit_metric,
            sales_projects_metric,
            order_intake_usd_metric,
            order_backlog_wt_usd_metric,
            order_intake_mw_metric,
            order_backlog_mw_metric,
        ]
        if m is not None
    ]
    selected_metrics = st.multiselect(
        "Select custom metrics",
        options=metric_options,
        default=default_custom if default_custom else metric_options[:8],
    )
    custom_frame = econ[econ["metric"].isin(selected_metrics)].copy()
    if not custom_frame.empty:
        fig_custom = px.line(
            custom_frame,
            x="year",
            y="value",
            color="metric",
            markers=True,
            template=plotly_template(),
            height=460,
            title="Custom Economics View",
        )
        fig_custom.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_custom, width="stretch")

def render_yearly_overview(orders: pd.DataFrame, platforms: pd.DataFrame, unannounced: pd.DataFrame) -> None:
    st.subheader("Year-by-Year Order Intake Overview")
    if orders.empty:
        st.info("No order intake rows after filtering.")
        return

    yearly = (
        orders.groupby("order_year", as_index=False)
        .agg(
            ordered_mw=("size_mw", "sum"),
            orders=("order_id", "nunique"),
            avg_order_mw=("size_mw", "mean"),
            countries=("country", "nunique"),
            customers=("customer", "nunique"),
        )
        .sort_values("order_year")
    )
    platform_yearly = platforms.groupby("order_year", as_index=False).agg(
        platform_mw=("slot_mw", "sum"),
        platforms=("platform", "nunique"),
    )
    yearly = yearly.merge(platform_yearly, on="order_year", how="left")
    if unannounced is not None and not unannounced.empty:
        unannounced_yearly = (
            unannounced.groupby("sheet_year", as_index=False)["unannounced_mw"].sum().rename(columns={"sheet_year": "order_year"})
        )
        yearly = yearly.merge(unannounced_yearly, on="order_year", how="left")
    else:
        yearly["unannounced_mw"] = 0.0

    yearly["unannounced_mw"] = yearly["unannounced_mw"].fillna(0.0)
    yearly["total_with_unannounced_mw"] = yearly["ordered_mw"] + yearly["unannounced_mw"]

    c1, c2 = st.columns(2)
    with c1:
        mw_mix = yearly.melt(
            id_vars=["order_year"],
            value_vars=["ordered_mw", "unannounced_mw"],
            var_name="mw_type",
            value_name="mw",
        )
        mw_type_labels = {
            "ordered_mw": "Announced MW",
            "unannounced_mw": "Unannounced MW",
        }
        mw_mix["mw_type"] = mw_mix["mw_type"].map(mw_type_labels)
        fig1 = px.bar(
            mw_mix,
            x="order_year",
            y="mw",
            color="mw_type",
            barmode="stack",
            template=plotly_template(),
            labels={"order_year": "Year", "mw": "MW", "mw_type": ""},
            title="Announced and Unannounced MW by Year",
            height=410,
        )
        fig1.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig1, width="stretch")

    with c2:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=yearly["order_year"],
                y=yearly["orders"],
                mode="lines+markers",
                name="Orders",
                line=dict(width=3),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=yearly["order_year"],
                y=yearly["avg_order_mw"],
                mode="lines+markers",
                name="Avg order MW",
                yaxis="y2",
                line=dict(width=3),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=yearly["order_year"],
                y=yearly["total_with_unannounced_mw"],
                mode="lines+markers",
                name="Total MW incl. unannounced",
                line=dict(width=3, dash="dot"),
            )
        )
        fig2.update_layout(
            template=plotly_template(),
            title="Order Count, Avg Order Size, and Total MW incl. Unannounced",
            height=410,
            yaxis=dict(title="Orders / MW"),
            yaxis2=dict(title="Avg MW", overlaying="y", side="right"),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        st.plotly_chart(fig2, width="stretch")

    continent_year = (
        orders.groupby(["order_year", "continent"], as_index=False)["size_mw"]
        .sum()
        .sort_values(["continent", "order_year"])
    )
    if not continent_year.empty:
        continent_year["cum_sold_mw"] = continent_year.groupby("continent")["size_mw"].cumsum()
        yearly_totals = (
            continent_year.groupby("order_year", as_index=False)["size_mw"]
            .sum()
            .rename(columns={"size_mw": "year_total_mw"})
        )
        continent_share = continent_year.merge(yearly_totals, on="order_year", how="left")
        continent_share = continent_share[continent_share["year_total_mw"] > 0].copy()
        continent_share["market_share_pct"] = 100.0 * continent_share["size_mw"] / continent_share["year_total_mw"]

        c5, c6 = st.columns(2)
        with c5:
            fig_continent_cum = px.area(
                continent_year,
                x="order_year",
                y="cum_sold_mw",
                color="continent",
                template=plotly_template(),
                labels={"order_year": "Year", "cum_sold_mw": "Accumulated MW", "continent": ""},
                title="Accumulated Sold MW by Continent",
                height=430,
            )
            fig_continent_cum.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_continent_cum, width="stretch")

        with c6:
            fig_continent_share = px.bar(
                continent_share,
                x="order_year",
                y="market_share_pct",
                color="continent",
                barmode="stack",
                template=plotly_template(),
                labels={"order_year": "Year", "market_share_pct": "Market share (%)", "continent": ""},
                title="Continent Market Share by Year",
                hover_data={"size_mw": ":,.0f", "year_total_mw": ":,.0f", "market_share_pct": ":.1f"},
                height=430,
            )
            fig_continent_share.update_layout(
                yaxis=dict(range=[0, 100], ticksuffix="%"),
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_continent_share, width="stretch")

        st.caption("Accumulated chart uses announced order MW by continent.")

    st.dataframe(
        yearly.rename(
            columns={
                "order_year": "Year",
                "ordered_mw": "Ordered MW",
                "unannounced_mw": "Unannounced MW",
                "total_with_unannounced_mw": "Total MW incl. Unannounced",
                "platform_mw": "Platform-Mapped MW",
                "avg_order_mw": "Avg Order MW",
                "orders": "Orders",
                "countries": "Countries",
                "customers": "Customers",
                "platforms": "Platforms",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    chosen_year = st.selectbox("Year detail", options=sorted(yearly["order_year"].unique().tolist(), reverse=True))
    yr_orders = orders[orders["order_year"] == chosen_year]
    yr_platforms = platforms[platforms["order_year"] == chosen_year]
    if unannounced is not None and not unannounced.empty:
        yr_unann = unannounced[unannounced["sheet_year"] == chosen_year]
        if not yr_unann.empty:
            total_unann = yr_unann["unannounced_mw"].sum()
            q_breakdown = ", ".join([f"Q{int(r.quarter)}: {r.unannounced_mw:,.0f} MW" for r in yr_unann.itertuples()])
            st.caption(f"Unannounced in {chosen_year}: {total_unann:,.0f} MW ({q_breakdown})")

    c3, c4 = st.columns(2)
    with c3:
        top_country = (
            yr_orders.groupby("country", as_index=False)["size_mw"]
            .sum()
            .sort_values("size_mw", ascending=False)
            .head(12)
        )
        fig3 = px.bar(
            top_country,
            x="size_mw",
            y="country",
            orientation="h",
            template=plotly_template(),
            title=f"Top Countries in {chosen_year}",
            labels={"size_mw": "MW", "country": ""},
            height=430,
        )
        fig3.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig3, width="stretch")

    with c4:
        top_platform = (
            yr_platforms.groupby("platform", as_index=False)["slot_mw"]
            .sum()
            .sort_values("slot_mw", ascending=False)
            .head(12)
        )
        fig4 = px.bar(
            top_platform,
            x="slot_mw",
            y="platform",
            orientation="h",
            template=plotly_template(),
            title=f"Top Platforms in {chosen_year}",
            labels={"slot_mw": "MW", "platform": ""},
            height=430,
        )
        fig4.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig4, width="stretch")


def render_quarterly_analytics(orders: pd.DataFrame, unannounced: pd.DataFrame) -> None:
    st.subheader("Quarterly Analytics")
    if orders.empty and (unannounced is None or unannounced.empty):
        st.info("No quarterly data available.")
        return

    announced = orders.dropna(subset=["order_date"]).copy()
    if not announced.empty:
        announced["year"] = announced["order_date"].dt.year.astype(int)
        announced["quarter"] = announced["order_date"].dt.quarter.astype(int)
        announced_q = (
            announced.groupby(["year", "quarter"], as_index=False)
            .agg(
                announced_mw=("size_mw", "sum"),
                orders=("order_id", "nunique"),
                avg_order_mw=("size_mw", "mean"),
            )
        )
    else:
        announced_q = pd.DataFrame(columns=["year", "quarter", "announced_mw", "orders", "avg_order_mw"])

    if unannounced is not None and not unannounced.empty:
        unann_q = (
            unannounced.groupby(["sheet_year", "quarter"], as_index=False)["unannounced_mw"]
            .sum()
            .rename(columns={"sheet_year": "year"})
        )
    else:
        unann_q = pd.DataFrame(columns=["year", "quarter", "unannounced_mw"])

    quarter_df = announced_q.merge(unann_q, on=["year", "quarter"], how="outer")
    if quarter_df.empty:
        st.info("No quarterly data available.")
        return

    quarter_df["announced_mw"] = pd.to_numeric(quarter_df["announced_mw"], errors="coerce").fillna(0.0)
    quarter_df["unannounced_mw"] = pd.to_numeric(quarter_df["unannounced_mw"], errors="coerce").fillna(0.0)
    quarter_df["orders"] = pd.to_numeric(quarter_df["orders"], errors="coerce").fillna(0).astype(int)
    quarter_df["total_mw"] = quarter_df["announced_mw"] + quarter_df["unannounced_mw"]
    quarter_df["quarter"] = pd.to_numeric(quarter_df["quarter"], errors="coerce").astype("Int64")
    quarter_df = quarter_df.dropna(subset=["year", "quarter"]).copy()
    quarter_df["year"] = quarter_df["year"].astype(int)
    quarter_df["quarter"] = quarter_df["quarter"].astype(int)
    quarter_df = quarter_df.sort_values(["year", "quarter"])
    quarter_df["quarter_label"] = quarter_df["quarter"].apply(lambda q: f"Q{q}")
    quarter_df["year_quarter"] = quarter_df["year"].astype(str) + "-" + quarter_df["quarter_label"]

    c1, c2 = st.columns(2)
    with c1:
        mw_mix = quarter_df.melt(
            id_vars=["year_quarter", "year", "quarter"],
            value_vars=["announced_mw", "unannounced_mw"],
            var_name="mw_type",
            value_name="mw",
        )
        mw_mix["mw_type"] = mw_mix["mw_type"].map(
            {"announced_mw": "Announced MW", "unannounced_mw": "Unannounced MW"}
        )
        fig1 = px.bar(
            mw_mix,
            x="year_quarter",
            y="mw",
            color="mw_type",
            barmode="stack",
            template=plotly_template(),
            labels={"year_quarter": "Quarter", "mw": "MW", "mw_type": ""},
            title="Quarterly MW Mix (Announced vs Unannounced)",
            height=420,
        )
        fig1.update_layout(xaxis_tickangle=-45, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig1, width="stretch")

    with c2:
        pivot = quarter_df.pivot(index="year", columns="quarter_label", values="total_mw").fillna(0)
        fig2 = px.imshow(
            pivot,
            text_auto=".0f",
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title="Total MW Heatmap (Year x Quarter)",
            height=420,
        )
        fig2.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig2, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        by_quarter = (
            quarter_df.groupby("quarter_label", as_index=False)
            .agg(
                avg_announced_mw=("announced_mw", "mean"),
                avg_unannounced_mw=("unannounced_mw", "mean"),
                avg_total_mw=("total_mw", "mean"),
            )
            .sort_values("quarter_label")
        )
        fig3 = px.bar(
            by_quarter,
            x="quarter_label",
            y=["avg_announced_mw", "avg_unannounced_mw", "avg_total_mw"],
            barmode="group",
            template=plotly_template(),
            title="Average Quarterly MW by Quarter Number",
            labels={"value": "Average MW", "quarter_label": "Quarter"},
            height=420,
        )
        fig3.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig3, width="stretch")

    with c4:
        corr_cols = ["announced_mw", "unannounced_mw", "total_mw", "orders", "avg_order_mw"]
        corr_src = quarter_df[corr_cols].dropna()
        if len(corr_src) >= 4:
            corr = corr_src.corr(numeric_only=True)
            fig4 = px.imshow(
                corr,
                text_auto=".2f",
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu",
                title="Quarterly Correlation Matrix",
                height=420,
            )
            fig4.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig4, width="stretch")

    st.dataframe(
        quarter_df.rename(
            columns={
                "year": "Year",
                "quarter_label": "Quarter",
                "announced_mw": "Announced MW",
                "unannounced_mw": "Unannounced MW",
                "total_mw": "Total MW",
                "orders": "Orders",
                "avg_order_mw": "Avg Order MW",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def render_across_years(orders: pd.DataFrame, platforms: pd.DataFrame) -> None:
    st.subheader("Across-Years Analytics")
    if orders.empty:
        st.info("No order intake rows after filtering.")
        return

    tab_country, tab_platform, tab_rotor, tab_customer = st.tabs(
        ["Country Stats", "Platform Stats", "Rotor and MW Rating", "Customer Stats"]
    )

    with tab_country:
        top_n = st.slider("Top countries to display", 5, 20, 12, key="top_country_years")
        top_countries = top_n_by_mw(orders, "country", "size_mw", top_n)
        country_year = (
            orders[orders["country"].isin(top_countries)]
            .groupby(["order_year", "country"], as_index=False)["size_mw"]
            .sum()
        )
        fig = px.area(
            country_year,
            x="order_year",
            y="size_mw",
            color="country",
            template=plotly_template(),
            labels={"size_mw": "MW", "order_year": "Year"},
            title="Country MW Trends Across Years",
            height=440,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

        c_geo1, c_geo2 = st.columns(2)
        with c_geo1:
            continent_year = (
                orders.groupby(["order_year", "continent"], as_index=False)["size_mw"]
                .sum()
                .sort_values("order_year")
            )
            if continent_year.empty:
                st.info("No continent data available for current filters.")
            else:
                fig_cont = px.bar(
                    continent_year,
                    x="order_year",
                    y="size_mw",
                    color="continent",
                    template=plotly_template(),
                    labels={"size_mw": "MW", "order_year": "Year"},
                    title="Continent MW Trends Across Years",
                    height=430,
                )
                fig_cont.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_cont, width="stretch")

        with c_geo2:
            top_regions = top_n_by_mw(orders[orders["region"] != "Unknown"], "region", "size_mw", 10)
            region_year = (
                orders[orders["region"].isin(top_regions)]
                .groupby(["order_year", "region"], as_index=False)["size_mw"]
                .sum()
                .sort_values("order_year")
            )
            if region_year.empty:
                st.info("No region data available for current filters.")
            else:
                fig_reg = px.area(
                    region_year,
                    x="order_year",
                    y="size_mw",
                    color="region",
                    template=plotly_template(),
                    labels={"size_mw": "MW", "order_year": "Year"},
                    title="Top Regions MW Trends Across Years",
                    height=430,
                )
                fig_reg.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_reg, width="stretch")

        geo_tree = (
            orders.assign(
                continent=orders["continent"].fillna("Unknown"),
                region=orders["region"].fillna("Unknown"),
                country=orders["country"].fillna("Unknown"),
            )
            .groupby(["continent", "region", "country"], as_index=False)["size_mw"]
            .sum()
            .sort_values("size_mw", ascending=False)
        )
        if not geo_tree.empty:
            fig_geo_tree = px.sunburst(
                geo_tree,
                path=["continent", "region", "country"],
                values="size_mw",
                color="continent",
                template=plotly_template(),
                title="Country/Region/Continent Mix (Ordered MW)",
                height=600,
            )
            fig_geo_tree.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_geo_tree, width="stretch")

        table = (
            orders.groupby(["country", "region", "continent"], as_index=False)
            .agg(total_mw=("size_mw", "sum"), orders=("order_id", "nunique"), avg_order_mw=("size_mw", "mean"))
            .sort_values("total_mw", ascending=False)
            .head(20)
        )
        st.dataframe(table, width="stretch", hide_index=True)

    with tab_platform:
        top_n = st.slider("Top platforms to display", 5, 20, 12, key="top_platform_years")
        p_base = platforms[platforms["platform"] != "Unknown"]
        top_platforms = top_n_by_mw(p_base, "platform", "slot_mw", top_n)
        platform_year = (
            p_base[p_base["platform"].isin(top_platforms)]
            .groupby(["order_year", "platform"], as_index=False)["slot_mw"]
            .sum()
        )
        fig = px.area(
            platform_year,
            x="order_year",
            y="slot_mw",
            color="platform",
            template=plotly_template(),
            labels={"slot_mw": "MW", "order_year": "Year"},
            title="Platform MW Trends Across Years",
            height=440,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

        d1, d2 = st.columns(2)
        with d1:
            platform_share = (
                p_base.groupby("platform", as_index=False)["slot_mw"]
                .sum()
                .sort_values("slot_mw", ascending=False)
            )
            if not platform_share.empty:
                top_share = platform_share.head(10).copy()
                other_mw = float(platform_share["slot_mw"].sum() - top_share["slot_mw"].sum())
                if other_mw > 0:
                    top_share = pd.concat(
                        [top_share, pd.DataFrame([{"platform": "Other", "slot_mw": other_mw}])],
                        ignore_index=True,
                    )
                fig_pie_platform = px.pie(
                    top_share,
                    names="platform",
                    values="slot_mw",
                    hole=0.52,
                    template=plotly_template(),
                    title="Platform Share (Ordered MW)",
                    height=430,
                )
                fig_pie_platform.update_traces(textposition="inside", textinfo="percent+label")
                fig_pie_platform.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_pie_platform, width="stretch")

        with d2:
            region_share = (
                p_base.groupby("region", as_index=False)["slot_mw"]
                .sum()
                .sort_values("slot_mw", ascending=False)
            )
            if not region_share.empty:
                fig_pie_region = px.pie(
                    region_share,
                    names="region",
                    values="slot_mw",
                    hole=0.52,
                    template=plotly_template(),
                    title="Region Share Across Platforms (MW)",
                    height=430,
                )
                fig_pie_region.update_traces(textposition="inside", textinfo="percent+label")
                fig_pie_region.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_pie_region, width="stretch")

        table = (
            p_base.groupby("platform", as_index=False)
            .agg(
                total_mw=("slot_mw", "sum"),
                orders=("order_id", "nunique"),
            )
            .sort_values("total_mw", ascending=False)
            .head(20)
        )
        st.dataframe(table, width="stretch", hide_index=True)

    with tab_rotor:
        rot = platforms[(platforms["platform"] != "Unknown") & (platforms["slot_mw"] > 0)].copy()
        rotor_stats = (
            rot.dropna(subset=["rotor_m"])
            .groupby("order_year", as_index=False)
            .agg(rotor_avg=("rotor_m", "mean"), rotor_min=("rotor_m", "min"), rotor_max=("rotor_m", "max"))
        )
        mw_stats = (
            rot.dropna(subset=["mw_rating"])
            .groupby("order_year", as_index=False)
            .agg(mw_avg=("mw_rating", "mean"), mw_min=("mw_rating", "min"), mw_max=("mw_rating", "max"))
        )

        c1, c2 = st.columns(2)
        with c1:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=rotor_stats["order_year"], y=rotor_stats["rotor_max"], mode="lines", line=dict(width=0), showlegend=False))
            fig_r.add_trace(
                go.Scatter(
                    x=rotor_stats["order_year"],
                    y=rotor_stats["rotor_min"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(43,119,173,0.22)",
                    line=dict(width=0),
                    name="Min-Max band",
                )
            )
            fig_r.add_trace(
                go.Scatter(
                    x=rotor_stats["order_year"],
                    y=rotor_stats["rotor_avg"],
                    mode="lines+markers",
                    line=dict(width=3, color="#1f5b86"),
                    name="Average",
                )
            )
            fig_r.update_layout(
                template=plotly_template(),
                title="Rotor Size Across Years (min/avg/max)",
                yaxis_title="Rotor diameter (m)",
                height=430,
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_r, width="stretch")

        with c2:
            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(x=mw_stats["order_year"], y=mw_stats["mw_max"], mode="lines", line=dict(width=0), showlegend=False))
            fig_m.add_trace(
                go.Scatter(
                    x=mw_stats["order_year"],
                    y=mw_stats["mw_min"],
                    mode="lines",
                    fill="tonexty",
                    fillcolor="rgba(241,120,50,0.20)",
                    line=dict(width=0),
                    name="Min-Max band",
                )
            )
            fig_m.add_trace(
                go.Scatter(
                    x=mw_stats["order_year"],
                    y=mw_stats["mw_avg"],
                    mode="lines+markers",
                    line=dict(width=3, color="#ca6318"),
                    name="Average",
                )
            )
            fig_m.update_layout(
                template=plotly_template(),
                title="Platform MW Rating Across Years (min/avg/max)",
                yaxis_title="MW rating",
                height=430,
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_m, width="stretch")

    with tab_customer:
        cust_base = orders[orders["customer"] != "Unknown"].copy()
        top_n = st.slider("Top customers to display", 5, 25, 12, key="top_customers_years")
        top_customers = top_n_by_mw(cust_base, "customer", "size_mw", top_n)
        by_year = (
            cust_base[cust_base["customer"].isin(top_customers)]
            .groupby(["order_year", "customer"], as_index=False)["size_mw"]
            .sum()
        )
        fig = px.bar(
            by_year,
            x="order_year",
            y="size_mw",
            color="customer",
            template=plotly_template(),
            title="Customer MW Bought Across Years",
            labels={"size_mw": "MW", "order_year": "Year"},
            height=430,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

        table = (
            cust_base.groupby("customer", as_index=False)
            .agg(total_mw=("size_mw", "sum"), orders=("order_id", "nunique"), avg_order_mw=("size_mw", "mean"))
            .sort_values("total_mw", ascending=False)
            .head(25)
        )
        st.dataframe(table, width="stretch", hide_index=True)


def render_platform_lens(platforms: pd.DataFrame) -> None:
    st.subheader("Across Platforms")
    if platforms.empty:
        st.info("No platform-level rows after filtering.")
        return

    base = platforms[(platforms["platform"] != "Unknown") & (platforms["slot_mw"] > 0)].copy()
    if base.empty:
        st.info("No platform rows with MW available.")
        return

    options = sorted(base["platform"].unique().tolist())
    default_sel = top_n_by_mw(base, "platform", "slot_mw", min(6, len(options)))
    selected = st.multiselect("Platforms to focus", options=options, default=default_sel)
    view = base[base["platform"].isin(selected)] if selected else base

    timeline = view.groupby(["order_year", "platform"], as_index=False)["slot_mw"].sum()
    fig = px.line(
        timeline,
        x="order_year",
        y="slot_mw",
        color="platform",
        template=plotly_template(),
        markers=True,
        title="Timeline: Sold MW per Platform per Year",
        labels={"slot_mw": "MW", "order_year": "Year"},
        height=430,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        platform_country = view.groupby(["platform", "country"], as_index=False)["slot_mw"].sum()
        top_pc = platform_country.sort_values("slot_mw", ascending=False).head(140)
        fig_t = px.treemap(
            top_pc,
            path=["platform", "country"],
            values="slot_mw",
            template=plotly_template(),
            title="Across Platforms: Country Mix",
            height=520,
        )
        fig_t.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_t, width="stretch")

    with c4:
        focus_platform = st.selectbox("Customer view by platform", options=sorted(view["platform"].unique().tolist()))
        customer_data = (
            view[(view["platform"] == focus_platform) & (view["customer"] != "Unknown")]
            .groupby("customer", as_index=False)["slot_mw"]
            .sum()
            .sort_values("slot_mw", ascending=False)
            .head(15)
        )
        fig_c = px.bar(
            customer_data,
            x="slot_mw",
            y="customer",
            orientation="h",
            template=plotly_template(),
            title=f"Across Platforms: Top Customers for {focus_platform}",
            labels={"slot_mw": "MW", "customer": ""},
            height=520,
        )
        fig_c.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_c, width="stretch")

def render_country_maps(
    orders: pd.DataFrame,
    platforms: pd.DataFrame,
    selected_orders: pd.DataFrame,
    selected_platforms: pd.DataFrame,
) -> None:
    st.markdown("### Zoomable Country Maps")
    map_scope = st.radio(
        "Map scope",
        options=["All countries", "Selected countries"],
        index=0,
        horizontal=True,
        key="country_map_scope",
    )
    if map_scope == "Selected countries":
        map_orders = selected_orders.copy()
        map_platforms = selected_platforms.copy()
    else:
        map_orders = orders.copy()
        map_platforms = platforms.copy()

    map_orders["size_mw"] = pd.to_numeric(map_orders["size_mw"], errors="coerce")
    map_base = (
        map_orders.groupby("country", as_index=False)
        .agg(
            total_mw=("size_mw", "sum"),
            orders=("order_id", "nunique"),
            avg_order_mw=("size_mw", "mean"),
        )
        .copy()
    )
    rc = map_base["country"].apply(country_region_continent)
    map_base["region"] = rc.apply(lambda t: t[0])
    map_base["continent"] = rc.apply(lambda t: t[1])

    valid_map_platforms = map_platforms[(map_platforms["platform"] != "Unknown") & (map_platforms["slot_mw"] > 0)].copy()
    platform_rank = (
        valid_map_platforms.groupby(["country", "platform"], as_index=False)["slot_mw"]
        .sum()
        .sort_values(["country", "slot_mw"], ascending=[True, False])
        .drop_duplicates("country")
        .rename(columns={"platform": "dominant_platform", "slot_mw": "dominant_platform_mw"})
    )
    map_base = map_base.merge(
        platform_rank[["country", "dominant_platform", "dominant_platform_mw"]],
        on="country",
        how="left",
    )
    map_base["dominant_platform"] = map_base["dominant_platform"].fillna("Unknown")

    platform_options = ["All platforms"]
    if not valid_map_platforms.empty:
        platform_options.extend(sorted(valid_map_platforms["platform"].dropna().unique().tolist()))
    selected_platform = st.selectbox(
        "Platform overlay for maps",
        options=platform_options,
        index=0,
        key="country_map_platform_overlay",
    )

    if valid_map_platforms.empty:
        overlay = pd.DataFrame(columns=["country", "platform_overlay_mw"])
    elif selected_platform == "All platforms":
        overlay = (
            valid_map_platforms.groupby("country", as_index=False)["slot_mw"]
            .sum()
            .rename(columns={"slot_mw": "platform_overlay_mw"})
        )
    else:
        overlay = (
            valid_map_platforms[valid_map_platforms["platform"] == selected_platform]
            .groupby("country", as_index=False)["slot_mw"]
            .sum()
            .rename(columns={"slot_mw": "platform_overlay_mw"})
        )

    map_base = map_base.merge(overlay, on="country", how="left")
    map_base["platform_overlay_mw"] = pd.to_numeric(map_base["platform_overlay_mw"], errors="coerce").fillna(0.0)
    map_base["iso3"] = map_base["country"].map(map_country_for_geo)
    map_geo = map_base[map_base["iso3"].notna()].copy()

    if map_geo.empty:
        st.info("No mappable countries available for the current scope.")
        return

    map_metric_labels = {
        "total_mw": "Total sold MW",
        "platform_overlay_mw": f"Sold MW for {selected_platform}",
        "orders": "Number of orders",
    }
    size_metric_labels = {
        "total_mw": "Total sold MW",
        "platform_overlay_mw": f"Sold MW for {selected_platform}",
        "orders": "Number of orders",
    }
    color_metric_labels = {
        "avg_order_mw": "Average order size (MW)",
        "orders": "Number of orders",
    }

    # Clear stale widget state values after metric-option changes.
    if "country_map_bubble_size_metric" in st.session_state and st.session_state["country_map_bubble_size_metric"] not in size_metric_labels:
        del st.session_state["country_map_bubble_size_metric"]
    if "country_map_bubble_color_metric" in st.session_state and st.session_state["country_map_bubble_color_metric"] not in color_metric_labels:
        del st.session_state["country_map_bubble_color_metric"]
    if "country_map_choropleth_metric" in st.session_state and st.session_state["country_map_choropleth_metric"] not in map_metric_labels:
        del st.session_state["country_map_choropleth_metric"]

    map_style = st.selectbox(
        "Map style",
        options=["Bubble map", "Choropleth map"],
        index=0,
        key="country_map_style",
    )

    map_geo["orders"] = pd.to_numeric(map_geo["orders"], errors="coerce")

    if map_style == "Bubble map":
        c5, c6 = st.columns(2)
        with c5:
            bubble_size_metric = st.selectbox(
                "Bubble size metric",
                options=list(size_metric_labels.keys()),
                format_func=lambda k: size_metric_labels[k],
                key="country_map_bubble_size_metric",
            )
        with c6:
            bubble_color_metric = st.selectbox(
                "Bubble color metric",
                options=list(color_metric_labels.keys()),
                format_func=lambda k: color_metric_labels[k],
                key="country_map_bubble_color_metric",
            )

        map_geo[bubble_size_metric] = pd.to_numeric(map_geo[bubble_size_metric], errors="coerce")
        map_geo[bubble_color_metric] = pd.to_numeric(map_geo[bubble_color_metric], errors="coerce")

        bubble_data = map_geo.dropna(subset=[bubble_size_metric]).copy()
        bubble_data = bubble_data[bubble_data[bubble_size_metric] > 0]
        if bubble_data[bubble_color_metric].notna().any():
            bubble_data = bubble_data.dropna(subset=[bubble_color_metric]).copy()
        else:
            bubble_color_metric = "orders"
            bubble_data = bubble_data.dropna(subset=[bubble_color_metric]).copy()

        if bubble_data.empty:
            st.info("No values available for the selected bubble map metrics.")
        else:
            bubble_data["_bubble_size"] = bubble_data[bubble_size_metric].clip(lower=0.1)
            fig_bubble = px.scatter_geo(
                bubble_data,
                locations="iso3",
                locationmode="ISO-3",
                size="_bubble_size",
                color=bubble_color_metric,
                hover_name="country",
                hover_data={
                    "total_mw": ":,.0f",
                    "platform_overlay_mw": ":,.0f",
                    "orders": ":,.0f",
                    "avg_order_mw": ":,.1f",
                    "region": True,
                    "continent": True,
                    "dominant_platform": True,
                    "_bubble_size": False,
                    "iso3": False,
                },
                size_max=45,
                projection="natural earth",
                template=plotly_template(),
                color_continuous_scale="Turbo",
                title=(
                    f"Zoomable Bubble Map: size={size_metric_labels[bubble_size_metric]}, "
                    f"color={color_metric_labels[bubble_color_metric]}"
                ),
                height=600,
            )
            fig_bubble.update_geos(showframe=False, showcoastlines=True, fitbounds="locations", bgcolor="rgba(0,0,0,0)")
            fig_bubble.update_layout(
                coloraxis_colorbar_title=color_metric_labels[bubble_color_metric],
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_bubble, width="stretch")
    else:
        choropleth_metric = st.selectbox(
            "Choropleth metric",
            options=list(map_metric_labels.keys()),
            format_func=lambda k: map_metric_labels[k],
            key="country_map_choropleth_metric",
        )
        map_geo[choropleth_metric] = pd.to_numeric(map_geo[choropleth_metric], errors="coerce")
        choropleth_data = map_geo.dropna(subset=[choropleth_metric]).copy()
        if choropleth_metric in {"total_mw", "platform_overlay_mw", "orders"}:
            choropleth_data = choropleth_data[choropleth_data[choropleth_metric] > 0]

        if choropleth_data.empty:
            st.info("No values available for the selected choropleth metric.")
        else:
            fig_map = px.choropleth(
                choropleth_data,
                locations="iso3",
                locationmode="ISO-3",
                color=choropleth_metric,
                hover_name="country",
                hover_data={
                    "total_mw": ":,.0f",
                    "platform_overlay_mw": ":,.0f",
                    "orders": ":,.0f",
                    "avg_order_mw": ":,.1f",
                    "region": True,
                    "continent": True,
                    "dominant_platform": True,
                    "iso3": False,
                },
                template=plotly_template(),
                color_continuous_scale="YlGnBu",
                title=f"Zoomable Choropleth: {map_metric_labels[choropleth_metric]}",
                height=600,
            )
            fig_map.update_geos(showframe=False, showcoastlines=True, fitbounds="locations", bgcolor="rgba(0,0,0,0)")
            fig_map.update_layout(
                coloraxis_colorbar_title=map_metric_labels[choropleth_metric],
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_map, width="stretch")


def render_country_lens(orders: pd.DataFrame, platforms: pd.DataFrame) -> None:
    st.subheader("Across Countries")
    if orders.empty:
        st.info("No country-level rows after filtering.")
        return

    country_options = sorted(orders["country"].unique().tolist())
    default = top_n_by_mw(orders, "country", "size_mw", min(6, len(country_options)))
    selected_countries = st.multiselect("Countries to focus", options=country_options, default=default)
    if selected_countries:
        o = orders[orders["country"].isin(selected_countries)].copy()
        p = platforms[platforms["country"].isin(selected_countries)].copy()
    else:
        o = orders.copy()
        p = platforms.copy()

    if o.empty:
        st.info("No rows available for selected countries.")
        return

    render_country_maps(orders, platforms, o, p)

    c1, c2 = st.columns(2)
    with c1:
        p_valid = p[(p["platform"] != "Unknown") & (p["slot_mw"] > 0)]
        top_platforms = top_n_by_mw(p_valid, "platform", "slot_mw", 10)
        mix = (
            p_valid[p_valid["platform"].isin(top_platforms)]
            .groupby(["country", "platform"], as_index=False)["slot_mw"]
            .sum()
        )
        fig = px.bar(
            mix,
            x="country",
            y="slot_mw",
            color="platform",
            template=plotly_template(),
            title="Across Country: Platform Mix",
            labels={"slot_mw": "MW", "country": ""},
            height=430,
        )
        fig.update_layout(xaxis_tickangle=-30, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

    with c2:
        country_orders = (
            o.groupby("country", as_index=False)
            .agg(total_mw=("size_mw", "sum"), orders=("order_id", "nunique"))
            .sort_values("total_mw", ascending=False)
        )
        fig2 = px.bar(
            country_orders,
            x="country",
            y="orders",
            template=plotly_template(),
            title="Across Country: Number of Orders",
            labels={"orders": "Orders", "country": ""},
            height=430,
        )
        fig2.update_layout(xaxis_tickangle=-30, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig2, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        order_size_country = (
            o.dropna(subset=["size_mw"])
            .groupby("country", as_index=False)
            .agg(avg_order_mw=("size_mw", "mean"), orders=("order_id", "nunique"))
            .sort_values("avg_order_mw", ascending=False)
        )
        fig3 = px.bar(
            order_size_country,
            x="avg_order_mw",
            y="country",
            orientation="h",
            template=plotly_template(),
            title="Across Country: Avg Order Size",
            labels={"avg_order_mw": "MW", "country": ""},
            height=430,
        )
        fig3.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig3, width="stretch")

    with c4:
        customer_spread = (
            o.groupby("country", as_index=False)
            .agg(customers=("customer", "nunique"), orders=("order_id", "nunique"), total_mw=("size_mw", "sum"))
            .sort_values("customers", ascending=False)
        )
        fig4 = px.bar(
            customer_spread,
            x="customers",
            y="country",
            orientation="h",
            template=plotly_template(),
            title="Across Country: Customer Diversity",
            labels={"customers": "Unique customers", "country": ""},
            height=430,
        )
        fig4.update_layout(yaxis=dict(categoryorder="total ascending"), margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig4, width="stretch")

    c5, c6 = st.columns(2)
    with c5:
        continent_mw = (
            o.groupby("continent", as_index=False)
            .agg(total_mw=("size_mw", "sum"), orders=("order_id", "nunique"))
            .sort_values("total_mw", ascending=False)
        )
        fig5 = px.bar(
            continent_mw,
            x="continent",
            y="total_mw",
            color="continent",
            template=plotly_template(),
            title="Across Countries: Continent Totals",
            labels={"total_mw": "MW", "continent": ""},
            height=430,
        )
        fig5.update_layout(showlegend=False, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig5, width="stretch")

    with c6:
        region_stats = (
            o.groupby("region", as_index=False)
            .agg(
                total_mw=("size_mw", "sum"),
                avg_order_mw=("size_mw", "mean"),
                orders=("order_id", "nunique"),
            )
            .sort_values("total_mw", ascending=False)
            .head(12)
        )
        region_stats = region_stats.dropna(subset=["avg_order_mw", "orders"])
        if region_stats.empty:
            st.info("No complete region-level order/customer data for current filters.")
        else:
            fig6 = px.scatter(
                region_stats,
                x="avg_order_mw",
                y="orders",
                size="total_mw",
                color="region",
                template=plotly_template(),
                title="Across Countries: Region Order Size vs Number of Orders",
                labels={"avg_order_mw": "Avg order size (MW)", "orders": "Orders"},
                height=430,
            )
            fig6.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig6, width="stretch")

    geo_tree = (
        o.assign(
            continent=o["continent"].fillna("Unknown"),
            region=o["region"].fillna("Unknown"),
            country=o["country"].fillna("Unknown"),
        )
        .groupby(["continent", "region", "country"], as_index=False)["size_mw"]
        .sum()
        .sort_values("size_mw", ascending=False)
    )
    if not geo_tree.empty:
        fig_tree = px.sunburst(
            geo_tree,
            path=["continent", "region", "country"],
            values="size_mw",
            color="continent",
            template=plotly_template(),
            title="Across Countries: Continent -> Region -> Country (Ordered MW)",
            height=640,
        )
        fig_tree.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_tree, width="stretch")

    c7, c8 = st.columns(2)
    with c7:
        continent_share = (
            o.groupby("continent", as_index=False)["size_mw"]
            .sum()
            .sort_values("size_mw", ascending=False)
        )
        if not continent_share.empty:
            fig7 = px.pie(
                continent_share,
                names="continent",
                values="size_mw",
                hole=0.52,
                template=plotly_template(),
                title="Across Countries: Continent Share (MW)",
                height=430,
            )
            fig7.update_traces(textposition="inside", textinfo="percent+label")
            fig7.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig7, width="stretch")

    with c8:
        country_share = (
            o.groupby("country", as_index=False)["size_mw"]
            .sum()
            .sort_values("size_mw", ascending=False)
        )
        if not country_share.empty:
            top_country_share = country_share.head(12).copy()
            other_mw = float(country_share["size_mw"].sum() - top_country_share["size_mw"].sum())
            if other_mw > 0:
                top_country_share = pd.concat(
                    [top_country_share, pd.DataFrame([{"country": "Other", "size_mw": other_mw}])],
                    ignore_index=True,
                )
            fig8 = px.pie(
                top_country_share,
                names="country",
                values="size_mw",
                hole=0.52,
                template=plotly_template(),
                title="Across Countries: Top Country Share (MW)",
                height=430,
            )
            fig8.update_traces(textposition="inside", textinfo="percent+label")
            fig8.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig8, width="stretch")


def render_delivery_capacity(orders: pd.DataFrame) -> None:
    _ = orders
    return


def render_correlations(orders: pd.DataFrame) -> None:
    st.subheader("Additional Correlations")
    if orders.empty:
        st.info("No rows for correlation analysis.")
        return

    numeric_cols = ["size_mw", "platform_mw_sum", "turbines_qty_sum"]
    corr_data = orders[numeric_cols].dropna()
    if len(corr_data) >= 12:
        corr = corr_data.corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu",
            title="Numeric Correlation Matrix",
            height=420,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

    scatter_data = orders.dropna(subset=["size_mw", "platform_mw_sum"]).copy()
    if not scatter_data.empty:
        scatter_data["primary_platform"] = scatter_data["primary_platform"].fillna("Unknown")
        scatter_data["platform_mw_sum"] = pd.to_numeric(scatter_data["platform_mw_sum"], errors="coerce")

        scatter_args: dict[str, Any] = {
            "data_frame": scatter_data,
            "x": "size_mw",
            "y": "platform_mw_sum",
            "color": "primary_platform",
            "hover_data": ["country", "customer", "primary_platform", "order_year"],
            "template": plotly_template(),
            "title": "Platform-Mapped MW vs Order Size",
            "labels": {"size_mw": "Order size (MW)", "platform_mw_sum": "Platform-mapped MW"},
            "height": 500,
        }

        if scatter_data["platform_mw_sum"].notna().any():
            positive_vals = scatter_data.loc[scatter_data["platform_mw_sum"] > 0, "platform_mw_sum"]
            fallback_size = float(positive_vals.median()) if not positive_vals.empty else 1.0
            scatter_data["_bubble_size"] = scatter_data["platform_mw_sum"].fillna(fallback_size).clip(lower=0.1)
            scatter_args["size"] = "_bubble_size"
            scatter_args["size_max"] = 38

        try:
            fig2 = px.scatter(**scatter_args)
        except ValueError:
            scatter_args.pop("size", None)
            scatter_args.pop("size_max", None)
            fig2 = px.scatter(**scatter_args)
            st.caption("Bubble size disabled for this view due missing/invalid platform MW values.")
        fig2.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig2, width="stretch")

    if len(corr_data) >= 12:
        flat_corr = (
            corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
            .stack()
            .reset_index()
            .rename(columns={"level_0": "Metric A", "level_1": "Metric B", 0: "Correlation"})
        )
        flat_corr["AbsCorr"] = flat_corr["Correlation"].abs()
        flat_corr = flat_corr.sort_values("AbsCorr", ascending=False).drop(columns=["AbsCorr"]).head(10)
        st.dataframe(flat_corr, width="stretch", hide_index=True)

def apply_global_filters(
    orders: pd.DataFrame, platforms: pd.DataFrame, unannounced: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if orders.empty:
        if unannounced is None:
            unannounced = pd.DataFrame(columns=["sheet_name", "sheet_year", "quarter", "unannounced_mw"])
        return orders, platforms, unannounced

    orders = apply_region_continent_columns(orders)
    platforms = apply_region_continent_columns(platforms)

    st.sidebar.header("Global Filters")
    y_min, y_max = int(orders["order_year"].min()), int(orders["order_year"].max())
    if y_min == y_max:
        year_range = (y_min, y_max)
        st.sidebar.caption(f"Order year: {y_min}")
    else:
        year_range = st.sidebar.slider("Order year range", y_min, y_max, (y_min, y_max))

    continents = sorted([c for c in orders["continent"].dropna().unique().tolist() if c != "Unknown"])
    regions = sorted([r for r in orders["region"].dropna().unique().tolist() if r != "Unknown"])
    countries = sorted(orders["country"].dropna().unique().tolist())
    platform_options = sorted(platforms["platform"].dropna().unique().tolist()) if not platforms.empty else []

    selected_continents = st.sidebar.multiselect("Continents", options=continents, default=[])
    selected_regions = st.sidebar.multiselect("Regions", options=regions, default=[])
    selected_countries = st.sidebar.multiselect("Countries", options=countries, default=[])
    selected_platforms = st.sidebar.multiselect("Platforms", options=platform_options, default=[])

    min_order_mw = float(np.nanmin(orders["size_mw"])) if orders["size_mw"].notna().any() else 0.0
    max_order_mw = float(np.nanmax(orders["size_mw"])) if orders["size_mw"].notna().any() else 1.0
    if math.isclose(min_order_mw, max_order_mw, rel_tol=0.0, abs_tol=1e-9):
        mw_floor = float(min_order_mw)
        st.sidebar.caption(f"Minimum order MW: {mw_floor:,.1f}")
    else:
        mw_floor = st.sidebar.slider(
            "Minimum order MW",
            min_value=float(min_order_mw),
            max_value=float(max_order_mw),
            value=float(min_order_mw),
        )

    o = orders[
        orders["order_year"].between(year_range[0], year_range[1]) & (orders["size_mw"].fillna(0) >= mw_floor)
    ].copy()

    if selected_continents:
        o = o[o["continent"].isin(selected_continents)]
    if selected_regions:
        o = o[o["region"].isin(selected_regions)]
    if selected_countries:
        o = o[o["country"].isin(selected_countries)]

    p = platforms[platforms["order_id"].isin(o["order_id"])].copy()

    if selected_platforms:
        p = p[p["platform"].isin(selected_platforms)]
        o = o[o["order_id"].isin(p["order_id"])]

    if unannounced is None or unannounced.empty:
        u = pd.DataFrame(columns=["sheet_name", "sheet_year", "quarter", "unannounced_mw"])
    else:
        u = unannounced[unannounced["sheet_year"].between(year_range[0], year_range[1])].copy()

    return o, p, u


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="G", layout="wide")
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = False
    st.sidebar.toggle("Dark mode", key="dark_mode")
    dark_mode = bool(st.session_state.get("dark_mode", False))
    apply_page_style(dark_mode)

    st.title(APP_TITLE)
    st.caption("Build on public available data.")

    workbook = find_default_workbook()
    if workbook is None:
        excel_files = [
            f.name
            for pattern in ("*.xlsx", "*.xlsm", "*.xls")
            for f in Path(".").glob(pattern)
            if not f.name.startswith("~$")
        ]
        if excel_files:
            st.warning(
                "No workbook with an 'economy' sheet was found in this folder. "
                f"Excel files detected: {', '.join(sorted(excel_files))}"
            )
        cached = load_data_from_cache_only()
        if cached is None:
            st.error(
                "No valid GE workbook found and no parsed cache available "
                "(`data_cache/ge_parsed_data.pkl`)."
            )
            return
        economy, orders, platforms, unannounced = cached
    else:
        stat = workbook.stat()
        supplement_sig = SUPPLEMENT_FILE.stat().st_mtime_ns if SUPPLEMENT_FILE.exists() else 0
        cache_key = f"{workbook.resolve()}::{stat.st_mtime_ns}::{stat.st_size}::{supplement_sig}"
        try:
            economy, orders, platforms, unannounced = load_data(cache_key, str(workbook))
        except ValueError as exc:
            cached = load_data_from_cache_only()
            if cached is None:
                st.error(str(exc))
                return
            st.warning(f"{exc}. Falling back to cached parsed dataset.")
            economy, orders, platforms, unannounced = cached

    if STOCK_MONTHLY_FILE.exists():
        stock_stat = STOCK_MONTHLY_FILE.stat()
        stock_cache_key = f"{STOCK_MONTHLY_FILE.resolve()}::{stock_stat.st_mtime_ns}::{stock_stat.st_size}"
    else:
        stock_cache_key = "missing_stock_file"
    stock_monthly = load_stock_data(stock_cache_key, str(STOCK_MONTHLY_FILE))
    if MARKET_MONTHLY_FILE.exists():
        market_stat = MARKET_MONTHLY_FILE.stat()
        market_cache_key = f"{MARKET_MONTHLY_FILE.resolve()}::{market_stat.st_mtime_ns}::{market_stat.st_size}"
    else:
        market_cache_key = "missing_market_file"
    market_monthly = load_market_data(market_cache_key, str(MARKET_MONTHLY_FILE))
    results = load_quarterly_results()
    reported_label = str(results["quarterly"].iloc[-1]["label"]) if results else None
    st.caption(latest_data_handled_text(economy, orders, unannounced, stock_monthly, market_monthly, reported_label))

    if orders.empty:
        st.error("No order intake records were parsed from the yearly order sheets.")
        return

    orders_f, platforms_f, unannounced_f = apply_global_filters(orders, platforms, unannounced)
    render_sidebar_footer()

    if orders_f.empty:
        st.warning("Filters returned no rows. Adjust the sidebar filters.")
        return

    intel = get_intel(economy, orders, platforms, stock_monthly, market_monthly)

    with st.container(key="header_kpis"):
        render_header_metrics(orders_f, platforms_f, unannounced_f)
    tabs = st.tabs(
        [
            "Economy Summary",
            "Executive Briefing",
            "Intelligence Lab",
            "Year-by-Year Overview",
            "Quarterly Analytics",
            "Across Years",
            "Platform Analytics",
            "Country Analytics",
            "Latest News",
            "Latest Patents",
            "Information",
        ]
    )

    with tabs[0]:
        render_overall_economics(economy, stock_monthly, market_monthly)
    with tabs[1]:
        render_executive_briefing(intel)
    with tabs[2]:
        render_intelligence_lab(intel, orders_f, platforms_f)
    with tabs[3]:
        render_yearly_overview(orders_f, platforms_f, unannounced_f)
    with tabs[4]:
        render_quarterly_analytics(orders_f, unannounced_f)
    with tabs[5]:
        render_across_years(orders_f, platforms_f)
    with tabs[6]:
        render_platform_lens(platforms_f)
    with tabs[7]:
        render_country_lens(orders_f, platforms_f)
    with tabs[8]:
        render_latest_news()
    with tabs[9]:
        render_latest_patents()
    with tabs[10]:
        render_information_page()


if __name__ == "__main__":
    main()
