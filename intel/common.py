"""Shared constants and helpers for the intelligence layer (pure pandas, no Streamlit)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MEGA_MW = 500.0  # orders at/above this size are "mega" orders (GE's median order is ~135 MW)
GIGA_MW = 1000.0  # orders at/above this size are "jumbo" orders (kept under the shared name GIGA_MW)
QUARTER_END_WINDOW_DAYS = 14
UNKNOWN = "Unknown"

MULTI_COUNTRY_LABELS = {"Unknown"}
RATING_BUCKET_ORDER = ["<2.5 MW", "2.5-3.9 MW", "4.0-5.9 MW", "6.0-9.9 MW", "10+ MW"]
SERIES_ORDER = ["5.5-6.1 MW class (158-164 m rotors)", "3.6-4.2 MW class", "1.5-3.4 MW (older)"]

# Customers appear under several spellings; (regex, group) pairs, first match wins.
CUSTOMER_GROUPS: list[tuple[str, str]] = [
    (r"bbwind", "BBWind and Greenvolt Power"),
    (r"greenvolt", "Greenvolt"),
    (r"ppc renewables|public power", "PPC Renewables"),
    (r"continuum", "Continuum Green Energy"),
    (r"aula energy", "Aula Energy"),
    (r"edf renewables", "EDF Renewables"),
    (r"powerica", "Powerica"),
    (r"ivpc", "IVPC Group"),
    (r"esb", "ESB"),
    (r"eurus", "Eurus Energy"),
]


def reported_results_path() -> Path:
    return Path("data") / "ge_quarterly_results.json"


def supplement_path() -> Path:
    return Path("data") / "ge_orders_supplement.json"


def same_day(year: int, ref: pd.Timestamp) -> pd.Timestamp:
    try:
        return pd.Timestamp(year=year, month=ref.month, day=ref.day)
    except ValueError:
        return pd.Timestamp(year=year, month=ref.month, day=28)


def rating_bucket(value: Any) -> str:
    if value is None or pd.isna(value):
        return UNKNOWN
    v = float(value)
    if v < 2.5:
        return RATING_BUCKET_ORDER[0]
    if v < 4.0:
        return RATING_BUCKET_ORDER[1]
    if v < 6.0:
        return RATING_BUCKET_ORDER[2]
    if v < 10.0:
        return RATING_BUCKET_ORDER[3]
    return RATING_BUCKET_ORDER[4]


_PLATFORM_RE = re.compile(r"^(?:GE)?(\d+(?:\.\d+|\.X)?)-(\d+(?:\.\d+)?)", re.IGNORECASE)


def _parse_platform(name: Any) -> tuple[str, float, float] | None:
    if name is None or pd.isna(name):
        return None
    text = str(name).upper().replace(" ", "").replace(",", ".")
    m = _PLATFORM_RE.match(text)
    if not m:
        return None
    rating_txt, rotor_txt = m.groups()
    rotor = float(rotor_txt)
    if rating_txt.endswith(".X"):
        return f"GE {rating_txt}-{rotor:g}", np.nan, rotor
    rating = float(rating_txt)
    return f"GE {rating:.1f}-{rotor:g}", rating, rotor


def platform_family(name: Any) -> str:
    """'GE6.1-158' -> 'GE 6.1-158'; 'GE6-164' and 'GE6.0-164' -> 'GE 6.0-164'; '2.8-127' -> 'GE 2.8-127'; no model -> 'Unknown'."""
    parsed = _parse_platform(name)
    return parsed[0] if parsed else UNKNOWN


def product_series(name: Any, rating: Any = None) -> str:
    parsed = _parse_platform(name)
    if parsed is None:
        return UNKNOWN
    r = parsed[1]
    if np.isnan(r):
        r = float(rating) if rating is not None and not pd.isna(rating) else float(re.match(r"GE (\d+)", parsed[0]).group(1))
    if r >= 5.5:
        return SERIES_ORDER[0]
    if r >= 3.6:
        return SERIES_ORDER[1]
    return SERIES_ORDER[2]


def customer_group(name: Any) -> str:
    if name is None or pd.isna(name):
        return UNKNOWN
    text = str(name).strip()
    if text.lower() in {"", "unknown", "nan", "none"}:
        return UNKNOWN
    low = text.lower()
    for pattern, group in CUSTOMER_GROUPS:
        if re.search(pattern, low):
            return group
    return text


def prepare_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns the analytics need (does not modify the input).

    ``book_year`` / ``book_quarter`` are the booking quarter when a press release states it, else the announcement quarter.
    """
    o = orders.copy()
    for col in ("booked_year", "booked_q"):
        if col not in o:
            o[col] = np.nan
    o["customer_raw"] = o["customer"]
    o["customer"] = o["customer_raw"].map(customer_group)
    o["country_clean"] = o["country"]
    o["multi_country"] = o["country_clean"].isin(MULTI_COUNTRY_LABELS)
    o["date_known"] = o["order_date"].notna()
    o["date_eff"] = o["order_date"]
    quarter_end = o["order_date"] + pd.offsets.QuarterEnd(0)
    o["days_to_quarter_end"] = (quarter_end - o["order_date"]).dt.days
    o["in_quarter_end_window"] = (o["days_to_quarter_end"] < QUARTER_END_WINDOW_DAYS) & o["date_known"]
    o["order_quarter_label"] = o["order_year"].astype(str) + "-Q" + o["order_quarter"].astype("Int64").astype(str)
    o["book_year"] = o["booked_year"].fillna(o["order_year"]).astype(int)
    o["book_quarter"] = o["booked_q"].fillna(o["order_quarter"]).astype(int)
    o["booked_earlier"] = (o["book_year"] * 4 + o["book_quarter"]) < (o["order_year"] * 4 + o["order_quarter"])
    o["booked_label"] = np.where(o["booked_year"].notna(), o["book_year"].astype(str) + " Q" + o["book_quarter"].astype(str), "-")
    o["series"] = [product_series(p) for p in o["primary_platform"]]
    o["is_mega"] = o["size_mw"] >= MEGA_MW
    o["is_giga"] = o["size_mw"] >= GIGA_MW
    return o


def prepare_platforms(platforms: pd.DataFrame) -> pd.DataFrame:
    p = platforms.copy()
    p["family"] = p["platform"].map(platform_family)
    p["series"] = [product_series(n, r) for n, r in zip(p["platform"], p["mw_rating"])]
    p["rating_bucket"] = p["mw_rating"].map(rating_bucket)
    return p


def as_of_date(orders: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(orders["order_date"].max()).normalize()


def _norm_metric(name: Any) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def eco_wide(economy: pd.DataFrame) -> pd.DataFrame:
    if economy is None or economy.empty:
        return pd.DataFrame()
    e = economy.copy()
    e["key"] = e["metric"].map(_norm_metric)
    return e.pivot_table(index="year", columns="key", values="value", aggfunc="first").sort_index()


def eco_series(wide: pd.DataFrame, *candidates: str) -> pd.Series:
    for cand in candidates:
        key = _norm_metric(cand)
        if key in wide.columns:
            return wide[key].dropna()
    return pd.Series(dtype=float)


def load_quarterly_results(path: Path | None = None) -> dict[str, Any] | None:
    """GE Vernova Wind segment results by calendar quarter (USD million)."""
    path = path or reported_results_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    q = pd.DataFrame(raw.get("quarterly", []))
    if q.empty:
        return None
    q = q.sort_values(["year", "quarter"]).reset_index(drop=True)
    q["period_end"] = [pd.Timestamp(year=int(y), month=3 * int(k), day=1) + pd.offsets.MonthEnd(0) for y, k in zip(q["year"], q["quarter"])]
    q["label"] = "Q" + q["quarter"].astype(int).astype(str) + " " + q["year"].astype(int).astype(str)
    return {
        "quarterly": q,
        "fy_totals": {int(k): v for k, v in raw.get("fy_totals", {}).items()},
        "guidance": raw.get("guidance", {}),
        "source": raw.get("source", ""),
        "retrieved": raw.get("retrieved", ""),
        "notes": raw.get("notes", ""),
    }


def load_supplement_meta(path: Path | None = None) -> dict[str, Any]:
    path = path or supplement_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: raw.get(k) for k in ("orders_checked_through", "orders_check_note", "description", "unlisted_bookings")}


def safe_div(a: float, b: float) -> float:
    if b is None or a is None or pd.isna(a) or pd.isna(b) or b == 0:
        return float("nan")
    return float(a) / float(b)


def pct_change(new: float, old: float) -> float:
    if old is None or pd.isna(old) or old == 0 or new is None or pd.isna(new):
        return float("nan")
    return float(new) / float(old) - 1.0


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))
