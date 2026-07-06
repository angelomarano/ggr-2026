"""
universe.py — Point-in-time S&P 500 universe.

Source: fja05680/sp500, file "S&P 500 Historical Components & Changes (Updated).csv"
Format: one row per date on which membership changes; `tickers` column with a
comma-separated list. Membership on date d is the last row with date <= d
(as-of lookup). NEVER use the current list for past dates.
"""

from __future__ import annotations

import pandas as pd


def load_membership(csv_path: str) -> pd.DataFrame:
    """Load the CSV and return a DataFrame indexed by date (asc),
    with `tickers` column = frozenset of tickers (original format, with '.')."""
    df = pd.read_csv(csv_path)
    if not {"date", "tickers"}.issubset(df.columns):
        raise ValueError(f"unexpected columns: {df.columns.tolist()}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df["tickers"] = df["tickers"].map(
        lambda s: frozenset(t.strip() for t in s.split(",") if t.strip())
    )
    return df


def constituents_at(membership: pd.DataFrame, date) -> frozenset:
    """Membership as-of `date` (last change with date <= date)."""
    date = pd.Timestamp(date)
    idx = membership.index.searchsorted(date, side="right") - 1
    if idx < 0:
        raise ValueError(f"no membership available before {date.date()}")
    return membership["tickers"].iloc[idx]


def to_yahoo(ticker: str) -> str:
    """Yahoo Finance normalization: share class with '-' (BRK.B -> BRK-B)."""
    return ticker.replace(".", "-")


def formation_calendar(first_trading_month: str, last_trading_month: str) -> pd.DataFrame:
    """Monthly run calendar.

    For each month m in [first, last]: the TRADING period starts on the first
    business day of m; the FORMATION period is the preceding FORMATION_DAYS
    trading days (exact alignment to trading days happens against prices;
    here we use the proxy: formation_start = 12 months earlier).
    Membership is taken at formation_start (protocol §1.2.1).
    """
    months = pd.period_range(first_trading_month, last_trading_month, freq="M")
    rows = []
    for m in months:
        trading_start = m.to_timestamp(how="start")
        formation_start = trading_start - pd.DateOffset(months=12)
        trading_end = trading_start + pd.DateOffset(months=6)
        rows.append(
            {
                "run_id": str(m),
                "formation_start": formation_start,
                "trading_start": trading_start,
                "trading_end_approx": trading_end,
            }
        )
    return pd.DataFrame(rows).set_index("run_id")


def universe_for_run(membership: pd.DataFrame, formation_start) -> list[str]:
    """Tickers (Yahoo format) of the point-in-time membership at formation_start."""
    raw = constituents_at(membership, formation_start)
    return sorted(to_yahoo(t) for t in raw)


def all_tickers_ever(membership: pd.DataFrame, start, end) -> list[str]:
    """Union of tickers ever appearing in membership in [start, end]:
    this is the download set for prices.py (includes future delistings)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    mask = (membership.index >= start) & (membership.index <= end)
    sel = membership.loc[mask, "tickers"]
    # also include the membership as-of start (last change before start)
    base = constituents_at(membership, start)
    out: set[str] = set(base)
    for s in sel:
        out |= s
    return sorted(to_yahoo(t) for t in out)
