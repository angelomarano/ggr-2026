"""
universe.py — Universo S&P 500 point-in-time.

Fonte: fja05680/sp500, file "S&P 500 Historical Components & Changes (Updated).csv"
Formato: una riga per ogni data in cui la membership cambia; colonna `tickers`
con lista separata da virgole. La membership alla data d e' l'ultima riga con
date <= d (as-of lookup). MAI usare la lista attuale per date passate.
"""

from __future__ import annotations

import pandas as pd


def load_membership(csv_path: str) -> pd.DataFrame:
    """Carica il CSV e restituisce un DataFrame indicizzato per data (asc),
    con colonna `tickers` = frozenset di ticker (formato originale, con '.')."""
    df = pd.read_csv(csv_path)
    if not {"date", "tickers"}.issubset(df.columns):
        raise ValueError(f"colonne inattese: {df.columns.tolist()}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df["tickers"] = df["tickers"].map(
        lambda s: frozenset(t.strip() for t in s.split(",") if t.strip())
    )
    return df


def constituents_at(membership: pd.DataFrame, date) -> frozenset:
    """Membership as-of `date` (ultima variazione con data <= date)."""
    date = pd.Timestamp(date)
    idx = membership.index.searchsorted(date, side="right") - 1
    if idx < 0:
        raise ValueError(f"nessuna membership disponibile prima di {date.date()}")
    return membership["tickers"].iloc[idx]


def to_yahoo(ticker: str) -> str:
    """Normalizzazione per Yahoo Finance: share class con '-' (BRK.B -> BRK-B)."""
    return ticker.replace(".", "-")


def formation_calendar(first_trading_month: str, last_trading_month: str) -> pd.DataFrame:
    """Calendario dei run mensili.

    Per ogni mese m in [first, last]: il TRADING period inizia il primo giorno
    lavorativo di m; il FORMATION period sono i FORMATION_DAYS giorni di borsa
    precedenti (l'allineamento esatto ai trading day avviene contro i prezzi;
    qui usiamo il proxy: formation_start = 12 mesi prima).
    La membership si prende a formation_start (protocollo §1.2.1).
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
    """Ticker (formato Yahoo) della membership point-in-time a formation_start."""
    raw = constituents_at(membership, formation_start)
    return sorted(to_yahoo(t) for t in raw)


def all_tickers_ever(membership: pd.DataFrame, start, end) -> list[str]:
    """Unione dei ticker mai apparsi nella membership in [start, end]:
    e' il set di download per prices.py (include i futuri delisted)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    mask = (membership.index >= start) & (membership.index <= end)
    sel = membership.loc[mask, "tickers"]
    # includi anche la membership as-of start (l'ultima variazione prima di start)
    base = constituents_at(membership, start)
    out: set[str] = set(base)
    for s in sel:
        out |= s
    return sorted(to_yahoo(t) for t in out)
