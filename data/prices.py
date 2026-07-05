"""
prices.py — Download prezzi (DA ESEGUIRE IN LOCALE) + report di attrition (Gate 0).

Uso:
    python -m data.download            # scarica tutto (ore, riprendibile)
    python -m data.attrition           # genera results/replication/attrition.csv

Decisioni codificate (protocollo §1):
- Fonte primaria: Yahoo (`Adj Close` = proxy total return, necessario per l'indice
  normalizzato GGR). Un ticker SENZA Adj Close Yahoo = attrition dichiarata.
- Stooq: SOLO validazione/spot-check (split-adjusted ma non dividend-adjusted:
  mischiarlo nel matching cambierebbe la definizione di rendimento).
- Ogni fallimento va in data/raw/download_failures.csv: quel file E' la materia
  prima della tabella di attrition, non un errore da nascondere.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

import config
from data.universe import all_tickers_ever, formation_calendar, load_membership, universe_for_run

RAW = Path(config.PRICES_DIR)
EXTRA_SYMBOLS = ["^GSPC", "^VIX"]  # mercato e regime H2
BATCH = 40
MAX_RETRIES = 3


def _save(tkr: str, df: pd.DataFrame) -> None:
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    out = df[keep].dropna(how="all")
    if out.empty:
        raise ValueError("frame vuoto")
    out.to_parquet(RAW / f"{tkr.replace('^','_idx_')}.parquet")


def download_all() -> None:
    import yfinance as yf

    RAW.mkdir(parents=True, exist_ok=True)
    membership = load_membership(config.CONSTITUENTS_CSV)
    tickers = all_tickers_ever(membership, config.DATA_START, config.DATA_END) + EXTRA_SYMBOLS
    todo = [t for t in tickers if not (RAW / f"{t.replace('^','_idx_')}.parquet").exists()]
    print(f"{len(tickers)} simboli totali, {len(todo)} da scaricare (cache ripresa)")

    failures: list[dict] = []
    for i in range(0, len(todo), BATCH):
        chunk = todo[i : i + BATCH]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                data = yf.download(
                    chunk, start=config.DATA_START, end=config.DATA_END,
                    auto_adjust=False, actions=False, group_by="ticker",
                    threads=True, progress=False,
                )
                break
            except Exception as e:  # noqa: BLE001
                print(f"  batch {i//BATCH}: tentativo {attempt} fallito ({e}); retry...")
                time.sleep(10 * attempt)
        else:
            data = None
        for t in chunk:
            try:
                df = data[t] if (data is not None and t in getattr(data.columns, "levels", [[]])[0]) else None
                if df is None or df.dropna(how="all").empty:
                    raise ValueError("nessun dato")
                if "Adj Close" not in df.columns or df["Adj Close"].dropna().empty:
                    raise ValueError("manca Adj Close (no total return)")
                _save(t, df)
            except Exception as e:  # noqa: BLE001
                failures.append({"ticker": t, "error": str(e)})
        print(f"  batch {i//BATCH + 1}/{(len(todo)-1)//BATCH + 1} completato")
        time.sleep(2)

    pd.DataFrame(failures).to_csv(config.FAILURES_LOG, index=False)
    print(f"falliti: {len(failures)} -> {config.FAILURES_LOG}")


# --------------------------------------------------------------------------
# Attrition (Gate 0): per ogni run mensile, quota della membership point-in-time
# con storia COMPLETA nel formation period (tutti i trading day di riferimento
# presenti, Adj Close valido, Volume > 0 — il filtro GGR "no-trade days").
# --------------------------------------------------------------------------

def _reference_days(start, end) -> pd.DatetimeIndex:
    idx = pd.read_parquet(RAW / "_idx_GSPC.parquet").loc[str(start) : str(end)].index
    return idx


def attrition_report() -> pd.DataFrame:
    membership = load_membership(config.CONSTITUENTS_CSV)
    cal = formation_calendar(
        config.REPLICATION_TRADING_START_FIRST, config.OOS_TRADING_START_LAST
    )
    cache = {p.stem: p for p in RAW.glob("*.parquet")}
    rows = []
    for run_id, r in cal.iterrows():
        days = _reference_days(r.formation_start, r.trading_start - pd.Timedelta(days=1))
        days = days[-config.FORMATION_DAYS :]
        members = universe_for_run(membership, r.formation_start)
        n_dl = n_complete = 0
        for t in members:
            p = cache.get(t)
            if p is None:
                continue
            n_dl += 1
            df = pd.read_parquet(p, columns=["Adj Close", "Volume"]).reindex(days)
            if df["Adj Close"].notna().all() and (df["Volume"].fillna(0) > 0).all():
                n_complete += 1
        rows.append(
            {"run": run_id, "members": len(members), "downloaded": n_dl,
             "complete": n_complete, "share_complete": n_complete / len(members)}
        )
        if int(run_id[-2:]) == 1:
            print(rows[-1])
    out = pd.DataFrame(rows).set_index("run")
    dest = Path("results/replication/attrition.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest)
    yearly = out.groupby(out.index.str[:4])["share_complete"].mean()
    print("\nMedia annua share_complete:\n", yearly.round(3).to_string())
    print(f"\nSoglia clausola Gate 0: {config.ATTRITION_MIN_COMPLETE_SHARE} sul 2003-2009")
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "attrition":
        attrition_report()
    else:
        download_all()
