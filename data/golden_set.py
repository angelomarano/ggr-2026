"""
golden_set.py — Gate 0-bis: costruisce il "golden set" di titoli con storico
Yahoo COMPLETO in ogni singolo run della finestra di replica (2003-2009),
usando il cache prezzi gia' scaricato. Nessun nuovo download.

Razionale (DEVIATIONS.md, vedi voce datata): Yahoo cancella l'intero storico
dei titoli delistati, non solo i giorni successivi al delisting. L'attrition
sull'universo point-in-time pieno e' quindi troppo alta (51-64% nel 2003-2009)
per validare la fedelta' meccanica del motore su quella base. Il golden set
disaccoppia le due cose:
  (a) fedelta' meccanica del motore -> validata SOLO sul golden set
  (b) quantificazione del bias da sopravvivenza -> resta sull'universo pieno,
      riportata come limite/robustness, non come cancello bloccante

Uso: python -m data.golden_set
Output: results/replication/golden_set.csv (lista ticker) +
        results/replication/golden_set_summary.txt
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from data.prices import _reference_days, RAW
from data.universe import formation_calendar, load_membership, universe_for_run


def build_golden_set() -> list[str]:
    membership = load_membership(config.CONSTITUENTS_CSV)
    cal = formation_calendar(
        config.REPLICATION_TRADING_START_FIRST, config.REPLICATION_TRADING_START_LAST
    )
    cache = {p.stem: p for p in RAW.glob("*.parquet")}

    # Candidati = unione dei membri point-in-time in TUTTI i run di replica
    # (cosi' non escludiamo a priori chi e' entrato/uscito dall'S&P 500 nel
    # mezzo, purche' i prezzi ci siano sempre).
    candidates: set[str] = set()
    for _, r in cal.iterrows():
        candidates |= set(universe_for_run(membership, r.formation_start))

    still_golden = set(candidates)
    per_run_complete_count = []

    for run_id, r in cal.iterrows():
        days = _reference_days(r.formation_start, r.trading_start - pd.Timedelta(days=1))
        days = days[-config.FORMATION_DAYS :]
        members_this_run = set(universe_for_run(membership, r.formation_start))

        complete_this_run = set()
        for t in still_golden & members_this_run:
            p = cache.get(t)
            if p is None:
                continue
            df = pd.read_parquet(p, columns=["Adj Close", "Volume"]).reindex(days)
            if df["Adj Close"].notna().all() and (df["Volume"].fillna(0) > 0).all():
                complete_this_run.add(t)

        # un ticker che NON e' membro in questo run semplicemente non conta
        # (ne' a favore ne' contro) per questo run; ma se e' membro e non e'
        # completo, esce dal golden set per sempre (serve integrita' totale).
        not_member = still_golden - members_this_run
        still_golden = complete_this_run | not_member
        per_run_complete_count.append((run_id, len(still_golden)))

    golden = sorted(still_golden)
    out_dir = Path("results/replication")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(golden, name="ticker").to_csv(out_dir / "golden_set.csv", index=False)

    with open(out_dir / "golden_set_summary.txt", "w") as f:
        f.write(f"Golden set finale: {len(golden)} titoli\n")
        f.write(f"Candidati iniziali (union membership 2003-2009): {len(candidates)}\n\n")
        f.write("Traiettoria (run, titoli ancora golden dopo questo run):\n")
        for run_id, n in per_run_complete_count:
            f.write(f"  {run_id}: {n}\n")

    print(f"Golden set: {len(golden)} titoli su {len(candidates)} candidati")
    print(f"Salvato in {out_dir/'golden_set.csv'} e {out_dir/'golden_set_summary.txt'}")
    if len(golden) < 100:
        print("\nATTENZIONE: golden set sotto 100 titoli, potrebbe essere insufficiente")
        print("per SSD/matching con abbastanza varieta'. Valutare di allentare il")
        print("requisito 'zero giorni mancanti' a 'few giorni' o restringere la")
        print("finestra di replica (es. 2005-2009 invece di 2003-2009).")
    return golden


if __name__ == "__main__":
    build_golden_set()
