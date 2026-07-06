"""
golden_set.py — Gate 0-bis: builds the "golden set" of tickers with COMPLETE
Yahoo history in every single run of the replication window (2003-2009),
using the already-downloaded price cache. No new downloads.

Rationale (DEVIATIONS.md, see dated entry): Yahoo deletes the entire history
of delisted tickers, not just the days after delisting. Attrition on the full
point-in-time universe is therefore too high (51-64% in 2003-2009) to validate
the engine's mechanical fidelity on that basis. The golden set decouples the
two concerns:
  (a) engine mechanical fidelity -> validated ONLY on the golden set
  (b) quantification of survivorship bias -> stays on the full universe,
      reported as a limitation/robustness check, not as a blocking gate

Usage: python -m data.golden_set
Output: results/replication/golden_set.csv (lista ticker) +
        results/replication/golden_set_summary.txt

build_golden_set() now takes the trading-start window and output
location as parameters (defaults reproduce the call above exactly).
`python -m data.golden_set oos` builds the OOS-window golden set into
results/frozen/golden_set_oos.csv instead.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from data.prices import _reference_days, RAW
from data.universe import formation_calendar, load_membership, universe_for_run


def build_golden_set(
    trading_start_first: str = config.REPLICATION_TRADING_START_FIRST,
    trading_start_last: str = config.REPLICATION_TRADING_START_LAST,
    out_dir: Path = Path("results/replication"),
    out_name: str = "golden_set",
) -> list[str]:
    """
    Builds the golden set for the given trading-start window: tickers with
    a complete Yahoo price history (no missing Adj Close, no zero/missing
    Volume) in every single monthly formation window across that window.

    trading_start_first, trading_start_last: same format as
        config.REPLICATION_TRADING_START_FIRST/LAST ("YYYY-MM"), so this can
        be called on any window (e.g. config.OOS_TRADING_START_FIRST/LAST),
        not just the replication one.
    out_dir, out_name: where to write `{out_name}.csv` and
        `{out_name}_summary.txt`.

    Calling with no arguments reproduces exactly the original
    replication-window behavior (results/replication/golden_set.csv).
    """
    membership = load_membership(config.CONSTITUENTS_CSV)
    cal = formation_calendar(trading_start_first, trading_start_last)
    cache = {p.stem: p for p in RAW.glob("*.parquet")}

    # Candidates = union of point-in-time members across ALL replication runs
    # (this way we don't a priori exclude anyone who entered/left the S&P 500
    # in between, as long as prices are always available).
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

        # a ticker that is NOT a member in this run simply doesn't count
        # (neither for nor against) for this run; but if it is a member and
        # not complete, it leaves the golden set for good (total integrity required).
        not_member = still_golden - members_this_run
        still_golden = complete_this_run | not_member
        per_run_complete_count.append((run_id, len(still_golden)))

    golden = sorted(still_golden)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(golden, name="ticker").to_csv(out_dir / f"{out_name}.csv", index=False)

    with open(out_dir / f"{out_name}_summary.txt", "w") as f:
        f.write(f"Final golden set: {len(golden)} tickers\n")
        f.write(
            f"Initial candidates (union membership {trading_start_first}..{trading_start_last}): "
            f"{len(candidates)}\n\n"
        )
        f.write("Trajectory (run, tickers still golden after this run):\n")
        for run_id, n in per_run_complete_count:
            f.write(f"  {run_id}: {n}\n")

    print(f"Golden set: {len(golden)} tickers out of {len(candidates)} candidates")
    print(f"Saved to {out_dir / f'{out_name}.csv'} and {out_dir / f'{out_name}_summary.txt'}")
    if len(golden) < 100:
        print("\nWARNING: golden set below 100 tickers, may be insufficient")
        print("for SSD/matching with enough variety. Consider relaxing the")
        print("'zero missing days' requirement to 'few days' or narrowing the")
        print("replication window (e.g. 2005-2009 instead of 2003-2009).")
    return golden


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "oos":
        build_golden_set(
            config.OOS_TRADING_START_FIRST, config.OOS_TRADING_START_LAST,
            Path("results/frozen"), "golden_set_oos",
        )
    else:
        build_golden_set()
