"""
06_regenerate_trade_logs.py -- Regenerates and persists per-pair trade logs
for the PRIMARY portfolio only (top_20 / wait_one_day), at EXACT parity of
input with the already-frozen Gate 1 (results/replication/gate1_results.json)
and Gate 2 (results/frozen/gate2_results.json) runs: same formation/trading
windows (_formation_and_trading_windows, duplicated from notebooks/02 and
notebooks/03 -- notebooks in this repo don't import each other), same
point-in-time universes and golden sets, same SSD selection
(formation.select_portfolios), same sigma (the "sigma" column it returns),
same simulator (simulate_pair_wait_one_day), same config.OPEN_TRIGGER_SIGMAS.
No new experiment, no parameter change -- purely a re-run to capture
trade-level detail (individual open/close events per pair, and each pair's
own daily payoff) that Gate 1/Gate 2 never persisted, only summarized via
src/diagnostics.trade_statistics. Needed for PROTOCOL.md §5's per-round-trip
transaction cost model (notebooks/07_cost_grid.py), which requires knowing
exactly which day each round trip closed on.

MANDATORY INTEGRITY CHECK (must pass before anything is written): every
already-published aggregate statistic for top_20/wait_one_day -- committed
AND employed capital (mean_monthly, se_nw, t_stat_nw, annualized_sharpe,
pct_negative_months, max_drawdown, n_months), plus trade_stats
(avg_round_trips_per_pair, avg_holding_duration_days,
pct_pairs_never_opened) -- is recomputed from this run's own regenerated
pair results and compared against the corresponding values already in
gate1_results.json / gate2_results.json (both arms). Every comparison must
match within a 1e-9 RELATIVE tolerance. Any mismatch is a hard failure:
the script prints every offending field and exits WITHOUT writing the
trade-log files, since a mismatch means this regeneration is not actually
at parity of input and the resulting trade logs would be the wrong thing
to build a cost analysis on top of.

Usage: python notebooks/06_regenerate_trade_logs.py
Outputs (ONLY written if the integrity check passes), JSON (not parquet:
consistent with every other results file in this repo, and this project's
volumes -- low tens of MB -- don't need parquet's efficiency; no
indentation, unlike gate1/gate2_results.json, since these are large
machine-consumed dumps, not meant to be read top-to-bottom by hand):
  results/replication/trade_log_gate1.json   {run_id: {trading_days, pairs:
                                                {pair_id: {ticker_1, ticker_2,
                                                rank, sigma, trades,
                                                daily_payoff}}}}
  results/frozen/trade_log_gate2.json        same shape, one level up:
                                               {arm: {run_id: {...}}}
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data.prices import RAW as PRICES_DIR, _reference_days
from data.universe import formation_calendar, load_membership, universe_for_run
from src.diagnostics import annualized_sharpe, max_drawdown, pct_negative_months, trade_statistics
from src.formation import load_formation_returns, load_trading_returns, normalized_price_indices, select_portfolios
from src.inference import newey_west_mean_test
from src.returns import aggregate_portfolio_run, combine_overlapping_portfolios, compound_to_monthly
from src.trading import simulate_pair_wait_one_day

GATE1_RESULTS_JSON = Path("results/replication/gate1_results.json")
GATE2_RESULTS_JSON = Path("results/frozen/gate2_results.json")
GOLDEN_SET_CSV = Path("results/replication/golden_set.csv")
GOLDEN_SET_OOS_CSV = Path("results/frozen/golden_set_oos.csv")
TRADE_LOG_GATE1_JSON = Path("results/replication/trade_log_gate1.json")
TRADE_LOG_GATE2_JSON = Path("results/frozen/trade_log_gate2.json")

PRIMARY_PORTFOLIO, PRIMARY_VARIANT = "top_20", "wait_one_day"
RELATIVE_TOLERANCE = 1e-9


def _formation_and_trading_windows(trading_start, trading_end_approx):
    """Exact FORMATION_DAYS / TRADING_DAYS reference-day windows for a run
    (identical to notebooks/02_gate1_replication.py and
    notebooks/03_gate2_frozen_run.py's own copy of this helper)."""
    pre = _reference_days(
        pd.Timestamp(trading_start) - pd.Timedelta(days=400), pd.Timestamp(trading_start) - pd.Timedelta(days=1)
    )
    formation_days = pre[-config.FORMATION_DAYS:]
    post = _reference_days(trading_start, trading_end_approx)
    trading_days = post[: config.TRADING_DAYS]
    return formation_days, trading_days


def _json_safe(obj):
    """Same NaN/Inf -> None, numpy -> native conversion as the other
    notebooks in this repo."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _run_one_run_primary_only(run_id, r, membership, universe_filter):
    """Same per-run pipeline as _run_one_run in notebooks/02 and
    notebooks/03, restricted to the top_20/wait_one_day combination only:
    same universe construction, same formation/trading windows, same SSD
    selection, same simulator call. Returns pair-level results (trades +
    daily_payoff, kept for persistence) alongside the aggregated run."""
    universe = set(universe_for_run(membership, r.formation_start))
    if universe_filter is not None:
        universe = universe & universe_filter
    universe = sorted(universe)

    formation_days, trading_days = _formation_and_trading_windows(r.trading_start, r.trading_end_approx)
    n_days = len(trading_days)

    formation_returns = load_formation_returns(universe, formation_days[0], formation_days[-1], PRICES_DIR)
    price_index = normalized_price_indices(formation_returns)
    table = select_portfolios(price_index)[PRIMARY_PORTFOLIO]

    trading_returns = load_trading_returns(universe, trading_days[0], trading_days[-1], PRICES_DIR)

    pair_results: dict[str, dict] = {}
    pair_meta: dict[str, dict] = {}
    for rank, row in table.iterrows():
        t1, t2 = row["ticker_1"], row["ticker_2"]
        if t1 not in trading_returns.columns or t2 not in trading_returns.columns:
            continue
        res = simulate_pair_wait_one_day(
            trading_returns[t1].to_numpy(), trading_returns[t2].to_numpy(),
            sigma=row["sigma"], k=config.OPEN_TRIGGER_SIGMAS,
        )
        pair_id = f"{t1}_{t2}_{rank}"
        pair_results[pair_id] = res
        pair_meta[pair_id] = {"ticker_1": t1, "ticker_2": t2, "rank": int(rank), "sigma": float(row["sigma"])}

    agg = aggregate_portfolio_run(pair_results, n_days=n_days, n_selected=config.TOP_PAIRS)

    return {
        "universe_size": len(universe),
        "trading_days": trading_days,
        "agg": agg,
        "pair_results": pair_results,
        "pair_meta": pair_meta,
    }


def _run_window(cal: pd.DataFrame, membership, universe_filter, label: str) -> dict:
    """Runs every monthly run in `cal` through _run_one_run_primary_only,
    accumulating what's needed for both the integrity check (recomputed
    stats) and the persisted trade log (per-pair trades + daily_payoff)."""
    t_start = time.time()
    monthly_committed_by_run: dict[str, pd.Series] = {}
    monthly_employed_by_run: dict[str, pd.Series] = {}
    all_pair_results: list[dict] = []
    trade_log_by_run: dict[str, dict] = {}
    failed_runs: dict[str, str] = {}

    n_runs = len(cal)
    for i, (run_id, r) in enumerate(cal.iterrows()):
        print(f"[{label} {i + 1}/{n_runs}] run {run_id} ...", flush=True)
        try:
            result = _run_one_run_primary_only(run_id, r, membership, universe_filter)
        except Exception as e:  # noqa: BLE001
            failed_runs[run_id] = f"{type(e).__name__}: {e}"
            print(f"  FAILED: {failed_runs[run_id]}")
            continue

        trading_days = result["trading_days"]
        monthly_committed_by_run[run_id] = compound_to_monthly(result["agg"]["committed_return"], trading_days)
        monthly_employed_by_run[run_id] = compound_to_monthly(result["agg"]["employed_return"], trading_days)
        all_pair_results.extend(result["pair_results"].values())

        trade_log_by_run[run_id] = {
            "trading_days": [str(pd.Timestamp(d).date()) for d in trading_days],
            "pairs": {
                pair_id: {
                    **result["pair_meta"][pair_id],
                    "trades": res["trades"],
                    "daily_payoff": np.asarray(res["daily_payoff"]).tolist(),
                }
                for pair_id, res in result["pair_results"].items()
            },
        }

    print(f"[{label}] all runs processed in {time.time() - t_start:.1f}s ({len(failed_runs)} failed).")

    committed_combined = combine_overlapping_portfolios(monthly_committed_by_run)
    employed_combined = combine_overlapping_portfolios(monthly_employed_by_run)

    return {
        "combo_stats": {
            "committed": _combo_stats(committed_combined),
            "employed": _combo_stats(employed_combined),
        },
        "trade_stats": trade_statistics(all_pair_results),
        "trade_log_by_run": trade_log_by_run,
        "failed_runs": failed_runs,
    }


def _combo_stats(monthly_returns: pd.Series) -> dict:
    nw = newey_west_mean_test(monthly_returns.to_numpy(), lags=config.NW_LAGS)
    return {
        "mean_monthly": nw["mean"],
        "se_nw": nw["se"],
        "t_stat_nw": nw["t_stat"],
        "annualized_sharpe": annualized_sharpe(monthly_returns),
        "pct_negative_months": pct_negative_months(monthly_returns),
        "max_drawdown": max_drawdown(monthly_returns),
        "n_months": nw["n"],
    }


def _relative_diff(a, b) -> float:
    """0.0 if both None or exactly equal; inf if only one is None;
    otherwise |a-b| / max(|a|, |b|)."""
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("inf")
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


def _check_integrity(label: str, recomputed: dict, published: dict) -> list[dict]:
    """Compares recomputed combo_stats (committed + employed) and
    trade_stats against the published gate1/gate2_results.json entry for
    top_20/wait_one_day. Returns a list of mismatches (empty if clean)."""
    mismatches = []
    for capital in ("committed", "employed"):
        rec = recomputed["combo_stats"][capital]
        pub = published[capital]
        for field in ("mean_monthly", "se_nw", "t_stat_nw", "annualized_sharpe", "pct_negative_months", "max_drawdown", "n_months"):
            diff = _relative_diff(rec[field], pub[field])
            if diff >= RELATIVE_TOLERANCE:
                mismatches.append({
                    "arm": label, "capital": capital, "field": field,
                    "recomputed": rec[field], "published": pub[field], "relative_diff": diff,
                })
    for field in ("avg_round_trips_per_pair", "avg_holding_duration_days", "pct_pairs_never_opened"):
        rec_v = recomputed["trade_stats"][field]
        pub_v = published["trade_stats"][field]
        diff = _relative_diff(rec_v, pub_v)
        if diff >= RELATIVE_TOLERANCE:
            mismatches.append({
                "arm": label, "capital": "n/a (trade_stats)", "field": field,
                "recomputed": rec_v, "published": pub_v, "relative_diff": diff,
            })
    return mismatches


def main():
    t_start = time.time()
    membership = load_membership(config.CONSTITUENTS_CSV)

    # ---- Gate 1: golden set, replication window ----
    golden_set = set(pd.read_csv(GOLDEN_SET_CSV)["ticker"])
    cal1 = formation_calendar(config.REPLICATION_TRADING_START_FIRST, config.REPLICATION_TRADING_START_LAST)
    gate1_result = _run_window(cal1, membership, golden_set, "gate1")

    # ---- Gate 2: full_universe (no filter) + golden_set_robustness ----
    golden_set_oos = set(pd.read_csv(GOLDEN_SET_OOS_CSV)["ticker"])
    cal2 = formation_calendar(config.OOS_TRADING_START_FIRST, config.OOS_TRADING_START_LAST)
    gate2_arm_filters = {"full_universe": None, "golden_set_robustness": golden_set_oos}
    gate2_results = {
        arm: _run_window(cal2, membership, filt, f"gate2/{arm}")
        for arm, filt in gate2_arm_filters.items()
    }

    print(f"\nAll regeneration runs done in {time.time() - t_start:.1f}s. Running integrity check...\n")

    # ---- Mandatory integrity check ----
    with open(GATE1_RESULTS_JSON) as f:
        gate1_published = json.load(f)
    with open(GATE2_RESULTS_JSON) as f:
        gate2_published = json.load(f)

    all_mismatches = []
    all_mismatches += _check_integrity(
        "gate1",
        gate1_result,
        gate1_published["portfolios"][PRIMARY_PORTFOLIO][PRIMARY_VARIANT],
    )
    for arm in gate2_arm_filters:
        all_mismatches += _check_integrity(
            f"gate2/{arm}",
            gate2_results[arm],
            gate2_published["arms"][arm]["portfolios"][PRIMARY_PORTFOLIO][PRIMARY_VARIANT],
        )

    if all_mismatches:
        print("=" * 90)
        print(f"INTEGRITY CHECK FAILED: {len(all_mismatches)} field(s) exceed the {RELATIVE_TOLERANCE:.0e} relative tolerance.")
        print("=" * 90)
        for m in all_mismatches:
            print(f"  [{m['arm']}] {m['capital']}.{m['field']}: recomputed={m['recomputed']!r} "
                  f"published={m['published']!r} relative_diff={m['relative_diff']:.3e}")
        print("\nRefusing to write trade-log files: this regeneration is not verified to be at parity "
              "of input with the published Gate 1/Gate 2 results. STOPPING per explicit instruction.")
        sys.exit(1)

    print(f"INTEGRITY CHECK PASSED: all fields match within {RELATIVE_TOLERANCE:.0e} relative tolerance "
          "for gate1 and both gate2 arms (committed + employed + trade_stats).")

    # ---- Write trade logs (only reached if the integrity check passed) ----
    TRADE_LOG_GATE1_JSON.parent.mkdir(parents=True, exist_ok=True)
    gate1_out = {
        "meta": {
            "note": "top_20/wait_one_day only, regenerated at parity of input with gate1_results.json "
                    "(see integrity check in this script) to persist per-pair trade-level detail for "
                    "PROTOCOL.md §5's transaction-cost model.",
            "n_runs": len(cal1),
            "failed_runs": gate1_result["failed_runs"],
        },
        "runs": gate1_result["trade_log_by_run"],
    }
    with open(TRADE_LOG_GATE1_JSON, "w") as f:
        json.dump(_json_safe(gate1_out), f)
    print(f"Saved {TRADE_LOG_GATE1_JSON}")

    TRADE_LOG_GATE2_JSON.parent.mkdir(parents=True, exist_ok=True)
    gate2_out = {
        "meta": {
            "note": "top_20/wait_one_day only, regenerated at parity of input with gate2_results.json "
                    "(see integrity check in this script) to persist per-pair trade-level detail for "
                    "PROTOCOL.md §5's transaction-cost model.",
            "n_runs": len(cal2),
        },
        "arms": {
            arm: {
                "failed_runs": gate2_results[arm]["failed_runs"],
                "runs": gate2_results[arm]["trade_log_by_run"],
            }
            for arm in gate2_arm_filters
        },
    }
    with open(TRADE_LOG_GATE2_JSON, "w") as f:
        json.dump(_json_safe(gate2_out), f)
    print(f"Saved {TRADE_LOG_GATE2_JSON}")

    print(f"\nDone in {time.time() - t_start:.1f}s total.")


if __name__ == "__main__":
    main()
