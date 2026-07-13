"""
05_h5_discovery_quality.py -- H5 discovery-quality comparison (PROTOCOL.md
§4/H5): does clustering-restricted pair selection find better pairs than
brute-force search, at the same nominal portfolio size (top-20)?

Sample: 8 runs distributed across the replication window (2003-01 through
2008-12, 72 monthly runs total), golden set universe -- same window/universe
as Gate 1, same cached data, no new downloads. Sampling rule: even spacing
over the 72 runs via np.linspace(0, 71, 8) rounded to the nearest integer
index. This is a NEW declaration, not a literal reproduction of the 5-run
sample used for the earlier Gate 1 SSD/sigma diagnosis (that was an
interactive, one-off check and was never committed as code, so its exact
run selection isn't recoverable from the repo) -- it follows the same
"evenly distributed across the window" spirit, extended from 5 to 8 runs
as requested, and is fully reproducible from this script alone.
Result: ['2003-01', '2003-11', '2004-09', '2005-07', '2006-06', '2007-04',
'2008-02', '2008-12'].

Four top-20 candidate lists per run, all built from the SAME formation
window/universe:
  1. ggr_ssd       -- formation.select_portfolios' top_20 (the existing
                       GGR baseline, unchanged).
  2. cluster_ssd    -- Variant A: selection_cluster.ssd_intra_cluster_ranking
                       (SSD ranking restricted to intra-cluster pairs).
  3. cluster_coint  -- Variant B: selection_cluster.cointegration_intra_
                       cluster_ranking (Engle-Granger + half-life filter,
                       restricted to intra-cluster pairs).
  4. brute_force    -- selection_cluster.brute_force_cointegration_screen's
                       "top_n" (Engle-Granger + half-life + Benjamini-
                       Hochberg, over EVERY pair of the formation universe).

For every list, on the SAME run's TRADING period (already-cached data):
  - discovery quality (H5's primary metric, PROTOCOL.md §4):
      - % of pairs whose OOS spread is stationary: a FRESH Engle-Granger
        test on the trading-period price levels (not a reuse of the
        formation-period regression -- this asks "does the relationship
        still hold on new data", the standard out-of-sample cointegration
        check), ADF p < config.OOS_ADF_PVALUE. A pair is excluded from
        this percentage (not counted as a failure) if the two tickers
        don't have at least MIN_OOS_DAYS_FOR_ADF=30 days of jointly valid
        trading-period prices (declared here, not in PROTOCOL.md: below
        this, an ADF/EG p-value isn't meaningful -- see _last_valid_day
        reuse below).
      - % that converge at least once: simulate_pair_wait_one_day on the
        FULL trading period (it already handles mid-period delisting
        internally); an event {"event": "close", "reason": "crossing"}
        counts as convergence, {"end_of_period", "delisting"} doesn't.
      - half-life OOS distribution: the half_life_days field from the same
        fresh trading-period Engle-Granger test used for stationarity.
  - multiple-testing accounting: n_tests declared per list (intra_cluster_
    pairs' length for lists 2/3, brute_force_cointegration_screen's
    n_tests for list 4; list 1 has no p-value test at all -- SSD ranking
    isn't a hypothesis test -- so it's reported separately as
    n_candidate_pairs, the size of the ranking pool, not conflated with a
    test count) and n_bh_survivors/n_by_survivors (list 4 only -- lists
    2/3 don't run a multiple-testing correction, per PROTOCOL.md §4/H5
    step 5 vs step 6: BH/BY is specifically the brute-force comparator's
    correction, not Variant B's raw p<0.05 filter).
  - search-space reduction: n_total_possible_pairs = C(n_universe, 2) on
    tickers with complete formation-period history, vs
    n_intra_cluster_pairs_tested = sum of C(cluster_size, 2) over all
    clusters (== len(intra_cluster_pairs(labels))) -- the ratio is the
    most direct measure of how much clustering shrinks the search space,
    reported before any BH/BY comparison.
  - net performance (SECONDARY, PROTOCOL.md §4/H5: "il messaggio non e'
    clustering=piu' soldi"): mean monthly return (committed capital, NW
    t-stat) and annualized Sharpe, at nominal n_selected=config.TOP_PAIRS
    (20) for every list regardless of how many pairs it actually
    populated -- same "committed capital divides by the NOMINAL portfolio
    size" convention as every other portfolio in this project (see
    src/returns.py's aggregate_portfolio_run docstring and Gate 1's
    PORTFOLIO_TARGET_SIZE), so the four arms stay comparable even when one
    of them finds fewer than 20 candidates.

Per-run clustering diagnostics are also recorded: method (optics or
kmeans_fallback), n_clusters, largest cluster size/share, noise share
(optics only).

Usage: python notebooks/05_h5_discovery_quality.py
Outputs (results/replication/):
  h5_discovery_quality.json   machine-readable, every number computed
  h5_discovery_quality.md     same numbers as markdown tables, per-run and
                               aggregated across the 8 sampled runs
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data.prices import RAW as PRICES_DIR, _reference_days
from data.universe import formation_calendar, load_membership, universe_for_run
from src.formation import (
    load_formation_returns,
    load_trading_returns,
    normalized_price_indices,
    select_portfolios,
    spread_sigma,
)
from src.inference import newey_west_mean_test
from src.returns import aggregate_portfolio_run, compound_to_monthly
from src.selection_cluster import (
    brute_force_cointegration_screen,
    cluster_tickers,
    cointegration_intra_cluster_ranking,
    engle_granger_pair,
    intra_cluster_pairs,
    pca_loadings,
    ssd_intra_cluster_ranking,
)
from src.trading import _last_valid_day, simulate_pair_wait_one_day

OUT_DIR = Path("results/replication")
GOLDEN_SET_CSV = OUT_DIR / "golden_set.csv"

LIST_NAMES = ("ggr_ssd", "cluster_ssd", "cluster_coint", "brute_force")
MIN_OOS_DAYS_FOR_ADF = 30  # declared here, see module docstring


def _sample_run_ids(cal: pd.DataFrame, n_samples: int = 8) -> list[str]:
    """Even spacing over the replication window's runs (see module
    docstring for why this is a fresh declaration, not a literal replay of
    the earlier 5-run diagnostic sample)."""
    idx = np.round(np.linspace(0, len(cal) - 1, n_samples)).astype(int)
    return [cal.index[i] for i in idx]


def _formation_and_trading_windows(trading_start, trading_end_approx):
    """Exact FORMATION_DAYS / TRADING_DAYS reference-day windows for a run
    (same helper as notebooks/02_gate1_replication.py, duplicated rather
    than imported -- notebooks in this repo are standalone scripts, none
    of them import from one another)."""
    pre = _reference_days(
        pd.Timestamp(trading_start) - pd.Timedelta(days=400), pd.Timestamp(trading_start) - pd.Timedelta(days=1)
    )
    formation_days = pre[-config.FORMATION_DAYS:]
    post = _reference_days(trading_start, trading_end_approx)
    trading_days = post[: config.TRADING_DAYS]
    return formation_days, trading_days


def _clustering_diagnostics(clustering: dict, n_universe: int) -> dict:
    labels = clustering["labels"]
    sizes = labels.value_counts()
    n_labeled = len(labels)
    diag = {
        "method": clustering["method"],
        "n_clusters": int(sizes.shape[0]),
        "largest_cluster_size": int(sizes.max()) if n_labeled else 0,
        "largest_cluster_share_of_labeled": float(sizes.max() / n_labeled) if n_labeled else None,
        "n_labeled": n_labeled,
        "n_universe": n_universe,
    }
    if clustering["method"] == "optics":
        diag["noise_excluded"] = n_universe - n_labeled
        diag["noise_share"] = (n_universe - n_labeled) / n_universe if n_universe else None
    else:
        diag["noise_excluded"] = 0
        diag["noise_share"] = 0.0
    return diag


def _build_four_lists(formation_returns: pd.DataFrame, price_index: pd.DataFrame) -> dict:
    """Returns {"lists": {name: DataFrame w/ ticker_1/ticker_2/...},
    "clustering": cluster_tickers's own output, "n_intra_cluster_pairs":
    int, "n_total_possible_pairs": int (C(n_universe, 2)),
    "brute_force_meta": {n_tests, n_bh_survivors, n_by_survivors}}."""
    ggr_ssd = select_portfolios(price_index)["top_20"]

    loadings = pca_loadings(formation_returns)
    clustering = cluster_tickers(loadings)
    labels = clustering["labels"]

    cluster_ssd = ssd_intra_cluster_ranking(price_index, labels, top_n=config.TOP_PAIRS)
    cluster_coint = cointegration_intra_cluster_ranking(price_index, labels, top_n=config.TOP_PAIRS)
    bf = brute_force_cointegration_screen(price_index, top_n=config.TOP_PAIRS)

    n_universe = price_index.shape[1]
    return {
        "lists": {
            "ggr_ssd": ggr_ssd,
            "cluster_ssd": cluster_ssd,
            "cluster_coint": cluster_coint,
            "brute_force": bf["top_n"],
        },
        "clustering": clustering,
        "n_intra_cluster_pairs": len(intra_cluster_pairs(labels)),
        "n_total_possible_pairs": math.comb(n_universe, 2),
        "brute_force_meta": {
            "n_tests": bf["n_tests"],
            "expected_false_positives": bf["expected_false_positives"],
            "n_bh_survivors": bf["n_bh_survivors"],
            "n_by_survivors": bf["n_by_survivors"],
        },
    }


def _oos_stationarity_and_half_life(trading_returns: pd.DataFrame, t1: str, t2: str) -> dict | None:
    """Fresh Engle-Granger test on the trading-period price level of
    (t1, t2), truncated to the last day both have valid prices
    (_last_valid_day, reused from src/trading.py -- same delisting
    convention as the trading simulator). Returns None (excluded, not a
    failure) if fewer than MIN_OOS_DAYS_FOR_ADF valid days are available."""
    r1 = trading_returns[t1].to_numpy()
    r2 = trading_returns[t2].to_numpy()
    last_day = _last_valid_day(r1, r2)
    if last_day < MIN_OOS_DAYS_FOR_ADF:
        return None
    truncated = pd.DataFrame({t1: r1[:last_day], t2: r2[:last_day]})
    oos_price_index = normalized_price_indices(truncated)
    result = engle_granger_pair(oos_price_index, t1, t2)
    result["stationary"] = result["p_value"] < config.OOS_ADF_PVALUE
    return result


def _converged_at_least_once(trades: list[dict]) -> bool:
    return any(ev["event"] == "close" and ev["reason"] == "crossing" for ev in trades)


def _discovery_quality_and_performance(
    pairs_table: pd.DataFrame,
    price_index: pd.DataFrame,
    trading_returns: pd.DataFrame,
    n_days: int,
) -> dict:
    """Discovery-quality metrics + committed-capital performance for one
    candidate list on one run. See module docstring for every design
    choice referenced by name below."""
    n_stationarity_evaluable = 0
    n_stationary = 0
    half_lives: list[float] = []
    n_converged = 0
    n_simulated = 0
    pair_results = {}

    for _, row in pairs_table.iterrows():
        t1, t2 = row["ticker_1"], row["ticker_2"]
        if t1 not in trading_returns.columns or t2 not in trading_returns.columns:
            continue
        if t1 not in price_index.columns or t2 not in price_index.columns:
            continue

        oos = _oos_stationarity_and_half_life(trading_returns, t1, t2)
        if oos is not None:
            n_stationarity_evaluable += 1
            if oos["stationary"]:
                n_stationary += 1
            if oos["half_life_days"] is not None:
                half_lives.append(oos["half_life_days"])

        sigma = spread_sigma(price_index, t1, t2)
        if sigma == 0.0:
            continue
        res = simulate_pair_wait_one_day(
            trading_returns[t1].to_numpy(), trading_returns[t2].to_numpy(),
            sigma=sigma, k=config.OPEN_TRIGGER_SIGMAS,
        )
        n_simulated += 1
        if _converged_at_least_once(res["trades"]):
            n_converged += 1
        pair_results[f"{t1}_{t2}"] = res

    agg = aggregate_portfolio_run(pair_results, n_days=n_days, n_selected=config.TOP_PAIRS)

    return {
        "n_candidates": len(pairs_table),
        "n_simulated": n_simulated,
        "discovery_quality": {
            "n_stationarity_evaluable": n_stationarity_evaluable,
            "n_oos_stationary": n_stationary,
            "pct_oos_stationary": (n_stationary / n_stationarity_evaluable) if n_stationarity_evaluable else None,
            "n_converged": n_converged,
            "pct_converged_at_least_once": (n_converged / n_simulated) if n_simulated else None,
            "half_life_oos_days": {
                "n": len(half_lives),
                "mean": float(np.mean(half_lives)) if half_lives else None,
                "median": float(np.median(half_lives)) if half_lives else None,
                "min": float(np.min(half_lives)) if half_lives else None,
                "max": float(np.max(half_lives)) if half_lives else None,
            },
        },
        "committed_monthly_return": agg["committed_return"],
    }


def _run_one_run(run_id: str, r, golden_set: set, membership) -> dict:
    universe = sorted(set(universe_for_run(membership, r.formation_start)) & golden_set)
    formation_days, trading_days = _formation_and_trading_windows(r.trading_start, r.trading_end_approx)
    n_days = len(trading_days)

    formation_returns = load_formation_returns(universe, formation_days[0], formation_days[-1], PRICES_DIR)
    price_index = normalized_price_indices(formation_returns)
    trading_returns = load_trading_returns(universe, trading_days[0], trading_days[-1], PRICES_DIR)

    built = _build_four_lists(formation_returns, price_index)

    per_list = {}
    monthly_by_list = {}
    for name in LIST_NAMES:
        table = built["lists"][name]
        result = _discovery_quality_and_performance(table, price_index, trading_returns, n_days)
        monthly_by_list[name] = compound_to_monthly(result.pop("committed_monthly_return"), trading_days)
        per_list[name] = result

    return {
        "n_universe": price_index.shape[1],
        "clustering": _clustering_diagnostics(built["clustering"], price_index.shape[1]),
        "n_total_possible_pairs": built["n_total_possible_pairs"],
        "n_intra_cluster_pairs": built["n_intra_cluster_pairs"],
        "search_space_reduction_ratio": built["n_intra_cluster_pairs"] / built["n_total_possible_pairs"],
        "brute_force_meta": built["brute_force_meta"],
        "per_list": per_list,
        "monthly_by_list": monthly_by_list,
        "trading_days": trading_days,
    }


def _combo_stats(monthly_returns: pd.Series) -> dict:
    if len(monthly_returns) == 0:
        return {"mean_monthly": None, "t_stat_nw": None, "annualized_sharpe": None, "n_months": 0}
    nw = newey_west_mean_test(monthly_returns.to_numpy(), lags=min(config.NW_LAGS, max(len(monthly_returns) - 1, 0)))
    sharpe = float(monthly_returns.mean() / monthly_returns.std(ddof=1) * np.sqrt(12)) if monthly_returns.std(ddof=1) > 0 else None
    return {
        "mean_monthly": nw["mean"],
        "t_stat_nw": nw["t_stat"],
        "annualized_sharpe": sharpe,
        "n_months": nw["n"],
    }


def _json_safe(obj):
    """Same NaN/Inf -> None, numpy -> native conversion as
    notebooks/02_gate1_replication.py, duplicated for the same reason as
    _formation_and_trading_windows (standalone scripts)."""
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


def main():
    t_start = time.time()
    membership = load_membership(config.CONSTITUENTS_CSV)
    golden_set = set(pd.read_csv(GOLDEN_SET_CSV)["ticker"])

    cal = formation_calendar(config.REPLICATION_TRADING_START_FIRST, config.REPLICATION_TRADING_START_LAST)
    sample_ids = _sample_run_ids(cal, n_samples=8)
    print(f"Sampled {len(sample_ids)} runs: {sample_ids}")

    per_run_results = {}
    run_warnings = {}
    failed_runs = {}

    for i, run_id in enumerate(sample_ids):
        r = cal.loc[run_id]
        print(f"[{i + 1}/{len(sample_ids)}] run {run_id} ...", flush=True)
        t_run = time.time()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = _run_one_run(run_id, r, golden_set, membership)
        except Exception as e:  # noqa: BLE001
            failed_runs[run_id] = f"{type(e).__name__}: {e}"
            print(f"  FAILED: {failed_runs[run_id]}")
            continue
        per_run_results[run_id] = result
        if caught:
            messages = [str(w.message) for w in caught]
            run_warnings[run_id] = {"count": len(messages), "sample": messages[:5]}
        print(f"  done in {time.time() - t_run:.1f}s "
              f"(universe={result['n_universe']}, clustering={result['clustering']['method']}, "
              f"n_clusters={result['clustering']['n_clusters']}, "
              f"intra/total pairs={result['n_intra_cluster_pairs']}/{result['n_total_possible_pairs']})")

    print(f"\nAll runs processed in {time.time() - t_start:.1f}s "
          f"({len(failed_runs)} failed, {len(run_warnings)} runs raised warnings).")

    # ---- Per-run performance stats + aggregation across the 8 runs ----
    per_run_stats = {}
    monthly_pooled = {name: [] for name in LIST_NAMES}
    for run_id, result in per_run_results.items():
        per_run_stats[run_id] = {}
        for name in LIST_NAMES:
            monthly = result["monthly_by_list"][name]
            per_run_stats[run_id][name] = _combo_stats(monthly)
            monthly_pooled[name].append(monthly)

    aggregated_performance = {
        name: _combo_stats(pd.concat(monthly_pooled[name])) if monthly_pooled[name] else _combo_stats(pd.Series(dtype=float))
        for name in LIST_NAMES
    }

    # ---- Aggregated discovery quality (pooled across runs, per list) ----
    aggregated_discovery = {}
    for name in LIST_NAMES:
        dq_by_run = [per_run_results[r]["per_list"][name]["discovery_quality"] for r in per_run_results]
        total_evaluable = sum(dq["n_stationarity_evaluable"] for dq in dq_by_run)
        total_stationary = sum(dq["n_oos_stationary"] for dq in dq_by_run)
        total_simulated = sum(per_run_results[r]["per_list"][name]["n_simulated"] for r in per_run_results)
        total_converged = sum(dq["n_converged"] for dq in dq_by_run)
        run_mean_half_lives = [dq["half_life_oos_days"]["mean"] for dq in dq_by_run if dq["half_life_oos_days"]["mean"] is not None]
        aggregated_discovery[name] = {
            "n_stationarity_evaluable": total_evaluable,
            "pct_oos_stationary": (total_stationary / total_evaluable) if total_evaluable else None,
            "n_simulated": total_simulated,
            "pct_converged_at_least_once": (total_converged / total_simulated) if total_simulated else None,
            "half_life_oos_days_mean_of_run_means": float(np.mean(run_mean_half_lives)) if run_mean_half_lives else None,
        }

    # ---- Search-space reduction, aggregated ----
    total_possible = sum(per_run_results[r]["n_total_possible_pairs"] for r in per_run_results)
    total_intra = sum(per_run_results[r]["n_intra_cluster_pairs"] for r in per_run_results)

    results = {
        "meta": {
            "sample_run_ids": sample_ids,
            "n_runs_ok": len(per_run_results),
            "n_runs_sampled": len(sample_ids),
            "failed_runs": failed_runs,
            "runs_with_warnings": run_warnings,
            "min_oos_days_for_adf": MIN_OOS_DAYS_FOR_ADF,
            "elapsed_seconds": time.time() - t_start,
        },
        "per_run": {
            run_id: {
                "n_universe": result["n_universe"],
                "clustering": result["clustering"],
                "n_total_possible_pairs": result["n_total_possible_pairs"],
                "n_intra_cluster_pairs": result["n_intra_cluster_pairs"],
                "search_space_reduction_ratio": result["search_space_reduction_ratio"],
                "brute_force_meta": result["brute_force_meta"],
                "discovery_quality": {name: result["per_list"][name]["discovery_quality"] for name in LIST_NAMES},
                "n_candidates": {name: result["per_list"][name]["n_candidates"] for name in LIST_NAMES},
                "performance": per_run_stats[run_id],
            }
            for run_id, result in per_run_results.items()
        },
        "aggregated": {
            "discovery_quality": aggregated_discovery,
            "performance": aggregated_performance,
            "search_space_reduction": {
                "total_possible_pairs": total_possible,
                "total_intra_cluster_pairs": total_intra,
                "ratio": (total_intra / total_possible) if total_possible else None,
            },
        },
    }

    # _json_safe converts NaN -> None once here, so the markdown report (built
    # from the SAME sanitized dict, not the raw one) never has to special-case
    # NaN separately from None when deciding whether to print "n/a" -- a
    # constant-zero return series (e.g. an empty candidate list) produces a
    # NaN Newey-West t-stat (0/0), which is a real "undefined", not a missing
    # value, but the report should render it the same way as None either way.
    safe_results = _json_safe(results)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "h5_discovery_quality.json", "w") as f:
        json.dump(safe_results, f, indent=2)
    print(f"Saved {OUT_DIR / 'h5_discovery_quality.json'}")

    _write_markdown_report(safe_results)
    print(f"\nDone in {time.time() - t_start:.1f}s total.")


def _list_label(name: str) -> str:
    return {
        "ggr_ssd": "GGR-SSD (baseline)",
        "cluster_ssd": "Cluster+SSD (Variant A)",
        "cluster_coint": "Cluster+Cointegration (Variant B)",
        "brute_force": "Brute-force+BH (comparator)",
    }[name]


def _write_markdown_report(results: dict) -> None:
    meta = results["meta"]
    lines = [
        "# H5 discovery-quality comparison",
        "",
        f"Sample: {meta['n_runs_ok']}/{meta['n_runs_sampled']} runs from the replication window "
        f"(2003-2009, golden set), evenly spaced: {', '.join(meta['sample_run_ids'])}.",
        "",
        "Discovery quality is H5's PRIMARY comparison metric (PROTOCOL.md §4/H5); "
        "net performance is reported SECONDARY, per the protocol's own framing "
        '("il messaggio non e\' clustering=piu\' soldi").',
        "",
        "## Search-space reduction",
        "",
        "| run | n universe | clustering method | n clusters | largest cluster | "
        "noise share | total possible pairs | intra-cluster pairs tested | reduction ratio |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for run_id, r in results["per_run"].items():
        c = r["clustering"]
        noise = f"{c['noise_share']:.1%}" if c["noise_share"] is not None else "n/a"
        lines.append(
            f"| {run_id} | {r['n_universe']} | {c['method']} | {c['n_clusters']} | "
            f"{c['largest_cluster_size']} ({c['largest_cluster_share_of_labeled']:.1%}) | {noise} | "
            f"{r['n_total_possible_pairs']:,} | {r['n_intra_cluster_pairs']:,} | "
            f"{r['search_space_reduction_ratio']:.2%} |"
        )
    agg_ss = results["aggregated"]["search_space_reduction"]
    lines.append(
        f"| **aggregate** | | | | | | {agg_ss['total_possible_pairs']:,} | "
        f"{agg_ss['total_intra_cluster_pairs']:,} | {agg_ss['ratio']:.2%} |"
    )

    lines += [
        "",
        "## Discovery quality (primary), aggregated across the 8 runs",
        "",
        "| list | n stationarity-evaluable | % OOS-stationary | n simulated | "
        "% converged >=1x | mean half-life OOS (days, mean of run means) |",
        "|---|---|---|---|---|---|",
    ]
    for name in LIST_NAMES:
        d = results["aggregated"]["discovery_quality"][name]
        pct_stat = f"{d['pct_oos_stationary']:.1%}" if d["pct_oos_stationary"] is not None else "n/a"
        pct_conv = f"{d['pct_converged_at_least_once']:.1%}" if d["pct_converged_at_least_once"] is not None else "n/a"
        hl = f"{d['half_life_oos_days_mean_of_run_means']:.1f}" if d["half_life_oos_days_mean_of_run_means"] is not None else "n/a"
        lines.append(
            f"| {_list_label(name)} | {d['n_stationarity_evaluable']} | {pct_stat} | "
            f"{d['n_simulated']} | {pct_conv} | {hl} |"
        )

    lines += [
        "",
        "## Discovery quality, per run",
        "",
        "| run | list | n candidates | % OOS-stationary | % converged >=1x | mean half-life OOS |",
        "|---|---|---|---|---|---|",
    ]
    for run_id, r in results["per_run"].items():
        for name in LIST_NAMES:
            dq = r["discovery_quality"][name]
            n_cand = r["n_candidates"][name]
            pct_stat = f"{dq['pct_oos_stationary']:.1%}" if dq["pct_oos_stationary"] is not None else "n/a"
            pct_conv = f"{dq['pct_converged_at_least_once']:.1%}" if dq["pct_converged_at_least_once"] is not None else "n/a"
            hl = f"{dq['half_life_oos_days']['mean']:.1f}" if dq["half_life_oos_days"]["mean"] is not None else "n/a"
            lines.append(f"| {run_id} | {_list_label(name)} | {n_cand} | {pct_stat} | {pct_conv} | {hl} |")

    lines += [
        "",
        "## Multiple-testing accounting",
        "",
        "List 1 (GGR-SSD) runs no hypothesis test at all -- SSD ranking, not reported here. "
        "Lists 2/3's n_tests come from intra_cluster_pairs (no BH/BY correction applied to "
        "them, per PROTOCOL.md §4/H5 step 5 vs step 6). List 4's n_tests/survivors come "
        "directly from brute_force_cointegration_screen.",
        "",
        "| run | cluster_ssd+cluster_coint n_tests (intra-cluster pairs) | "
        "brute_force n_tests | expected false positives | n BH survivors | n BY survivors |",
        "|---|---|---|---|---|---|",
    ]
    for run_id, r in results["per_run"].items():
        bf = r["brute_force_meta"]
        lines.append(
            f"| {run_id} | {r['n_intra_cluster_pairs']:,} | {bf['n_tests']:,} | "
            f"{bf['expected_false_positives']:.1f} | {bf['n_bh_survivors']} | {bf['n_by_survivors']} |"
        )

    lines += [
        "",
        "## Net performance (secondary), aggregated across the 8 runs",
        "",
        "Committed capital, wait-one-day, nominal n_selected=20 for every list "
        "(see module docstring for why the nominal size is used even when a list finds "
        "fewer than 20 candidates).",
        "",
        "| list | mean/month | t (NW) | ann. Sharpe | n months |",
        "|---|---|---|---|---|",
    ]
    for name in LIST_NAMES:
        s = results["aggregated"]["performance"][name]
        mean_s = f"{s['mean_monthly']:.4%}" if s["mean_monthly"] is not None else "n/a"
        t_s = f"{s['t_stat_nw']:.2f}" if s["t_stat_nw"] is not None else "n/a"
        sharpe_s = f"{s['annualized_sharpe']:.2f}" if s["annualized_sharpe"] is not None else "n/a"
        lines.append(f"| {_list_label(name)} | {mean_s} | {t_s} | {sharpe_s} | {s['n_months']} |")

    lines += [
        "",
        "## Anomalies",
        "",
        f"Failed runs: {len(meta['failed_runs'])}.",
    ]
    for run_id, err in meta["failed_runs"].items():
        lines.append(f"- {run_id}: {err}")
    lines.append(f"\nRuns that raised warnings during processing: {len(meta['runs_with_warnings'])}.")
    total_warnings = sum(w["count"] for w in meta["runs_with_warnings"].values())
    lines.append(f"Total warning count across all runs: {total_warnings}.")

    with open(OUT_DIR / "h5_discovery_quality.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {OUT_DIR / 'h5_discovery_quality.md'}")


if __name__ == "__main__":
    main()
