"""
02_gate1_replication.py -- Gate 1 replication engine (PROTOCOL.md §3).

Runs the GGR pairs-trading pipeline over the replication window (72 monthly
trading-period starts, 2003-01 through 2008-12, closing through June 2009),
using ONLY the golden set (results/replication/golden_set.csv) as the
universe -- not the full point-in-time universe.

For each of {top-5, top-20, control(101-120)} portfolios and each of
{same-day, wait-one-day} execution variants, computes committed- and
employed-capital daily returns per run, combines the 6 overlapping monthly
portfolios into a single monthly series (PROTOCOL.md §2.2), and reports
descriptive statistics (mean, Newey-West t/SE, annualized Sharpe, % negative
months, max drawdown, average round-trips per pair, average holding
duration, % pairs never opened) for all 12 combinations.

Falsification tests (PROTOCOL.md §2.4) run ONLY on the primary portfolio
(top-20, wait-one-day, committed capital): factor regression, long/short
alpha decomposition, decile-matched random-pairs bootstrap.

This script computes and reports numbers. It does NOT compare them against
PROTOCOL.md §3's acceptance band and does NOT judge whether Gate 1 passes.

Usage: python notebooks/02_gate1_replication.py
Outputs (results/replication/):
  gate1_results.json     machine-readable, every number computed
  gate1_report.md         same numbers as markdown tables
  gate1_equity_curve.png  cumulative wealth, wait-one-day/committed, 2003-2009
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from data.factors import load_factors
from data.prices import RAW as PRICES_DIR, _reference_days
from data.universe import formation_calendar, load_membership, universe_for_run
from src.diagnostics import annualized_sharpe, max_drawdown, pct_negative_months, trade_statistics
from src.formation import (
    load_formation_returns,
    load_trading_returns,
    normalized_price_indices,
    select_portfolios,
    spread_sigma,
)
from src.inference import (
    decile_matched_bootstrap_pairs,
    factor_regression,
    long_short_leg_regression,
    newey_west_mean_test,
)
from src.returns import aggregate_portfolio_run, combine_overlapping_portfolios, compound_to_monthly
from src.trading import simulate_pair_same_day, simulate_pair_wait_one_day

OUT_DIR = Path("results/replication")
GOLDEN_SET_CSV = OUT_DIR / "golden_set.csv"

PORTFOLIOS = ("top_5", "top_20", "control")
VARIANTS = ("same_day", "wait_one_day")
SIMULATORS = {"same_day": simulate_pair_same_day, "wait_one_day": simulate_pair_wait_one_day}
PORTFOLIO_TARGET_SIZE = {
    "top_5": config.TOP_PAIRS_SMALL,
    "top_20": config.TOP_PAIRS,
    "control": config.CONTROL_PAIRS_RANGE[1] - config.CONTROL_PAIRS_RANGE[0] + 1,
}
PRIMARY_PORTFOLIO, PRIMARY_VARIANT, PRIMARY_CAPITAL = "top_20", "wait_one_day", "committed"
FACTOR_COLS = ("Mkt-RF", "SMB", "HML", "Mom", "ST_Rev")


def _formation_and_trading_windows(trading_start, trading_end_approx):
    """Exact FORMATION_DAYS / TRADING_DAYS reference-day windows for a run."""
    pre = _reference_days(
        pd.Timestamp(trading_start) - pd.Timedelta(days=400), pd.Timestamp(trading_start) - pd.Timedelta(days=1)
    )
    formation_days = pre[-config.FORMATION_DAYS:]
    post = _reference_days(trading_start, trading_end_approx)
    trading_days = post[: config.TRADING_DAYS]
    return formation_days, trading_days


def _to_excess(monthly_returns: pd.Series, factors_df: pd.DataFrame) -> pd.Series:
    """Align monthly_returns with factors_df's RF column (inner join) and
    subtract it. Months missing from the factor cache are dropped, not
    filled - see data/factors.py's own inner-join rationale."""
    aligned = pd.concat([monthly_returns.rename("r"), factors_df["RF"]], axis=1, join="inner").dropna()
    return aligned["r"] - aligned["RF"]


def _run_one_run(run_id, r, golden_set, membership):
    """Process a single monthly run: pair selection, simulation for all
    portfolios/variants, and the decile-matched bootstrap for the primary
    portfolio. Returns a dict of everything the caller needs to accumulate.
    Raises on unexpected failure; caller is responsible for catching it."""
    universe = sorted(set(universe_for_run(membership, r.formation_start)) & golden_set)

    formation_days, trading_days = _formation_and_trading_windows(r.trading_start, r.trading_end_approx)
    n_days = len(trading_days)

    formation_returns = load_formation_returns(universe, formation_days[0], formation_days[-1], PRICES_DIR)
    price_index = normalized_price_indices(formation_returns)
    portfolios = select_portfolios(price_index)

    trading_returns = load_trading_returns(universe, trading_days[0], trading_days[-1], PRICES_DIR)

    prior_month_returns = (1 + formation_returns.tail(21)).prod() - 1

    daily_by_portfolio_variant = {}
    pair_results_by_portfolio_variant = {}
    long_short_daily = None

    for portfolio_name in PORTFOLIOS:
        table = portfolios[portfolio_name]
        n_selected = PORTFOLIO_TARGET_SIZE[portfolio_name]
        for variant in VARIANTS:
            simulate = SIMULATORS[variant]
            pair_results = {}
            for rank, row in table.iterrows():
                t1, t2 = row["ticker_1"], row["ticker_2"]
                if t1 not in trading_returns.columns or t2 not in trading_returns.columns:
                    continue
                res = simulate(
                    trading_returns[t1].to_numpy(), trading_returns[t2].to_numpy(),
                    sigma=row["sigma"], k=config.OPEN_TRIGGER_SIGMAS,
                )
                pair_results[f"{t1}_{t2}_{rank}"] = res
            agg = aggregate_portfolio_run(pair_results, n_days=n_days, n_selected=n_selected)
            daily_by_portfolio_variant[(portfolio_name, variant)] = agg
            pair_results_by_portfolio_variant[(portfolio_name, variant)] = list(pair_results.values())

            if portfolio_name == PRIMARY_PORTFOLIO and variant == PRIMARY_VARIANT:
                long_short_daily = agg[["long_committed_return", "short_committed_return"]]

    # Decile-matched bootstrap (PROTOCOL.md §2.4, falsification 1) --
    # primary portfolio/variant only.
    primary_table = portfolios[PRIMARY_PORTFOLIO]
    selected_pairs = list(zip(primary_table["ticker_1"], primary_table["ticker_2"]))
    bootstrap_reps = decile_matched_bootstrap_pairs(
        selected_pairs, prior_month_returns, n_deciles=10,
        n_reps=config.RANDOM_PAIRS_BOOTSTRAP_REPS, seed=config.SEED,
    )
    bootstrap_monthly = []
    for rep_pairs in bootstrap_reps:
        fake_results = {}
        for idx, (t1, t2) in enumerate(rep_pairs):
            if t1 not in price_index.columns or t2 not in price_index.columns:
                continue
            sigma = spread_sigma(price_index, t1, t2)
            if sigma == 0.0:
                continue
            if t1 not in trading_returns.columns or t2 not in trading_returns.columns:
                continue
            res = simulate_pair_wait_one_day(
                trading_returns[t1].to_numpy(), trading_returns[t2].to_numpy(),
                sigma=sigma, k=config.OPEN_TRIGGER_SIGMAS,
            )
            fake_results[f"{t1}_{t2}_{idx}"] = res
        fake_agg = aggregate_portfolio_run(fake_results, n_days=n_days, n_selected=config.TOP_PAIRS)
        bootstrap_monthly.append(compound_to_monthly(fake_agg["committed_return"], trading_days))

    return {
        "universe_size": len(universe),
        "trading_days": trading_days,
        "daily_by_portfolio_variant": daily_by_portfolio_variant,
        "pair_results_by_portfolio_variant": pair_results_by_portfolio_variant,
        "long_short_daily": long_short_daily,
        "bootstrap_monthly": bootstrap_monthly,
    }


def _combo_stats(monthly_returns: pd.Series) -> dict:
    nw = newey_west_mean_test(monthly_returns.to_numpy(), lags=config.NW_LAGS)
    return {
        "mean_monthly": nw["mean"],
        "se_nw": nw["se"],
        "t_stat_nw": nw["t_stat"],
        "p_value_nw": nw["p_value"],
        "annualized_sharpe": annualized_sharpe(monthly_returns),
        "pct_negative_months": pct_negative_months(monthly_returns),
        "max_drawdown": max_drawdown(monthly_returns),
        "n_months": nw["n"],
    }


def _json_safe(obj):
    """Recursively replace NaN/Inf with None and numpy scalars with native
    Python types, so json.dumps never chokes or emits non-standard NaN
    literals."""
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
    factors_df = load_factors()

    cal = formation_calendar(config.REPLICATION_TRADING_START_FIRST, config.REPLICATION_TRADING_START_LAST)

    daily_by_run: dict[tuple, dict[str, pd.DataFrame]] = {
        (p, v): {} for p in PORTFOLIOS for v in VARIANTS
    }
    trading_days_by_run: dict[str, pd.DatetimeIndex] = {}
    all_pair_results: dict[tuple, list] = {(p, v): [] for p in PORTFOLIOS for v in VARIANTS}
    long_short_daily_by_run: dict[str, pd.DataFrame] = {}
    bootstrap_monthly_by_run: dict[str, list] = {}
    universe_size_by_run: dict[str, int] = {}
    run_warnings: dict[str, dict] = {}
    failed_runs: dict[str, str] = {}

    n_runs = len(cal)
    for i, (run_id, r) in enumerate(cal.iterrows()):
        print(f"[{i + 1}/{n_runs}] run {run_id} ...", flush=True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = _run_one_run(run_id, r, golden_set, membership)
        except Exception as e:  # noqa: BLE001
            failed_runs[run_id] = f"{type(e).__name__}: {e}"
            print(f"  FAILED: {failed_runs[run_id]}")
            continue

        universe_size_by_run[run_id] = result["universe_size"]
        trading_days_by_run[run_id] = result["trading_days"]
        for (p, v), df in result["daily_by_portfolio_variant"].items():
            daily_by_run[(p, v)][run_id] = df
        for (p, v), pair_list in result["pair_results_by_portfolio_variant"].items():
            all_pair_results[(p, v)].extend(pair_list)
        long_short_daily_by_run[run_id] = result["long_short_daily"]
        bootstrap_monthly_by_run[run_id] = result["bootstrap_monthly"]

        if caught:
            messages = [str(w.message) for w in caught]
            run_warnings[run_id] = {"count": len(messages), "sample": messages[:5]}

    print(f"\nAll runs processed in {time.time() - t_start:.1f}s "
          f"({len(failed_runs)} failed, {len(run_warnings)} runs raised warnings).")

    # ---- Combine daily -> monthly -> single series, per (portfolio, variant, capital measure) ----
    combined_monthly = {}
    for (p, v), runs in daily_by_run.items():
        for capital in ("committed", "employed"):
            monthly_by_run = {
                run_id: compound_to_monthly(df[f"{capital}_return"], trading_days_by_run[run_id])
                for run_id, df in runs.items()
            }
            combined_monthly[(p, v, capital)] = combine_overlapping_portfolios(monthly_by_run)

    # ---- Descriptive stats for all 12 combinations ----
    portfolio_stats = {p: {} for p in PORTFOLIOS}
    for p in PORTFOLIOS:
        for v in VARIANTS:
            portfolio_stats[p][v] = {
                "committed": _combo_stats(combined_monthly[(p, v, "committed")]),
                "employed": _combo_stats(combined_monthly[(p, v, "employed")]),
                "trade_stats": trade_statistics(all_pair_results[(p, v)]),
            }

    # ---- Falsifications (primary portfolio only) ----
    primary_monthly = combined_monthly[(PRIMARY_PORTFOLIO, PRIMARY_VARIANT, PRIMARY_CAPITAL)]
    primary_excess = _to_excess(primary_monthly, factors_df)
    factor_reg_result = factor_regression(primary_excess, factors_df, factor_cols=FACTOR_COLS)

    long_monthly_by_run = {
        run_id: compound_to_monthly(df["long_committed_return"], trading_days_by_run[run_id])
        for run_id, df in long_short_daily_by_run.items()
    }
    short_monthly_by_run = {
        run_id: compound_to_monthly(df["short_committed_return"], trading_days_by_run[run_id])
        for run_id, df in long_short_daily_by_run.items()
    }
    long_combined = combine_overlapping_portfolios(long_monthly_by_run)
    short_combined = combine_overlapping_portfolios(short_monthly_by_run)
    long_excess = _to_excess(long_combined, factors_df)
    short_excess = _to_excess(short_combined, factors_df)
    long_short_result = long_short_leg_regression(long_excess, short_excess, factors_df, factor_cols=FACTOR_COLS)

    n_reps = config.RANDOM_PAIRS_BOOTSTRAP_REPS
    bootstrap_rep_means = []
    for b in range(n_reps):
        monthly_by_run_b = {
            run_id: series_list[b] for run_id, series_list in bootstrap_monthly_by_run.items() if len(series_list) > b
        }
        combined_b = combine_overlapping_portfolios(monthly_by_run_b)
        bootstrap_rep_means.append(float(combined_b.mean()))
    bootstrap_rep_means = np.array(bootstrap_rep_means)
    bootstrap_summary = {
        "n_reps": n_reps,
        "mean_of_rep_means": float(bootstrap_rep_means.mean()),
        "std_of_rep_means": float(bootstrap_rep_means.std(ddof=1)),
        "min": float(bootstrap_rep_means.min()),
        "max": float(bootstrap_rep_means.max()),
        "pct_5": float(np.percentile(bootstrap_rep_means, 5)),
        "pct_95": float(np.percentile(bootstrap_rep_means, 95)),
        "real_primary_mean_monthly": float(primary_monthly.mean()),
    }

    # ---- Assemble output ----
    universe_sizes = list(universe_size_by_run.values())
    results = {
        "meta": {
            "n_runs": n_runs,
            "n_runs_ok": n_runs - len(failed_runs),
            "trading_start_first": config.REPLICATION_TRADING_START_FIRST,
            "trading_start_last": config.REPLICATION_TRADING_START_LAST,
            "universe": "golden_set (results/replication/golden_set.csv)",
            "golden_set_size": len(golden_set),
            "universe_size_per_run": {
                "min": min(universe_sizes) if universe_sizes else None,
                "max": max(universe_sizes) if universe_sizes else None,
                "mean": float(np.mean(universe_sizes)) if universe_sizes else None,
            },
            "failed_runs": failed_runs,
            "runs_with_warnings": run_warnings,
            "elapsed_seconds": time.time() - t_start,
        },
        "portfolios": portfolio_stats,
        "falsifications": {
            "primary_portfolio": f"{PRIMARY_PORTFOLIO} / {PRIMARY_VARIANT} / {PRIMARY_CAPITAL}",
            "factor_regression": factor_reg_result,
            "long_short_decomposition": long_short_result,
            "decile_bootstrap": bootstrap_summary,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "gate1_results.json", "w") as f:
        json.dump(_json_safe(results), f, indent=2)
    print(f"Saved {OUT_DIR / 'gate1_results.json'}")

    _write_markdown_report(results, combined_monthly)
    _plot_equity_curve(combined_monthly)

    print(f"\nDone in {time.time() - t_start:.1f}s total.")


def _write_markdown_report(results: dict, combined_monthly: dict) -> None:
    meta = results["meta"]
    lines = [
        "# Gate 1 replication report",
        "",
        f"Replication window: trading start {meta['trading_start_first']} to {meta['trading_start_last']} "
        "(last trading period closes ~June 2009).",
        f"Universe: golden set only ({meta['golden_set_size']} tickers), "
        f"{meta['n_runs_ok']}/{meta['n_runs']} runs completed.",
        f"Universe size per run: min {meta['universe_size_per_run']['min']}, "
        f"max {meta['universe_size_per_run']['max']}, "
        f"mean {meta['universe_size_per_run']['mean']:.1f}.",
        "",
        "No comparison against PROTOCOL.md §3's acceptance band is made here; "
        "these are the computed numbers only.",
        "",
        "## Descriptive statistics",
        "",
        "| portfolio | variant | capital | mean/month | SE (NW) | t (NW) | ann. Sharpe | % neg months | max drawdown |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in PORTFOLIOS:
        for v in VARIANTS:
            for capital in ("committed", "employed"):
                s = results["portfolios"][p][v][capital]
                lines.append(
                    f"| {p} | {v} | {capital} | {s['mean_monthly']:.4%} | {s['se_nw']:.4%} | "
                    f"{s['t_stat_nw']:.2f} | {s['annualized_sharpe']:.2f} | "
                    f"{s['pct_negative_months']:.1%} | {s['max_drawdown']:.1%} |"
                )

    lines += [
        "",
        "## Trade statistics",
        "",
        "| portfolio | variant | avg round-trips/pair | avg holding days | % pairs never opened |",
        "|---|---|---|---|---|",
    ]
    for p in PORTFOLIOS:
        for v in VARIANTS:
            ts = results["portfolios"][p][v]["trade_stats"]
            lines.append(
                f"| {p} | {v} | {ts['avg_round_trips_per_pair']:.2f} | "
                f"{ts['avg_holding_duration_days']:.1f} | {ts['pct_pairs_never_opened']:.1%} |"
            )

    fal = results["falsifications"]
    fr = fal["factor_regression"]
    ls = fal["long_short_decomposition"]
    bs = fal["decile_bootstrap"]
    lines += [
        "",
        "## Falsifications (primary portfolio only: " + fal["primary_portfolio"] + ")",
        "",
        "### Factor regression (5-factor excess return alpha)",
        "",
        f"alpha = {fr['alpha']:.4%}/month (t = {fr['alpha_t']:.2f}), R² = {fr['r_squared']:.3f}, "
        f"n = {fr['n_obs']} months.",
        "",
        "| factor | loading | t (NW) |",
        "|---|---|---|",
    ]
    for c in FACTOR_COLS:
        lines.append(f"| {c} | {fr['loadings'][c]:.3f} | {fr['loadings_t'][c]:.2f} |")

    lines += [
        "",
        "### Long/short leg decomposition",
        "",
        f"Long leg alpha = {ls['long']['alpha']:.4%}/month (t = {ls['long']['alpha_t']:.2f})",
        f"Short leg alpha = {ls['short']['alpha']:.4%}/month (t = {ls['short']['alpha_t']:.2f})",
        "",
        "### Decile-matched random-pairs bootstrap",
        "",
        f"{bs['n_reps']} replications. Real primary portfolio mean monthly return: "
        f"{bs['real_primary_mean_monthly']:.4%}.",
        f"Bootstrap replicate means: mean = {bs['mean_of_rep_means']:.4%}, "
        f"std = {bs['std_of_rep_means']:.4%}, "
        f"[5th, 95th] percentile = [{bs['pct_5']:.4%}, {bs['pct_95']:.4%}], "
        f"range = [{bs['min']:.4%}, {bs['max']:.4%}].",
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

    with open(OUT_DIR / "gate1_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {OUT_DIR / 'gate1_report.md'}")


def _plot_equity_curve(combined_monthly: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for p, label in zip(PORTFOLIOS, ("top-5", "top-20", "control (101-120)")):
        monthly = combined_monthly[(p, PRIMARY_VARIANT, PRIMARY_CAPITAL)]
        wealth = (1 + monthly).cumprod()
        ax.plot(wealth.index, wealth.to_numpy(), label=label)
    ax.set_title(f"Gate 1 replication, {PRIMARY_VARIANT} / {PRIMARY_CAPITAL} capital, 2003-2009")
    ax.set_ylabel("cumulative wealth (start = 1.0)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate1_equity_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'gate1_equity_curve.png'}")


if __name__ == "__main__":
    main()
