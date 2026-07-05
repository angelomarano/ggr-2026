"""
03_gate2_frozen_run.py -- Gate 2 frozen run (PROTOCOL.md §1.1, §3).

Runs the GGR pairs-trading pipeline over the OOS window (192 monthly
trading-period starts, 2010-01 through 2025-12, closing through mid-2026).

This runs the ENTIRE analysis TWICE in the same execution, through the same
code path (run_arm), with only the universe definition changed:
  - "full_universe": the full point-in-time S&P 500 membership each month
    (PRIMARY basis, per PROTOCOL.md/DEVIATIONS.md).
  - "golden_set_robustness": point-in-time membership intersected with the
    OOS golden set (results/frozen/golden_set_oos.csv, tickers with a
    complete Yahoo history in every single OOS formation window) -- an
    explicit, pre-registered robustness check, not a second attempt made
    after seeing results.

For each arm, for each of {top-5, top-20, control(101-120)} portfolios and
each of {same-day, wait-one-day} execution variants: committed/employed
monthly returns, descriptive statistics, and trade statistics (12
combinations). Falsification tests (factor regression, long/short
decomposition, decile-matched bootstrap) run on the primary portfolio only
(top-20, wait-one-day, committed).

Additionally, per arm:
  H2 - VIX regime regression (primary return ~ a + b*HighVol) and cumulative
       return in the two declared event windows, with block-bootstrap CI.
  H3 - rolling 24-month correlation between top-20 and control, on raw
       returns and on 5-factor regression residuals.
  H4 - same-day minus wait-one-day delta on top-20 committed capital, for
       the 2010-2017 and 2018-2026 subperiods, with paired-bootstrap SE.
       The 2003-2009 number is REUSED from results/replication/gate1_results.json,
       not recomputed.

This script computes and reports numbers. It does NOT compare them against
any acceptance band and does NOT judge whether any hypothesis is confirmed.

ONE-SHOT RULE: this is meant to be run once on this frozen window. If a run
fails partway through, the script does not retry or patch itself -- it
records the failure and moves on to the next run, and the exact failure is
reported in the output. Any actual re-execution of this script (e.g. after
a real code fix) must be logged explicitly, not silently repeated.

Usage: python notebooks/03_gate2_frozen_run.py
Outputs (results/frozen/):
  gate2_results.json         machine-readable, every number computed, both arms
  gate2_report.md             same numbers as markdown tables, both arms
  gate2_equity_curve.png      cumulative wealth, wait-one-day/committed, full-universe arm
  gate2_h3_rolling_corr.png   rolling 24m correlation top-20 vs control, full-universe arm
  gate2_h4_delta.png          same-day - wait-one-day delta by subperiod, both arms
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
import statsmodels.api as sm

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
    stationary_bootstrap_ci,
)
from src.returns import aggregate_portfolio_run, combine_overlapping_portfolios, compound_to_monthly
from src.trading import simulate_pair_same_day, simulate_pair_wait_one_day

OUT_DIR = Path("results/frozen")
GOLDEN_SET_OOS_CSV = OUT_DIR / "golden_set_oos.csv"
GATE1_RESULTS_JSON = Path("results/replication/gate1_results.json")

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
H3_ROLLING_WINDOW_MONTHS = 24
H4_SUBPERIODS = {"2010_2017": ("2010-01", "2017-12"), "2018_2026": ("2018-01", "2026-12")}
ARMS = ("full_universe", "golden_set_robustness")


# --------------------------------------------------------------- shared helpers (same as Gate 1)

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


# --------------------------------------------------------------- new helpers for H2/H3

def _load_vix_monthly() -> pd.Series:
    """Monthly average VIX close, from the already-cached _idx_VIX.parquet
    (no new download)."""
    vix = pd.read_parquet(PRICES_DIR / "_idx_VIX.parquet", columns=["Close"])
    monthly = vix["Close"].resample("MS").mean()
    monthly.index.name = "month"
    return monthly


def _factor_residuals(monthly_returns: pd.Series, factors_df: pd.DataFrame, factor_cols) -> pd.Series:
    """OLS residuals of monthly_returns on factor_cols (same inner-join
    convention as inference.factor_regression, which does not itself
    expose residuals - needed for H3's residual correlation). Not a
    modification of inference.py: a separate, new computation here."""
    aligned = pd.concat([monthly_returns.rename("y"), factors_df[list(factor_cols)]], axis=1, join="inner").dropna()
    y = aligned["y"].to_numpy()
    X = sm.add_constant(aligned[list(factor_cols)].to_numpy(), has_constant="add")
    model = sm.OLS(y, X).fit()
    return pd.Series(model.resid, index=aligned.index)


def _rolling_correlation(series_a: pd.Series, series_b: pd.Series, window: int) -> pd.Series:
    aligned = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1, join="inner").dropna()
    return aligned["a"].rolling(window).corr(aligned["b"])


def _series_to_json_dict(s: pd.Series) -> dict:
    return {str(pd.Timestamp(k).date()): (None if pd.isna(v) else float(v)) for k, v in s.items()}


# --------------------------------------------------------------- per-run processing (same logic as Gate 1)

def _run_one_run(run_id, r, membership, universe_filter):
    """Process a single monthly run. universe_filter: set of tickers to
    intersect the point-in-time membership with, or None for the full
    point-in-time universe (no filter)."""
    universe = set(universe_for_run(membership, r.formation_start))
    if universe_filter is not None:
        universe = universe & universe_filter
    universe = sorted(universe)

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


def _load_gate1_h4_reference() -> dict | None:
    """Reuses the already-computed 2003-2009 same-day/wait-one-day delta on
    top-20 committed capital from Gate 1's saved output. Does not recompute
    Gate 1. Returns None (with the caller expected to note it) if that file
    isn't there."""
    if not GATE1_RESULTS_JSON.exists():
        return None
    with open(GATE1_RESULTS_JSON) as f:
        gate1 = json.load(f)
    same = gate1["portfolios"]["top_20"]["same_day"]["committed"]
    wait = gate1["portfolios"]["top_20"]["wait_one_day"]["committed"]
    return {
        "source": str(GATE1_RESULTS_JSON),
        "n_months": same["n_months"],
        "mean_same_day": same["mean_monthly"],
        "mean_wait_one_day": wait["mean_monthly"],
        "mean_delta": same["mean_monthly"] - wait["mean_monthly"],
        "bootstrap_se": None,
        "note": (
            "reused from Gate 1's saved aggregate stats (difference of means); "
            "Gate 1 did not persist the paired monthly series, so no paired-bootstrap "
            "SE is available for this subperiod, unlike the two OOS subperiods below."
        ),
    }


# --------------------------------------------------------------- one full arm (run twice from main())

def run_arm(arm_label, universe_filter, membership, factors_df, vix_monthly, cal, gate1_h4_ref):
    """Runs the entire Gate 2 analysis for ONE universe definition
    (universe_filter=None for full_universe, or a ticker set for the golden
    set robustness check). This is the SAME code for both arms - main()
    calls it twice with different arguments, not a duplicated script."""
    print(f"\n{'=' * 90}\nARM: {arm_label}\n{'=' * 90}")
    t_start = time.time()

    daily_by_run: dict[tuple, dict[str, pd.DataFrame]] = {(p, v): {} for p in PORTFOLIOS for v in VARIANTS}
    trading_days_by_run: dict[str, pd.DatetimeIndex] = {}
    all_pair_results: dict[tuple, list] = {(p, v): [] for p in PORTFOLIOS for v in VARIANTS}
    long_short_daily_by_run: dict[str, pd.DataFrame] = {}
    bootstrap_monthly_by_run: dict[str, list] = {}
    universe_size_by_run: dict[str, int] = {}
    run_warnings: dict[str, dict] = {}
    failed_runs: dict[str, str] = {}

    n_runs = len(cal)
    for i, (run_id, r) in enumerate(cal.iterrows()):
        print(f"[{arm_label} {i + 1}/{n_runs}] run {run_id} ...", flush=True)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = _run_one_run(run_id, r, membership, universe_filter)
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

    print(f"[{arm_label}] all runs processed in {time.time() - t_start:.1f}s "
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

    # ---- 12-combination descriptive stats ----
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

    # ---- H2: VIX regime + event windows ----
    high_vol = (vix_monthly >= config.VIX_HIGH_THRESHOLD).astype(float).rename("HighVol")
    vix_df = high_vol.to_frame()
    h2_reg = factor_regression(primary_monthly, vix_df, factor_cols=("HighVol",), lags=config.NW_LAGS)

    event_results = {}
    for name, (start, end) in config.EVENT_WINDOWS.items():
        window_returns = primary_monthly.loc[start:end]
        n_months = len(window_returns)
        if n_months == 0:
            event_results[name] = {"n_months": 0, "note": "no overlap with this arm's monthly series"}
            continue
        cumulative_point = float((1 + window_returns).prod() - 1)
        boot = stationary_bootstrap_ci(window_returns.to_numpy(), seed=config.SEED)
        event_results[name] = {
            "n_months": n_months,
            "cumulative_return": cumulative_point,
            "mean_monthly": boot["mean"],
            "mean_monthly_ci_low": boot["ci_low"],
            "mean_monthly_ci_high": boot["ci_high"],
            "cumulative_return_ci_low_approx": float((1 + boot["ci_low"]) ** n_months - 1),
            "cumulative_return_ci_high_approx": float((1 + boot["ci_high"]) ** n_months - 1),
            "approx_note": "CI bounds compound the bootstrap CI of the mean monthly return over n_months; not a direct bootstrap of the compounded statistic itself.",
        }

    h2 = {
        "vix_high_threshold": config.VIX_HIGH_THRESHOLD,
        "n_high_vol_months": int(high_vol.reindex(primary_monthly.index).sum()),
        "n_months_total": int(len(primary_monthly)),
        "a_intercept": h2_reg["alpha"],
        "b_highvol": h2_reg["loadings"]["HighVol"],
        "t_b_highvol": h2_reg["loadings_t"]["HighVol"],
        "n_obs": h2_reg["n_obs"],
        "event_windows": event_results,
    }

    # ---- H3: rolling 24m correlation top_20 vs control ----
    top20_monthly = combined_monthly[("top_20", PRIMARY_VARIANT, PRIMARY_CAPITAL)]
    control_monthly = combined_monthly[("control", PRIMARY_VARIANT, PRIMARY_CAPITAL)]
    raw_corr = _rolling_correlation(top20_monthly, control_monthly, H3_ROLLING_WINDOW_MONTHS)
    top20_resid = _factor_residuals(top20_monthly, factors_df, FACTOR_COLS)
    control_resid = _factor_residuals(control_monthly, factors_df, FACTOR_COLS)
    resid_corr = _rolling_correlation(top20_resid, control_resid, H3_ROLLING_WINDOW_MONTHS)

    h3 = {
        "rolling_window_months": H3_ROLLING_WINDOW_MONTHS,
        "raw_correlation_mean": float(raw_corr.mean(skipna=True)),
        "residual_correlation_mean": float(resid_corr.mean(skipna=True)),
        "raw_correlation_series": _series_to_json_dict(raw_corr),
        "residual_correlation_series": _series_to_json_dict(resid_corr),
    }

    # ---- H4: same-day - wait-one-day delta on top-20 committed, by subperiod ----
    top20_same = combined_monthly[("top_20", "same_day", "committed")]
    top20_wait = combined_monthly[("top_20", "wait_one_day", "committed")]
    aligned = pd.concat([top20_same.rename("same"), top20_wait.rename("wait")], axis=1, join="inner").dropna()
    delta = aligned["same"] - aligned["wait"]

    h4_subperiods = {}
    for name, (start, end) in H4_SUBPERIODS.items():
        d = delta.loc[start:end]
        if len(d) == 0:
            h4_subperiods[name] = {"n_months": 0, "note": "no overlap with this arm's monthly series"}
            continue
        boot = stationary_bootstrap_ci(d.to_numpy(), seed=config.SEED)
        h4_subperiods[name] = {
            "n_months": len(d),
            "mean_delta": float(d.mean()),
            "bootstrap_se": float(boot["boot_means"].std(ddof=1)),
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
        }
    if gate1_h4_ref is not None:
        h4_subperiods["2003_2009_gate1_reused"] = gate1_h4_ref

    h4 = {"subperiods": h4_subperiods}

    universe_sizes = list(universe_size_by_run.values())
    meta = {
        "arm": arm_label,
        "n_runs": n_runs,
        "n_runs_ok": n_runs - len(failed_runs),
        "trading_start_first": config.OOS_TRADING_START_FIRST,
        "trading_start_last": config.OOS_TRADING_START_LAST,
        "universe_size_per_run": {
            "min": min(universe_sizes) if universe_sizes else None,
            "max": max(universe_sizes) if universe_sizes else None,
            "mean": float(np.mean(universe_sizes)) if universe_sizes else None,
        },
        "failed_runs": failed_runs,
        "runs_with_warnings": run_warnings,
        "elapsed_seconds": time.time() - t_start,
    }

    return {
        "meta": meta,
        "portfolios": portfolio_stats,
        "falsifications": {
            "primary_portfolio": f"{PRIMARY_PORTFOLIO} / {PRIMARY_VARIANT} / {PRIMARY_CAPITAL}",
            "factor_regression": factor_reg_result,
            "long_short_decomposition": long_short_result,
            "decile_bootstrap": bootstrap_summary,
        },
        "h2_vix_regime": h2,
        "h3_rolling_correlation": h3,
        "h4_same_day_vs_wait_one_day": h4,
        "_combined_monthly": combined_monthly,  # kept for plotting only, stripped before JSON dump
    }


def main():
    t_start = time.time()
    membership = load_membership(config.CONSTITUENTS_CSV)
    factors_df = load_factors()
    vix_monthly = _load_vix_monthly()
    cal = formation_calendar(config.OOS_TRADING_START_FIRST, config.OOS_TRADING_START_LAST)
    golden_set_oos = set(pd.read_csv(GOLDEN_SET_OOS_CSV)["ticker"])
    gate1_h4_ref = _load_gate1_h4_reference()

    arm_universe_filters = {"full_universe": None, "golden_set_robustness": golden_set_oos}
    arm_results = {}
    for arm_label in ARMS:
        arm_results[arm_label] = run_arm(
            arm_label, arm_universe_filters[arm_label], membership, factors_df, vix_monthly, cal, gate1_h4_ref
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    serializable = {
        "golden_set_oos_size": len(golden_set_oos),
        "arms": {
            arm_label: {k: v for k, v in res.items() if k != "_combined_monthly"}
            for arm_label, res in arm_results.items()
        },
    }
    with open(OUT_DIR / "gate2_results.json", "w") as f:
        json.dump(_json_safe(serializable), f, indent=2)
    print(f"\nSaved {OUT_DIR / 'gate2_results.json'}")

    _write_markdown_report(serializable)
    _plot_equity_curve(arm_results["full_universe"]["_combined_monthly"])
    _plot_h3_rolling_corr(arm_results["full_universe"]["h3_rolling_correlation"])
    _plot_h4_delta(serializable)

    print(f"\nDone in {time.time() - t_start:.1f}s total.")


def _write_markdown_report(serializable: dict) -> None:
    lines = [
        "# Gate 2 frozen run report",
        "",
        f"OOS window: trading start {config.OOS_TRADING_START_FIRST} to {config.OOS_TRADING_START_LAST} "
        "(last trading period closes ~mid-2026).",
        f"Golden set (OOS) size: {serializable['golden_set_oos_size']} tickers "
        "(results/frozen/golden_set_oos.csv).",
        "",
        "No comparison against any acceptance band is made here; these are the computed numbers only.",
        "",
        "## Descriptive statistics (both arms, adjacent rows per combination)",
        "",
        "| portfolio | variant | capital | universe | mean/month | SE (NW) | t (NW) | ann. Sharpe | % neg months | max drawdown |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in PORTFOLIOS:
        for v in VARIANTS:
            for capital in ("committed", "employed"):
                for arm_label in ARMS:
                    s = serializable["arms"][arm_label]["portfolios"][p][v][capital]
                    lines.append(
                        f"| {p} | {v} | {capital} | {arm_label} | {s['mean_monthly']:.4%} | {s['se_nw']:.4%} | "
                        f"{s['t_stat_nw']:.2f} | {s['annualized_sharpe']:.2f} | "
                        f"{s['pct_negative_months']:.1%} | {s['max_drawdown']:.1%} |"
                    )

    lines += [
        "",
        "## Trade statistics (both arms)",
        "",
        "| portfolio | variant | universe | avg round-trips/pair | avg holding days | % pairs never opened |",
        "|---|---|---|---|---|---|",
    ]
    for p in PORTFOLIOS:
        for v in VARIANTS:
            for arm_label in ARMS:
                ts = serializable["arms"][arm_label]["portfolios"][p][v]["trade_stats"]
                lines.append(
                    f"| {p} | {v} | {arm_label} | {ts['avg_round_trips_per_pair']:.2f} | "
                    f"{ts['avg_holding_duration_days']:.1f} | {ts['pct_pairs_never_opened']:.1%} |"
                )

    lines += ["", "## Falsifications (primary portfolio only, both arms)", ""]
    for arm_label in ARMS:
        fal = serializable["arms"][arm_label]["falsifications"]
        fr = fal["factor_regression"]
        ls = fal["long_short_decomposition"]
        bs = fal["decile_bootstrap"]
        lines += [
            f"### {arm_label}: {fal['primary_portfolio']}",
            "",
            f"Factor regression: alpha = {fr['alpha']:.4%}/month (t = {fr['alpha_t']:.2f}), "
            f"R² = {fr['r_squared']:.3f}, n = {fr['n_obs']} months.",
            "",
            "| factor | loading | t (NW) |",
            "|---|---|---|",
        ]
        for c in FACTOR_COLS:
            lines.append(f"| {c} | {fr['loadings'][c]:.3f} | {fr['loadings_t'][c]:.2f} |")
        lines += [
            "",
            f"Long leg alpha = {ls['long']['alpha']:.4%}/month (t = {ls['long']['alpha_t']:.2f}); "
            f"short leg alpha = {ls['short']['alpha']:.4%}/month (t = {ls['short']['alpha_t']:.2f}).",
            "",
            f"Decile-matched bootstrap ({bs['n_reps']} reps): real mean monthly return = "
            f"{bs['real_primary_mean_monthly']:.4%}; bootstrap replicate means: mean = "
            f"{bs['mean_of_rep_means']:.4%}, std = {bs['std_of_rep_means']:.4%}, "
            f"[5th, 95th] pct = [{bs['pct_5']:.4%}, {bs['pct_95']:.4%}].",
            "",
        ]

    lines += ["## H2 - VIX regime and event windows (both arms)", ""]
    for arm_label in ARMS:
        h2 = serializable["arms"][arm_label]["h2_vix_regime"]
        lines += [
            f"### {arm_label}",
            "",
            f"High-vol threshold: VIX >= {h2['vix_high_threshold']}. "
            f"{h2['n_high_vol_months']}/{h2['n_months_total']} months classified high-vol.",
            f"Regression: primary_return = a + b*HighVol -> a = {h2['a_intercept']:.4%}, "
            f"b = {h2['b_highvol']:.4%}, t(b) = {h2['t_b_highvol']:.2f} (n = {h2['n_obs']}).",
            "",
        ]
        for name, ev in h2["event_windows"].items():
            if ev.get("n_months", 0) == 0:
                lines.append(f"- {name}: {ev.get('note', 'no data')}")
                continue
            lines.append(
                f"- {name} ({ev['n_months']} months): cumulative return = {ev['cumulative_return']:.4%}, "
                f"mean monthly bootstrap CI = [{ev['mean_monthly_ci_low']:.4%}, {ev['mean_monthly_ci_high']:.4%}], "
                f"approx. compounded CI = [{ev['cumulative_return_ci_low_approx']:.4%}, "
                f"{ev['cumulative_return_ci_high_approx']:.4%}]."
            )
        lines.append("")

    lines += ["## H3 - rolling 24-month correlation, top-20 vs control (both arms)", ""]
    for arm_label in ARMS:
        h3 = serializable["arms"][arm_label]["h3_rolling_correlation"]
        lines.append(
            f"- {arm_label}: mean raw correlation = {h3['raw_correlation_mean']:.3f}, "
            f"mean residual (5-factor) correlation = {h3['residual_correlation_mean']:.3f} "
            f"(full series saved in gate2_results.json)."
        )
    lines.append("")

    lines += ["## H4 - same-day minus wait-one-day delta, top-20 committed (both arms)", ""]
    for arm_label in ARMS:
        h4 = serializable["arms"][arm_label]["h4_same_day_vs_wait_one_day"]["subperiods"]
        lines.append(f"### {arm_label}")
        lines.append("")
        for name, d in h4.items():
            if d.get("n_months", 0) == 0:
                lines.append(f"- {name}: {d.get('note', 'no data')}")
                continue
            se_txt = f"{d['bootstrap_se']:.4%}" if d.get("bootstrap_se") is not None else "n/a"
            lines.append(
                f"- {name} ({d['n_months']} months): mean delta = {d['mean_delta']:.4%}, "
                f"bootstrap SE = {se_txt}"
                + (f", CI = [{d['ci_low']:.4%}, {d['ci_high']:.4%}]" if "ci_low" in d else "")
                + (f" -- {d['note']}" if "note" in d else "")
            )
        lines.append("")

    lines += ["## Anomalies", ""]
    for arm_label in ARMS:
        meta = serializable["arms"][arm_label]["meta"]
        lines.append(f"### {arm_label}")
        lines.append(
            f"Universe size per run: min {meta['universe_size_per_run']['min']}, "
            f"max {meta['universe_size_per_run']['max']}, mean {meta['universe_size_per_run']['mean']:.1f}. "
            f"{meta['n_runs_ok']}/{meta['n_runs']} runs completed."
        )
        lines.append(f"Failed runs: {len(meta['failed_runs'])}.")
        for run_id, err in meta["failed_runs"].items():
            lines.append(f"- {run_id}: {err}")
        total_warnings = sum(w["count"] for w in meta["runs_with_warnings"].values())
        lines.append(
            f"Runs with warnings: {len(meta['runs_with_warnings'])}, total warning count: {total_warnings}."
        )
        lines.append("")

    with open(OUT_DIR / "gate2_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {OUT_DIR / 'gate2_report.md'}")


def _plot_equity_curve(combined_monthly: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for p, label in zip(PORTFOLIOS, ("top-5", "top-20", "control (101-120)")):
        monthly = combined_monthly[(p, PRIMARY_VARIANT, PRIMARY_CAPITAL)]
        wealth = (1 + monthly).cumprod()
        ax.plot(wealth.index, wealth.to_numpy(), label=label)
    ax.set_title(f"Gate 2 frozen run (full universe), {PRIMARY_VARIANT} / {PRIMARY_CAPITAL} capital, 2010-2026")
    ax.set_ylabel("cumulative wealth (start = 1.0)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate2_equity_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'gate2_equity_curve.png'}")


def _plot_h3_rolling_corr(h3: dict) -> None:
    raw = pd.Series({pd.Timestamp(k): v for k, v in h3["raw_correlation_series"].items()}).sort_index()
    resid = pd.Series({pd.Timestamp(k): v for k, v in h3["residual_correlation_series"].items()}).sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(raw.index, raw.to_numpy(), label="raw returns")
    ax.plot(resid.index, resid.to_numpy(), label="5-factor residuals")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Gate 2 (full universe): rolling {h3['rolling_window_months']}m correlation, top-20 vs control")
    ax.set_ylabel("correlation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate2_h3_rolling_corr.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'gate2_h3_rolling_corr.png'}")


def _plot_h4_delta(serializable: dict) -> None:
    subperiod_names = ["2003_2009_gate1_reused", "2010_2017", "2018_2026"]
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    x = np.arange(len(subperiod_names))
    for i, arm_label in enumerate(ARMS):
        h4 = serializable["arms"][arm_label]["h4_same_day_vs_wait_one_day"]["subperiods"]
        values = [h4.get(name, {}).get("mean_delta") for name in subperiod_names]
        values = [v if v is not None else 0.0 for v in values]
        errors = [h4.get(name, {}).get("bootstrap_se") or 0.0 for name in subperiod_names]
        ax.bar(x + (i - 0.5) * width, values, width, yerr=errors, label=arm_label, capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(subperiod_names)
    ax.set_ylabel("mean monthly delta (same-day - wait-one-day)")
    ax.set_title("Gate 2: top-20 committed same-day minus wait-one-day, by subperiod")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate2_h4_delta.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'gate2_h4_delta.png'}")


if __name__ == "__main__":
    main()
