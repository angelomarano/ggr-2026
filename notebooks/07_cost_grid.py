"""
07_cost_grid.py -- PROTOCOL.md §5 transaction-cost curve, Level 2 only
(Levels 0/1 are same-day-gross and wait-one-day-gross, both already
computed in gate1_results.json/gate2_results.json -- reused here, never
recomputed).

Reads the per-pair trade logs persisted by notebooks/06_regenerate_trade_logs.py
(results/replication/trade_log_gate1.json, results/frozen/trade_log_gate2.json)
-- NOT the trading engine: src/trading.py's simulate_pair_* is never called
here. For each of the 3 already-frozen windows (Gate 1 replication
2003-2009/golden set, Gate 2 OOS 2010-2026/full_universe,
Gate 2 OOS 2010-2026/golden_set_robustness) and each level c in
config.COST_GRID_BP_PER_SIDE = (0, 5, 10, 20, 40) bp per side:
  1. src/costs.apply_cost_grid subtracts 4*c from every pair's daily_payoff
     on each round trip's closing day (PROTOCOL.md §5).
  2. src/returns.aggregate_portfolio_run (UNCHANGED, reused as-is) turns
     the cost-adjusted pair payoffs back into a committed-capital daily
     return series, exactly like every other portfolio in this project.
  3. compound_to_monthly + combine_overlapping_portfolios (also unchanged)
     produce one monthly series per window per cost level.
  4. Newey-West mean/t-stat and annualized Sharpe, same as everywhere else.
Break-even cost c* (PROTOCOL.md §5) via src/costs.breakeven_cost_bp,
linear interpolation on the 5-point grid; reported as None (not forced) if
the mean return never crosses zero within the grid.

Short costs (PROTOCOL.md §5, "scenario qualitativo +25bp/anno di borrow su
general collateral, nomi hard-to-borrow come limite"): NOT modeled
quantitatively anywhere in this script -- a text note in the generated
report states this explicitly, per the protocol's own framing of it as a
qualitative limitation, not a number to compute.

Usage: python notebooks/07_cost_grid.py
Reads: results/replication/trade_log_gate1.json, results/frozen/trade_log_gate2.json,
       results/replication/gate1_results.json, results/frozen/gate2_results.json
       (Level 0/1 reference numbers only, read not recomputed)
Outputs:
  results/replication/cost_curve.json
  results/replication/cost_curve.md
  results/figures/fig4_cost_curve.png   (2 panels: mean return vs c, Sharpe vs c;
                                          one line per window, dashed horizontal
                                          reference per window for same-day gross)
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.costs import apply_cost_grid, breakeven_cost_bp
from src.inference import newey_west_mean_test
from src.diagnostics import annualized_sharpe
from src.returns import aggregate_portfolio_run, combine_overlapping_portfolios, compound_to_monthly

TRADE_LOG_GATE1_JSON = Path("results/replication/trade_log_gate1.json")
TRADE_LOG_GATE2_JSON = Path("results/frozen/trade_log_gate2.json")
GATE1_RESULTS_JSON = Path("results/replication/gate1_results.json")
GATE2_RESULTS_JSON = Path("results/frozen/gate2_results.json")
OUT_JSON = Path("results/replication/cost_curve.json")
OUT_MD = Path("results/replication/cost_curve.md")
OUT_FIG = Path("results/figures/fig4_cost_curve.png")

COST_GRID_BP = config.COST_GRID_BP_PER_SIDE  # reused as-is, not redefined

WINDOWS = ("gate1", "gate2_full_universe", "gate2_golden_set_robustness")
WINDOW_LABEL = {
    "gate1": "Gate 1 (2003-2009, golden set)",
    "gate2_full_universe": "Gate 2 (2010-2026, full universe)",
    "gate2_golden_set_robustness": "Gate 2 (2010-2026, golden set robustness)",
}


def _json_safe(obj):
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


def _combo_stats(monthly_returns: pd.Series) -> dict:
    if len(monthly_returns) == 0:
        return {"mean_monthly": None, "t_stat_nw": None, "annualized_sharpe": None, "n_months": 0}
    nw = newey_west_mean_test(monthly_returns.to_numpy(), lags=config.NW_LAGS)
    return {
        "mean_monthly": nw["mean"],
        "t_stat_nw": nw["t_stat"],
        "annualized_sharpe": annualized_sharpe(monthly_returns),
        "n_months": nw["n"],
    }


def _cost_curve_for_window(runs: dict) -> dict:
    """runs: {run_id: {"trading_days": [...], "pairs": {pair_id: {..., trades, daily_payoff}}}}
    (one window's worth of trade_log_by_run, as persisted by
    notebooks/06_regenerate_trade_logs.py). Returns {c_bp: combo_stats}
    for every level in COST_GRID_BP."""
    # Trading days per run, parsed once (reused across all 5 cost levels).
    trading_days_by_run = {
        run_id: pd.DatetimeIndex([pd.Timestamp(d) for d in data["trading_days"]])
        for run_id, data in runs.items()
    }

    # aggregate_portfolio_run (reused unmodified) also reads daily_long_payoff/
    # daily_short_payoff for the long/short decomposition -- notebooks/06 didn't
    # persist those (not needed for this analysis: committed_return/employed_return
    # depend only on daily_payoff and trades, verified against src/returns.py's own
    # source). Zero-filled placeholders here are inert for every value this script
    # actually uses; only the (unused) long/short columns of aggregate_portfolio_run's
    # output would be wrong, and this script never reads them.
    pair_results_by_run = {}
    for run_id, data in runs.items():
        n_days = len(data["trading_days"])
        pair_results_by_run[run_id] = {
            pid: {**p, "daily_long_payoff": np.zeros(n_days), "daily_short_payoff": np.zeros(n_days)}
            for pid, p in data["pairs"].items()
        }

    stats_by_cost = {}
    for c_bp in COST_GRID_BP:
        monthly_by_run = {}
        for run_id, data in runs.items():
            pair_results = pair_results_by_run[run_id]
            n_days = len(data["trading_days"])
            cost_adjusted = apply_cost_grid(pair_results, cost_bp_per_side=c_bp)
            agg = aggregate_portfolio_run(cost_adjusted, n_days=n_days, n_selected=config.TOP_PAIRS)
            monthly_by_run[run_id] = compound_to_monthly(agg["committed_return"], trading_days_by_run[run_id])
        combined = combine_overlapping_portfolios(monthly_by_run)
        stats_by_cost[c_bp] = _combo_stats(combined)

    return stats_by_cost


def main():
    with open(TRADE_LOG_GATE1_JSON) as f:
        gate1_log = json.load(f)
    with open(TRADE_LOG_GATE2_JSON) as f:
        gate2_log = json.load(f)
    with open(GATE1_RESULTS_JSON) as f:
        gate1_published = json.load(f)
    with open(GATE2_RESULTS_JSON) as f:
        gate2_published = json.load(f)

    runs_by_window = {
        "gate1": gate1_log["runs"],
        "gate2_full_universe": gate2_log["arms"]["full_universe"]["runs"],
        "gate2_golden_set_robustness": gate2_log["arms"]["golden_set_robustness"]["runs"],
    }

    # Level 0 reference (same-day gross), reused verbatim -- not recomputed.
    level0_reference = {
        "gate1": gate1_published["portfolios"]["top_20"]["same_day"]["committed"],
        "gate2_full_universe": gate2_published["arms"]["full_universe"]["portfolios"]["top_20"]["same_day"]["committed"],
        "gate2_golden_set_robustness": gate2_published["arms"]["golden_set_robustness"]["portfolios"]["top_20"]["same_day"]["committed"],
    }
    # Level 1 reference (wait-one-day gross), reused verbatim for cross-check
    # against this script's own c=0bp point (must match, since c=0bp IS
    # wait-one-day gross by construction).
    level1_reference = {
        "gate1": gate1_published["portfolios"]["top_20"]["wait_one_day"]["committed"],
        "gate2_full_universe": gate2_published["arms"]["full_universe"]["portfolios"]["top_20"]["wait_one_day"]["committed"],
        "gate2_golden_set_robustness": gate2_published["arms"]["golden_set_robustness"]["portfolios"]["top_20"]["wait_one_day"]["committed"],
    }

    results = {}
    for window in WINDOWS:
        print(f"Computing cost curve for {window} ({len(runs_by_window[window])} runs)...")
        stats_by_cost = _cost_curve_for_window(runs_by_window[window])
        means = [stats_by_cost[c]["mean_monthly"] for c in COST_GRID_BP]
        breakeven = breakeven_cost_bp(COST_GRID_BP, means)
        results[window] = {
            "cost_grid_bp": list(COST_GRID_BP),
            "stats_by_cost_bp": {str(c): stats_by_cost[c] for c in COST_GRID_BP},
            "breakeven": breakeven,
            "level0_same_day_gross_reference": level0_reference[window],
            "level1_wait_one_day_gross_reference": level1_reference[window],
        }
        c0 = stats_by_cost[COST_GRID_BP[0]]["mean_monthly"]
        ref1 = level1_reference[window]["mean_monthly"]
        print(f"  c=0bp mean_monthly={c0:.6%} vs Level-1 (wait-one-day gross) reference={ref1:.6%} "
              f"(should match exactly: c=0bp cost grid IS wait-one-day gross)")
        print(f"  break-even c* = {breakeven['c_star_bp']}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(_json_safe(results), f, indent=2)
    print(f"Saved {OUT_JSON}")

    _write_report(results)
    _plot_cost_curve(results)


def _write_report(results: dict) -> None:
    lines = [
        "# Transaction cost curve (PROTOCOL.md §5)",
        "",
        "Primary portfolio only: top-20, wait-one-day, committed capital. Level 2 (explicit cost "
        f"grid c in {list(COST_GRID_BP)} bp per side, round-trip cost = 4c, PROTOCOL.md §5) computed "
        "here by applying src/costs.apply_round_trip_cost to the per-pair trade logs persisted by "
        "notebooks/06_regenerate_trade_logs.py -- no re-run of the trading engine. Level 0 (same-day "
        "gross) and Level 1 (wait-one-day gross) are reused verbatim from gate1_results.json/"
        "gate2_results.json, not recomputed; Level 1 is exactly this script's own c=0bp point "
        "(cross-checked below, must match to reused-data precision).",
        "",
        "## Return and Sharpe vs cost, by window",
        "",
    ]

    for window in WINDOWS:
        r = results[window]
        lines += [
            f"### {WINDOW_LABEL[window]}",
            "",
            f"Level 0 (same-day gross, reused): {r['level0_same_day_gross_reference']['mean_monthly']:.4%}/month, "
            f"Sharpe {r['level0_same_day_gross_reference']['annualized_sharpe']:.2f}.",
            f"Level 1 (wait-one-day gross, reused): {r['level1_wait_one_day_gross_reference']['mean_monthly']:.4%}/month, "
            f"Sharpe {r['level1_wait_one_day_gross_reference']['annualized_sharpe']:.2f}.",
            "",
            "| c (bp/side) | mean/month | t (NW) | ann. Sharpe | n months |",
            "|---|---|---|---|---|",
        ]
        for c in COST_GRID_BP:
            s = r["stats_by_cost_bp"][str(c)]
            lines.append(
                f"| {c} | {s['mean_monthly']:.4%} | {s['t_stat_nw']:.2f} | "
                f"{s['annualized_sharpe']:.2f} | {s['n_months']} |"
            )
        be = r["breakeven"]
        c_star_txt = f"{be['c_star_bp']:.1f} bp/side" if be["c_star_bp"] is not None else "not reached within the grid"
        lines += ["", f"**Break-even cost c\\* = {c_star_txt}.** {be['note']}", ""]

    lines += [
        "## Short costs (PROTOCOL.md §5) -- not modeled quantitatively",
        "",
        "PROTOCOL.md §5 declares short costs as a qualitative scenario only "
        '("+25 bp/anno di borrow su general collateral"; hard-to-borrow names mentioned as a limit, '
        "not a number to compute) -- this is not simulated anywhere in this repository. The cost grid "
        "above covers only the round-trip trading costs on both legs (PROTOCOL.md §5, Level 2); it "
        "does not include any securities-lending/borrow cost for the short leg. A strategy shown "
        "profitable above a given c on this grid could still be unprofitable once realistic borrow "
        "costs are added, particularly for small/illiquid names that are more likely to be "
        "hard-to-borrow than the large/mid-cap S&P 500 constituents this project trades. This is "
        "stated here as an explicit limitation, not quantified.",
        "",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")


def _plot_cost_curve(results: dict) -> None:
    colors = {"gate1": "tab:blue", "gate2_full_universe": "tab:orange", "gate2_golden_set_robustness": "tab:green"}
    fig, (ax_ret, ax_sharpe) = plt.subplots(1, 2, figsize=(13, 5))

    for window in WINDOWS:
        r = results[window]
        color = colors[window]
        means = [r["stats_by_cost_bp"][str(c)]["mean_monthly"] for c in COST_GRID_BP]
        sharpes = [r["stats_by_cost_bp"][str(c)]["annualized_sharpe"] for c in COST_GRID_BP]
        label = WINDOW_LABEL[window]

        ax_ret.plot(COST_GRID_BP, means, marker="o", color=color, label=label)
        ax_ret.axhline(r["level0_same_day_gross_reference"]["mean_monthly"], color=color, linestyle="--", linewidth=1, alpha=0.6)

        ax_sharpe.plot(COST_GRID_BP, sharpes, marker="o", color=color, label=label)
        ax_sharpe.axhline(r["level0_same_day_gross_reference"]["annualized_sharpe"], color=color, linestyle="--", linewidth=1, alpha=0.6)

    ax_ret.axhline(0, color="black", linewidth=0.8)
    ax_ret.set_xlabel("cost c (bp per side)")
    ax_ret.set_ylabel("mean monthly return (committed capital)")
    ax_ret.set_title("Return vs transaction cost")
    ax_ret.grid(alpha=0.3)
    ax_ret.legend(fontsize=8)

    ax_sharpe.axhline(0, color="black", linewidth=0.8)
    ax_sharpe.set_xlabel("cost c (bp per side)")
    ax_sharpe.set_ylabel("annualized Sharpe")
    ax_sharpe.set_title("Sharpe vs transaction cost")
    ax_sharpe.grid(alpha=0.3)

    fig.suptitle("Top-20/wait-one-day/committed: Level 2 cost grid (solid) vs Level 0 same-day gross (dashed)")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_FIG}")


if __name__ == "__main__":
    main()
