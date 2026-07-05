"""
03b_gate2_refit_full_universe.py -- SECOND execution of Gate 2's
full_universe arm only, after fixing two post-hoc data-quality filters in
src/formation.py (config.MAX_ABS_DAILY_RETURN, config.MAX_CONSECUTIVE_FROZEN_DAYS
- see DEVIATIONS.md). This reuses run_arm() and the plotting functions from
notebooks/03_gate2_frozen_run.py unmodified - it is not a parallel copy of
that script's logic.

This is an explicitly logged re-run, not a silent overwrite: the pre-fix
full_universe results from the first execution are preserved under
"full_universe_before_fix" in the output, alongside the corrected
"full_universe" results, for direct before/after comparison.

golden_set_robustness is NOT re-run here: none of the corrupted tickers
belong to the golden set (verified separately), so its first-execution
results are carried over unchanged.

Usage: python notebooks/03b_gate2_refit_full_universe.py
Reads: results/frozen/gate2_results.json (the first execution's output)
Outputs (results/frozen/), replacing the first execution's files:
  gate2_results.json    full_universe (post-fix), full_universe_before_fix
                          (pre-fix, preserved), golden_set_robustness (unchanged)
  gate2_report.md         base report regenerated from the post-fix numbers,
                          with an appended before/after section
  gate2_equity_curve.png, gate2_h3_rolling_corr.png, gate2_h4_delta.png
                          regenerated from the post-fix full_universe arm
                          (h4 plot also uses the unchanged golden_set_robustness)
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import importlib
import json
import time

import config
from data.factors import load_factors
from data.universe import formation_calendar, load_membership

gate2 = importlib.import_module("notebooks.03_gate2_frozen_run")

OUT_DIR = gate2.OUT_DIR
RESULTS_JSON = OUT_DIR / "gate2_results.json"

SECOND_EXECUTION_NOTE = (
    "full_universe re-run after fixing two post-hoc data-quality filters in "
    "src/formation.py (config.MAX_ABS_DAILY_RETURN, config.MAX_CONSECUTIVE_FROZEN_DAYS); "
    "see DEVIATIONS.md. golden_set_robustness is unchanged from the first execution "
    "(none of the corrupted tickers belong to the golden set)."
)


def main():
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(f"{RESULTS_JSON} not found - run notebooks/03_gate2_frozen_run.py first")

    with open(RESULTS_JSON) as f:
        previous = json.load(f)

    full_universe_before_fix = previous["arms"]["full_universe"]
    golden_set_robustness = previous["arms"]["golden_set_robustness"]

    print("=" * 90)
    print("SECOND EXECUTION: full_universe arm only.")
    print(SECOND_EXECUTION_NOTE)
    print("=" * 90)

    t_start = time.time()
    membership = load_membership(config.CONSTITUENTS_CSV)
    factors_df = load_factors()
    vix_monthly = gate2._load_vix_monthly()
    cal = formation_calendar(config.OOS_TRADING_START_FIRST, config.OOS_TRADING_START_LAST)
    gate1_h4_ref = gate2._load_gate1_h4_reference()

    full_universe_after_fix = gate2.run_arm(
        "full_universe", None, membership, factors_df, vix_monthly, cal, gate1_h4_ref
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    serializable = {
        "golden_set_oos_size": previous["golden_set_oos_size"],
        "second_execution_note": SECOND_EXECUTION_NOTE,
        "arms": {
            "full_universe": {k: v for k, v in full_universe_after_fix.items() if k != "_combined_monthly"},
            "full_universe_before_fix": full_universe_before_fix,
            "golden_set_robustness": golden_set_robustness,
        },
    }
    with open(OUT_DIR / "gate2_results.json", "w") as f:
        json.dump(gate2._json_safe(serializable), f, indent=2)
    print(f"\nSaved {OUT_DIR / 'gate2_results.json'} "
          "(full_universe post-fix + full_universe_before_fix + golden_set_robustness)")

    _write_markdown_report(serializable)

    gate2._plot_equity_curve(full_universe_after_fix["_combined_monthly"])
    gate2._plot_h3_rolling_corr(full_universe_after_fix["h3_rolling_correlation"])
    h4_plot_view = {
        "arms": {
            "full_universe": serializable["arms"]["full_universe"],
            "golden_set_robustness": serializable["arms"]["golden_set_robustness"],
        }
    }
    gate2._plot_h4_delta(h4_plot_view)

    print(f"\nDone in {time.time() - t_start:.1f}s total.")


def _write_markdown_report(serializable: dict) -> None:
    # Reuse the existing 2-arm report writer for the main body (post-fix
    # full_universe + unchanged golden_set_robustness): same format as a
    # normal single execution would produce.
    standard_view = {
        "golden_set_oos_size": serializable["golden_set_oos_size"],
        "arms": {
            "full_universe": serializable["arms"]["full_universe"],
            "golden_set_robustness": serializable["arms"]["golden_set_robustness"],
        },
    }
    gate2._write_markdown_report(standard_view)

    before = serializable["arms"]["full_universe_before_fix"]
    after = serializable["arms"]["full_universe"]

    lines = [
        "",
        "---",
        "",
        "## Second execution: full_universe re-run after data-quality fix",
        "",
        serializable["second_execution_note"],
        "",
        "This is a SECOND, explicitly logged execution of the full_universe arm "
        "only - not a silent overwrite. The pre-fix numbers are preserved below "
        "for direct comparison.",
        "",
        "### Descriptive statistics, full_universe: before vs after fix",
        "",
        "| portfolio | variant | capital | before/after | mean/month | t (NW) | ann. Sharpe | max drawdown |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in gate2.PORTFOLIOS:
        for v in gate2.VARIANTS:
            for capital in ("committed", "employed"):
                for label, arm in (("before_fix", before), ("after_fix", after)):
                    s = arm["portfolios"][p][v][capital]
                    lines.append(
                        f"| {p} | {v} | {capital} | {label} | {s['mean_monthly']:.4%} | "
                        f"{s['t_stat_nw']:.2f} | {s['annualized_sharpe']:.2f} | {s['max_drawdown']:.1%} |"
                    )

    lines += [
        "",
        "### Decile-matched bootstrap (primary portfolio), before vs after fix",
        "",
        "Before the fix, a severely corrupted ticker (CBE, wild price swings between "
        "$0.005 and $170 with zero-volume stale quotes) was drawn as a random decile-matched "
        "substitute in the bootstrap, producing astronomical (numerically meaningless) "
        "replicate statistics.",
        "",
    ]
    for label, arm in (("before_fix", before), ("after_fix", after)):
        bs = arm["falsifications"]["decile_bootstrap"]
        lines.append(
            f"- {label}: real mean = {bs['real_primary_mean_monthly']:.4%}; bootstrap replicate means: "
            f"mean = {bs['mean_of_rep_means']:.4%}, std = {bs['std_of_rep_means']:.4%}, "
            f"[5th, 95th] pct = [{bs['pct_5']:.4%}, {bs['pct_95']:.4%}], "
            f"range = [{bs['min']:.4%}, {bs['max']:.4%}]."
        )

    lines += [
        "",
        "### Confirmed real-selection contamination cases and their resolution",
        "",
        "BMC (long frozen/stale-price runs, not extreme jumps) was selected as a real pair "
        "member - not just a possible bootstrap substitute - in 2 runs before the fix:",
        "- 2014-01, control rank 117: BMC/BXP.",
        "- 2014-04, top_20 ranks 10 and 17: BMC/MCD, BMC/BMS.",
        "- 2014-04, control ranks 112 and 118: BMC/KO, BMC/CNP.",
        "",
        "After the fix, BMC is excluded from formation in both runs "
        "(frozen-price filter, not the extreme-return filter), and every rank shifts up "
        "to the next legitimately-ranked pair - no other corrupted ticker enters in its place.",
        "",
    ]

    with open(gate2.OUT_DIR / "gate2_report.md", "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Appended before/after comparison to {gate2.OUT_DIR / 'gate2_report.md'}")


if __name__ == "__main__":
    main()
