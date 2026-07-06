"""
04_readme_figures.py -- generates the 3 figures referenced in README.md.

Read-only with respect to the frozen Gate 1 / Gate 2 results: figures 2 (H3
rolling correlation) and 3 (H4 same-day/wait-one-day delta) are built
directly from the already-saved results/replication/gate1_results.json and
results/frozen/gate2_results.json - no re-simulation, no new numbers.

Figure 1 (concatenated equity curve) needs the primary portfolio's monthly
return series, which was not persisted to JSON (only aggregate stats were).
It is re-derived here using the exact same deterministic formation/trading/
returns code already used by notebooks/02_gate1_replication.py and
notebooks/03_gate2_frozen_run.py - same universe definitions, same
parameters, same formulas - restricted to the top-20/wait-one-day/committed
portfolio only (the other portfolios/variants and the decile bootstrap are
skipped here since they are not needed for this figure and are already
computed and saved elsewhere).

Usage: python notebooks/04_readme_figures.py
Outputs (results/figures/):
  fig1_equity_curve_2003_2026.png
  fig2_h3_rolling_correlation.png
  fig3_h4_delta_by_subperiod.png
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import importlib
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from data.prices import RAW as PRICES_DIR
from data.universe import formation_calendar, load_membership, universe_for_run
from src.formation import load_formation_returns, load_trading_returns, normalized_price_indices, select_portfolios
from src.returns import aggregate_portfolio_run, combine_overlapping_portfolios, compound_to_monthly
from src.trading import simulate_pair_wait_one_day

gate1_mod = importlib.import_module("notebooks.02_gate1_replication")
gate2_mod = importlib.import_module("notebooks.03_gate2_frozen_run")

OUT_DIR = Path("results/figures")
GATE1_JSON = Path("results/replication/gate1_results.json")
GATE2_JSON = Path("results/frozen/gate2_results.json")
GOLDEN_SET_CSV = Path("results/replication/golden_set.csv")

warnings.filterwarnings("ignore", category=UserWarning)


def _primary_monthly_series(trading_start_first, trading_start_last, universe_filter):
    """Top-20/wait-one-day/committed monthly return series only, combined
    across the overlapping monthly runs - same formulas as
    aggregate_portfolio_run/compound_to_monthly/combine_overlapping_portfolios
    already used elsewhere, restricted to the one portfolio/variant needed
    for the equity curve figure."""
    membership = load_membership(config.CONSTITUENTS_CSV)
    cal = formation_calendar(trading_start_first, trading_start_last)

    monthly_by_run = {}
    for run_id, r in cal.iterrows():
        try:
            universe = set(universe_for_run(membership, r.formation_start))
            if universe_filter is not None:
                universe = universe & universe_filter
            universe = sorted(universe)

            formation_days, trading_days = gate2_mod._formation_and_trading_windows(
                r.trading_start, r.trading_end_approx
            )
            formation_returns = load_formation_returns(universe, formation_days[0], formation_days[-1], PRICES_DIR)
            price_index = normalized_price_indices(formation_returns)
            table = select_portfolios(price_index)["top_20"]
            trading_returns = load_trading_returns(universe, trading_days[0], trading_days[-1], PRICES_DIR)

            pair_results = {}
            for rank, row in table.iterrows():
                t1, t2 = row["ticker_1"], row["ticker_2"]
                if t1 not in trading_returns.columns or t2 not in trading_returns.columns:
                    continue
                res = simulate_pair_wait_one_day(
                    trading_returns[t1].to_numpy(), trading_returns[t2].to_numpy(),
                    sigma=row["sigma"], k=config.OPEN_TRIGGER_SIGMAS,
                )
                pair_results[f"{t1}_{t2}_{rank}"] = res
            agg = aggregate_portfolio_run(pair_results, n_days=len(trading_days), n_selected=config.TOP_PAIRS)
            monthly_by_run[run_id] = compound_to_monthly(agg["committed_return"], trading_days)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {run_id}: {type(e).__name__}: {e}")
            continue
        print(f"  ...{run_id} done", flush=True)

    return combine_overlapping_portfolios(monthly_by_run)


def make_figure1():
    print("Figure 1: re-deriving primary portfolio monthly series (Gate 1 golden set + Gate 2 full universe)...")
    golden_set = set(pd.read_csv(GOLDEN_SET_CSV)["ticker"])

    print("Gate 1 window (golden set)...")
    gate1_monthly = _primary_monthly_series(
        config.REPLICATION_TRADING_START_FIRST, config.REPLICATION_TRADING_START_LAST, golden_set
    )
    print("Gate 2 window (full universe)...")
    gate2_monthly = _primary_monthly_series(
        config.OOS_TRADING_START_FIRST, config.OOS_TRADING_START_LAST, None
    )

    combined = pd.concat([gate1_monthly, gate2_monthly]).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    # Cross-check against the saved aggregate stats (sanity, not a new number):
    with open(GATE1_JSON) as f:
        g1 = json.load(f)
    with open(GATE2_JSON) as f:
        g2 = json.load(f)
    g1_mean_saved = g1["portfolios"]["top_20"]["wait_one_day"]["committed"]["mean_monthly"]
    g2_mean_saved = g2["arms"]["full_universe"]["portfolios"]["top_20"]["wait_one_day"]["committed"]["mean_monthly"]
    print(f"  sanity check: gate1 mean recomputed={gate1_monthly.mean():.6%} vs saved={g1_mean_saved:.6%}")
    print(f"  sanity check: gate2 mean recomputed={gate2_monthly.mean():.6%} vs saved={g2_mean_saved:.6%}")

    # Build a continuous-within-window, broken-across-the-gap wealth curve:
    full_index = pd.date_range(combined.index.min(), combined.index.max(), freq="MS")
    r = combined.reindex(full_index)
    wealth = (1 + r.fillna(0)).cumprod()
    wealth[r.isna()] = np.nan  # break the line where there is no data (the ~6-month gap)

    vix_monthly = gate2_mod._load_vix_monthly().reindex(full_index)
    high_vol = vix_monthly >= config.VIX_HIGH_THRESHOLD

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(wealth.index, wealth.to_numpy(), color="tab:blue", linewidth=1.3)

    # Shade high-vol months
    in_band = False
    band_start = None
    for dt, flag in high_vol.items():
        flag = bool(flag) if pd.notna(flag) else False
        if flag and not in_band:
            in_band, band_start = True, dt
        elif not flag and in_band:
            ax.axvspan(band_start, dt, color="tab:red", alpha=0.12, lw=0)
            in_band = False
    if in_band:
        ax.axvspan(band_start, high_vol.index[-1], color="tab:red", alpha=0.12, lw=0)

    for name, (start, end) in config.EVENT_WINDOWS.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="tab:orange", alpha=0.25, lw=0)
        ax.text(pd.Timestamp(start), ax.get_ylim()[1] * 0.995, name, fontsize=8, rotation=90,
                va="top", ha="left", color="tab:orange")

    ax.axvspan(pd.Timestamp("2003-01-01"), pd.Timestamp("2009-06-30"), color="gray", alpha=0.05, lw=0)
    ax.axvspan(pd.Timestamp("2010-01-01"), pd.Timestamp("2026-06-30"), color="gray", alpha=0.0, lw=0)
    ax.text(pd.Timestamp("2004-06-01"), ax.get_ylim()[0], "Gate 1 (replication, golden set)",
            fontsize=8, color="dimgray", va="bottom")
    ax.text(pd.Timestamp("2015-06-01"), ax.get_ylim()[0], "Gate 2 (OOS, full universe)",
            fontsize=8, color="dimgray", va="bottom")

    ax.set_title("Primary portfolio (top-20 / wait-one-day / committed), cumulative wealth\n"
                  "red shading = VIX>=25 months, orange = declared event windows, gap = untested 2009-2010")
    ax.set_ylabel("cumulative wealth (start = 1.0)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig1_equity_curve_2003_2026.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'fig1_equity_curve_2003_2026.png'}")


def make_figure2():
    print("\nFigure 2: H3 rolling correlation (from saved JSON, no re-simulation)...")
    with open(GATE2_JSON) as f:
        g2 = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 5))
    styles = {
        ("full_universe", "raw"): dict(color="tab:blue", linestyle="-", label="full universe, raw"),
        ("full_universe", "residual"): dict(color="tab:blue", linestyle="--", label="full universe, 5-factor residual"),
        ("golden_set_robustness", "raw"): dict(color="tab:green", linestyle="-", label="golden set robustness, raw"),
        ("golden_set_robustness", "residual"): dict(color="tab:green", linestyle="--", label="golden set robustness, 5-factor residual"),
    }
    for arm in ("full_universe", "golden_set_robustness"):
        h3 = g2["arms"][arm]["h3_rolling_correlation"]
        for kind, key in (("raw", "raw_correlation_series"), ("residual", "residual_correlation_series")):
            s = pd.Series({pd.Timestamp(k): v for k, v in h3[key].items()}).sort_index()
            ax.plot(s.index, s.to_numpy(), **styles[(arm, kind)])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("H3: rolling 24-month correlation, top-20 vs control (101-120), Gate 2 (OOS, 2010-2026)")
    ax.set_ylabel("correlation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig2_h3_rolling_correlation.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'fig2_h3_rolling_correlation.png'}")


def make_figure3():
    print("\nFigure 3: H4 same-day - wait-one-day delta by subperiod (from saved JSON, no re-simulation)...")
    with open(GATE2_JSON) as f:
        g2 = json.load(f)

    subperiods = ["2003_2009_gate1_reused", "2010_2017", "2018_2026"]
    labels = ["2003-2009\n(Gate 1)", "2010-2017", "2018-2026"]
    arms = ["full_universe", "golden_set_robustness"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.35
    x = np.arange(len(subperiods))
    for i, arm in enumerate(arms):
        h4 = g2["arms"][arm]["h4_same_day_vs_wait_one_day"]["subperiods"]
        values = [h4[name]["mean_delta"] for name in subperiods]
        errors = [h4[name]["bootstrap_se"] for name in subperiods]
        has_se = [e is not None for e in errors]
        errors_plot = [e if e is not None else 0.0 for e in errors]

        bar_positions = x + (i - 0.5) * width
        colors = ["tab:blue" if arm == "full_universe" else "tab:green"] * len(subperiods)
        hatches = [None if has else "//" for has in has_se]
        for xpos, val, err, hatch, se_avail in zip(bar_positions, values, errors_plot, hatches, has_se):
            ax.bar(
                xpos, val, width,
                yerr=(err if se_avail else None), capsize=3,
                color=colors[0], alpha=(1.0 if se_avail else 0.5),
                hatch=hatch, edgecolor="black" if hatch else None,
                label=arm if xpos == bar_positions[0] else None,
            )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean monthly delta, same-day minus wait-one-day (top-20, committed)")
    ax.set_title("H4: bid-ask bounce proxy by subperiod\n"
                  "hatched/lighter bar = no paired-bootstrap SE available (Gate 1 series not persisted)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="tab:blue", label="full universe"),
        plt.Rectangle((0, 0), 1, 1, color="tab:green", label="golden set robustness"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="//", label="no SE available"),
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig3_h4_delta_by_subperiod.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR / 'fig3_h4_delta_by_subperiod.png'}")


if __name__ == "__main__":
    make_figure1()
    make_figure2()
    make_figure3()
