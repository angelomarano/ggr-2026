"""
diagnostics.py -- Portfolio-level descriptive statistics (PROTOCOL.md §2.3):
annualized Sharpe, max drawdown, % negative months on a monthly return
series, plus trade-log statistics (average round-trips per pair, average
holding duration, % pairs never opened) computed from the raw simulate_pair_*
outputs across a set of runs.

These are purely descriptive: no comparison against PROTOCOL.md §3's
acceptance band, no pass/fail judgment. That happens elsewhere.
"""
from __future__ import annotations

import numpy as np


def annualized_sharpe(monthly_returns) -> float:
    """
    mean(r) / std(r, ddof=1) * sqrt(12) on a monthly return series.
    ddof=1: sample standard deviation, the usual convention for a Sharpe
    ratio estimated from a finite sample rather than a known population.
    Returns nan if std is exactly zero (a constant series has no
    meaningful ratio) instead of raising a ZeroDivisionError.
    """
    r = np.asarray(monthly_returns, dtype=float)
    std = r.std(ddof=1)
    if std == 0.0:
        return float("nan")
    return float(r.mean() / std * np.sqrt(12))


def max_drawdown(monthly_returns) -> float:
    """
    Largest peak-to-trough decline of the cumulative wealth index built
    from `monthly_returns` (starting wealth = 1.0), expressed as a negative
    fraction (e.g. -0.20 = a 20% drawdown). 0.0 if the series never falls
    below its running peak.
    """
    r = np.asarray(monthly_returns, dtype=float)
    wealth = np.concatenate([[1.0], np.cumprod(1 + r)])
    running_max = np.maximum.accumulate(wealth)
    drawdown = wealth / running_max - 1.0
    return float(drawdown.min())


def pct_negative_months(monthly_returns) -> float:
    """Fraction of months with a strictly negative return."""
    r = np.asarray(monthly_returns, dtype=float)
    if len(r) == 0:
        return 0.0
    return float(np.mean(r < 0))


def trade_statistics(all_pair_results: list[dict]) -> dict:
    """
    Trade-log statistics (PROTOCOL.md §2.3) over a flat list of
    simulate_pair_same_day/simulate_pair_wait_one_day result dicts - one
    entry per (run, pair) combination for a given portfolio and execution
    variant, across every monthly run being reported on.

    avg_round_trips_per_pair: total "close" events / number of pairs.
    avg_holding_duration_days: mean (close day - open day) over every
      completed round trip across all pairs (a same-day open+close, e.g.
      a signal on the very last valid day, counts as duration 0).
    pct_pairs_never_opened: share of pairs with zero "open" events.

    Returns all-zero stats for an empty input (nothing to summarize, not
    an error).
    """
    n_pairs = len(all_pair_results)
    if n_pairs == 0:
        return {
            "avg_round_trips_per_pair": 0.0,
            "avg_holding_duration_days": 0.0,
            "pct_pairs_never_opened": 0.0,
        }

    total_round_trips = 0
    durations: list[int] = []
    n_never_opened = 0

    for res in all_pair_results:
        trades = res["trades"]
        if not any(ev["event"] == "open" for ev in trades):
            n_never_opened += 1
        total_round_trips += sum(1 for ev in trades if ev["event"] == "close")

        open_day = None
        for ev in trades:
            if ev["event"] == "open":
                open_day = ev["day"]
            elif ev["event"] == "close" and open_day is not None:
                durations.append(ev["day"] - open_day)
                open_day = None

    return {
        "avg_round_trips_per_pair": total_round_trips / n_pairs,
        "avg_holding_duration_days": float(np.mean(durations)) if durations else 0.0,
        "pct_pairs_never_opened": n_never_opened / n_pairs,
    }
