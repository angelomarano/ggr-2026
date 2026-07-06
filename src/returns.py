"""
returns.py — Portfolio-level aggregation of pair payoffs (output of
simulate_pair_same_day / simulate_pair_wait_one_day) and composition of the
6 staggered monthly portfolios into a single series (PROTOCOL.md §2.2).

Two levels:
1. aggregate_portfolio_run: WITHIN a single run (one formation date), sums
   the daily payoffs of all selected pairs and computes
   - return on COMMITTED capital = payoff sum / N selected pairs
     (the denominator is the NOMINAL portfolio size, e.g. 20: it must be
     passed explicitly as n_selected, not inferred from how many pairs are
     actually available — see the function's docstring);
   - return on EMPLOYED capital = payoff sum / N pairs OPEN that specific
     day; if zero pairs are open, the return is 0.0 (never NaN, never a
     silent division by zero).
2. compound_to_monthly + combine_overlapping_portfolios: bring the daily
   series of ONE portfolio to monthly returns (compounding within the
   calendar month), then combine the monthly returns of multiple staggered
   runs with Jegadeesh-Titman averaging (simple average across the
   portfolios active in each calendar month).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pair_employed_mask(trades: list[dict], n_days: int) -> np.ndarray:
    """
    Boolean mask (length n_days, index 0..n_days-1 <-> day 1..n_days) that
    marks the days on which the pair has an open position "employed" for
    the purposes of employed capital.

    Convention (consistent with trading.py: no payoff on the opening day,
    weights start marking to market ONLY from the following day): a pair
    opened on day open_day and closed on day close_day is "employed" on
    days (open_day, close_day] inclusive — i.e. it excludes the opening day
    itself (payoff=0 by construction, no capital yet marked to market) and
    includes the closing day (the last payoff is still realized that day
    before closing).

    "Missed" events (wait-one-day, missed opportunity) are ignored: they
    never open a position, they contribute no employed capital.
    """
    mask = np.zeros(n_days, dtype=bool)
    open_day = None
    for ev in trades:
        if ev["event"] == "open":
            open_day = ev["day"]
        elif ev["event"] == "close" and open_day is not None:
            close_day = ev["day"]
            mask[open_day:close_day] = True
            open_day = None
    return mask


def aggregate_portfolio_run(
    pair_results: dict[str, dict], n_days: int, n_selected: int | None = None
) -> pd.DataFrame:
    """
    pair_results: {pair_id: result of simulate_pair_same_day/wait_one_day}
    for the portfolio's pairs in ONE run (including pairs never opened:
    they contribute payoff=0 every day and no "employed" day).
    n_days: length of the trading period (must match for all pairs: same
    run, same trading period).
    n_selected: NOMINAL portfolio size (e.g. config.TOP_PAIRS=20 for top-20;
    can differ from len(pair_results) if the universe doesn't supply enough
    candidate pairs, PROTOCOL.md §2.1 assumes a large universe but
    src/formation.py can return fewer pairs on a reduced golden set).
    Default: len(pair_results) if not specified.
    If n_selected == 0 (no selectable pair), the committed return is 0.0
    for every day instead of a ZeroDivisionError: an empty portfolio
    generates no return, it's not an undefined case.

    Returns a DataFrame indexed 0..n_days-1 (day = index+1) with columns:
      payoff_sum          sum of daily payoffs across all pairs
      n_open              number of pairs open that day (employed)
      committed_return    payoff_sum / n_selected (0.0 if n_selected==0)
      employed_return     payoff_sum / n_open, 0.0 if n_open==0 (never NaN)

    Also returns long_payoff_sum/short_payoff_sum and their committed-capital
    returns (long_committed_return, short_committed_return), summing each
    pair's daily_long_payoff/daily_short_payoff (PROTOCOL.md §2.4, long/short
    alpha decomposition — feed these into inference.long_short_leg_regression
    after compound_to_monthly). Same n_selected denominator and zero-division
    handling as committed_return; requires pair_results built from a
    simulate_pair_* version that returns daily_long_payoff/daily_short_payoff.
    """
    if n_selected is None:
        n_selected = len(pair_results)

    payoff_sum = np.zeros(n_days)
    long_payoff_sum = np.zeros(n_days)
    short_payoff_sum = np.zeros(n_days)
    n_open = np.zeros(n_days, dtype=int)
    for res in pair_results.values():
        daily_payoff = np.asarray(res["daily_payoff"])
        assert len(daily_payoff) == n_days, "all pairs in the run must have the same trading period"
        payoff_sum += daily_payoff
        long_payoff_sum += np.asarray(res["daily_long_payoff"])
        short_payoff_sum += np.asarray(res["daily_short_payoff"])
        n_open += pair_employed_mask(res["trades"], n_days).astype(int)

    if n_selected > 0:
        committed_return = payoff_sum / n_selected
        long_committed_return = long_payoff_sum / n_selected
        short_committed_return = short_payoff_sum / n_selected
    else:
        committed_return = np.zeros(n_days)
        long_committed_return = np.zeros(n_days)
        short_committed_return = np.zeros(n_days)

    employed_return = np.divide(
        payoff_sum, n_open, out=np.zeros_like(payoff_sum), where=n_open > 0
    )

    return pd.DataFrame({
        "payoff_sum": payoff_sum,
        "n_open": n_open,
        "committed_return": committed_return,
        "employed_return": employed_return,
        "long_payoff_sum": long_payoff_sum,
        "short_payoff_sum": short_payoff_sum,
        "long_committed_return": long_committed_return,
        "short_committed_return": short_committed_return,
    })


def compound_to_monthly(daily_returns, dates) -> pd.Series:
    """
    daily_returns: array-like of length n, decimal daily returns of ONE
    portfolio in ONE trading period (e.g. the committed_return or
    employed_return column of aggregate_portfolio_run).
    dates: corresponding calendar dates (same length n).

    Returns the compounded return per calendar month:
    prod_t(1+r_t) - 1 over the month's days, indexed by start-of-month
    Timestamp (consistent with data/factors.py).
    """
    s = pd.Series(np.asarray(daily_returns, dtype=float), index=pd.DatetimeIndex(dates))
    monthly = s.groupby(s.index.to_period("M")).apply(lambda x: float(np.prod(1 + x.to_numpy()) - 1))
    monthly.index = monthly.index.to_timestamp(how="start")
    monthly.index.name = "month"
    return monthly


def combine_overlapping_portfolios(monthly_by_run: dict[str, pd.Series]) -> pd.Series:
    """
    monthly_by_run: {run_id: monthly return series of ONE staggered
    portfolio (one per monthly formation date), indexed by calendar month
    (start-of-month Timestamp), as produced by compound_to_monthly.

    Jegadeesh-Titman averaging (PROTOCOL.md §2.2): the strategy's return in
    a given calendar month is the SIMPLE AVERAGE of the returns of the
    portfolios active that month (up to N_OVERLAPPING=6 runs, one for each
    of the preceding months' formation dates). A month in which no
    portfolio is active (window edges) doesn't appear at all in the
    resulting series: it's not a value of 0, it's simply out of domain.
    """
    combined = pd.concat(monthly_by_run.values(), axis=1)
    return combined.mean(axis=1, skipna=True).dropna().sort_index()
