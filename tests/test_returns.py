"""
test_returns.py — aggregate_portfolio_run (committed/employed capital),
pair_employed_mask (multiple episodes), compound_to_monthly and
combine_overlapping_portfolios (Jegadeesh-Titman averaging), with numbers
computed by hand on a small fictitious portfolio.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.returns import (
    aggregate_portfolio_run,
    combine_overlapping_portfolios,
    compound_to_monthly,
    pair_employed_mask,
)
from src.trading import simulate_pair_same_day, simulate_pair_wait_one_day

TOL = 1e-9


def test_pair_employed_mask_excludes_open_day_includes_close_day():
    """
    A single episode: opens on day 2, closes on day 4 (n_days=5).
    Employed = days (2,4] = {3,4} -> 0-based indices {2,3}.
    """
    trades = [
        {"event": "open", "day": 2},
        {"event": "close", "day": 4, "reason": "crossing"},
    ]
    mask = pair_employed_mask(trades, n_days=5)
    assert list(mask) == [False, False, True, True, False]


def test_pair_employed_mask_multiple_episodes():
    """
    Two episodes in the same run (reopening after convergence, PROTOCOL.md
    §2.1): open@1-close@2, then open@3-close@5 (n_days=5).
    Expected employed: day2 (episode1) and days4,5 (episode2).
    0-based indices: {1, 3, 4} -> [F,T,F,T,T].
    """
    trades = [
        {"event": "open", "day": 1},
        {"event": "close", "day": 2, "reason": "crossing"},
        {"event": "open", "day": 3},
        {"event": "close", "day": 5, "reason": "end_of_period"},
    ]
    mask = pair_employed_mask(trades, n_days=5)
    assert list(mask) == [False, True, False, True, True]


def test_aggregate_portfolio_run_committed_and_employed():
    """
    Portfolio of 2 pairs (n_selected=2), n_days=2:
      Pair A: identical to episode1 of test_synthetic_pair.py (same-day).
        returns_1=[0.20,-1/12], returns_2=[0.00,0.10], sigma=0.05.
        Opens day1, closes day2 on crossing. daily_payoff=[0, 11/60].
        trades=[open@1, close@2] -> employed = [False, True].
      Pair B: never opens (spread always below threshold). daily_payoff=[0,0],
        trades=[] -> employed=[False, False].

    Expected:
      payoff_sum = [0, 11/60]
      n_open     = [0, 1]
      committed_return = payoff_sum / 2 = [0, 11/120]
      employed_return  = [0.0 (n_open=0 -> 0, never NaN), (11/60)/1]
    """
    pair_A = simulate_pair_same_day(
        np.array([0.20, -1 / 12]), np.array([0.00, 0.10]), sigma=0.05, k=2.0
    )
    pair_B = simulate_pair_same_day(
        np.array([0.01, -0.01]), np.array([0.00, 0.00]), sigma=0.05, k=2.0
    )
    assert pair_B["trades"] == [], "pair B must never open (test precondition)"

    agg = aggregate_portfolio_run({"A": pair_A, "B": pair_B}, n_days=2, n_selected=2)

    assert np.allclose(agg["payoff_sum"], [0.0, 11 / 60])
    assert list(agg["n_open"]) == [0, 1]
    assert np.allclose(agg["committed_return"], [0.0, 11 / 120])
    assert abs(agg["employed_return"].iloc[0] - 0.0) < TOL, "n_open=0 -> return 0.0, never NaN"
    assert abs(agg["employed_return"].iloc[1] - 11 / 60) < TOL


def test_aggregate_portfolio_run_long_short_committed_returns():
    """
    Same 2-pair portfolio as test_aggregate_portfolio_run_committed_and_employed.
    Pair A's day-2 payoff (11/60) decomposes into long_contribution=0.10
    (long leg 2, w=1.0 * r2[1]=0.10) and short_contribution=1/12
    (short leg 1, w=1.0 * r1[1]=-1/12, i.e. daily_short_payoff=-1/12).
    Pair B never opens: zero contribution on both legs.
    Expected: long_payoff_sum=[0, 0.10], short_payoff_sum=[0, -1/12],
    long_committed_return = long_payoff_sum/2, short_committed_return = short_payoff_sum/2.
    """
    pair_A = simulate_pair_same_day(
        np.array([0.20, -1 / 12]), np.array([0.00, 0.10]), sigma=0.05, k=2.0
    )
    pair_B = simulate_pair_same_day(
        np.array([0.01, -0.01]), np.array([0.00, 0.00]), sigma=0.05, k=2.0
    )

    agg = aggregate_portfolio_run({"A": pair_A, "B": pair_B}, n_days=2, n_selected=2)

    assert np.allclose(agg["long_payoff_sum"], [0.0, 0.10])
    assert np.allclose(agg["short_payoff_sum"], [0.0, -1 / 12])
    assert np.allclose(agg["long_committed_return"], [0.0, 0.05])
    assert np.allclose(agg["short_committed_return"], [0.0, -1 / 24])
    # sanity: long - short must reconcile with the combined committed_return
    assert np.allclose(
        agg["long_committed_return"] - agg["short_committed_return"], agg["committed_return"]
    )


def test_aggregate_portfolio_run_zero_selected_no_division_error():
    """n_selected=0 (no selectable pair, e.g. golden set too small):
    committed_return must be zero everywhere, not an error."""
    agg = aggregate_portfolio_run({}, n_days=3, n_selected=0)
    assert np.allclose(agg["committed_return"], [0.0, 0.0, 0.0])
    assert np.allclose(agg["employed_return"], [0.0, 0.0, 0.0])


def test_aggregate_portfolio_run_ignores_wait_one_day_missed_event():
    """
    PROTOCOL.md §2.1 audit (wait-one-day execution always run alongside
    same-day): a result with a "missed" event (missed opportunity, no
    position ever opened) must aggregate as a pair that never opened —
    zero payoff, zero n_open, no crash on an event other than open/close.
      returns_1=[0.20,-0.15], returns_2=[0.00,0.00], sigma=0.05 -> signal
      on day1, fell back below threshold on day2 -> trades=[missed].
    """
    pair_missed = simulate_pair_wait_one_day(
        np.array([0.20, -0.15]), np.array([0.00, 0.00]), sigma=0.05, k=2.0
    )
    assert pair_missed["trades"][0]["event"] == "missed", "test precondition"

    agg = aggregate_portfolio_run({"X": pair_missed}, n_days=2, n_selected=1)
    assert np.allclose(agg["payoff_sum"], [0.0, 0.0])
    assert list(agg["n_open"]) == [0, 0]
    assert np.allclose(agg["committed_return"], [0.0, 0.0])
    assert np.allclose(agg["employed_return"], [0.0, 0.0])


def test_compound_to_monthly_groups_by_calendar_month():
    """
    3 days: 2 in January, 1 in February.
      January: r = [0.01, 0.02] -> (1.01*1.02)-1 = 0.0302
      February: r = [-0.01] -> -0.01
    """
    daily = [0.01, 0.02, -0.01]
    dates = pd.to_datetime(["2020-01-15", "2020-01-20", "2020-02-05"])
    monthly = compound_to_monthly(daily, dates)

    assert list(monthly.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
    assert abs(monthly.iloc[0] - (1.01 * 1.02 - 1)) < TOL
    assert abs(monthly.iloc[1] - (-0.01)) < TOL


def test_combine_overlapping_portfolios_simple_average():
    """
    run1: Jan=0.01, Feb=0.02
    run2:      Feb=0.04, Mar=0.03
    Expected: Jan=0.01 (run1 only), Feb=(0.02+0.04)/2=0.03 (average of the
    2 active runs), Mar=0.03 (run2 only).
    """
    run1 = pd.Series(
        {pd.Timestamp("2020-01-01"): 0.01, pd.Timestamp("2020-02-01"): 0.02}
    )
    run2 = pd.Series(
        {pd.Timestamp("2020-02-01"): 0.04, pd.Timestamp("2020-03-01"): 0.03}
    )
    combined = combine_overlapping_portfolios({"run1": run1, "run2": run2})

    assert abs(combined[pd.Timestamp("2020-01-01")] - 0.01) < TOL
    assert abs(combined[pd.Timestamp("2020-02-01")] - 0.03) < TOL
    assert abs(combined[pd.Timestamp("2020-03-01")] - 0.03) < TOL
    assert len(combined) == 3


if __name__ == "__main__":
    test_pair_employed_mask_excludes_open_day_includes_close_day()
    test_pair_employed_mask_multiple_episodes()
    test_aggregate_portfolio_run_committed_and_employed()
    test_aggregate_portfolio_run_long_short_committed_returns()
    test_aggregate_portfolio_run_zero_selected_no_division_error()
    test_aggregate_portfolio_run_ignores_wait_one_day_missed_event()
    test_compound_to_monthly_groups_by_calendar_month()
    test_combine_overlapping_portfolios_simple_average()
    print("test_returns: all tests PASSED.")
