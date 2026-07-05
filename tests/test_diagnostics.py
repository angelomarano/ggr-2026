"""
test_diagnostics.py -- annualized_sharpe, max_drawdown, pct_negative_months,
trade_statistics, with hand-computed expected values.
"""
import sys
sys.path.insert(0, ".")

import numpy as np

from src.diagnostics import annualized_sharpe, max_drawdown, pct_negative_months, trade_statistics

TOL = 1e-9


def test_annualized_sharpe_hand_computed():
    """
    monthly_returns = [0.02, 0.04, -0.01, 0.03]
    mean = 0.08/4 = 0.02
    deviations: 0.00, 0.02, -0.03, 0.01 -> squares sum = 0.0014
    sample variance (ddof=1, n-1=3) = 0.0014/3 = 0.00046666...
    std = sqrt(0.00046666...) = 0.0216024689946929
    sharpe_monthly = 0.02 / 0.0216024689946929 = 0.9258200997725512
    annualized = sharpe_monthly * sqrt(12) = 3.2071349029490914
    """
    r = [0.02, 0.04, -0.01, 0.03]
    assert abs(annualized_sharpe(r) - 3.2071349029490914) < 1e-9


def test_annualized_sharpe_zero_std_is_nan_not_crash():
    """A constant series has an undefined Sharpe ratio: nan, not a
    ZeroDivisionError."""
    assert np.isnan(annualized_sharpe([0.01, 0.01, 0.01]))


def test_max_drawdown_hand_computed():
    """
    monthly_returns = [0.10, -0.20, 0.05, 0.10]
    wealth: 1.0, 1.10, 0.88, 0.924, 1.0164
    running max: 1.0, 1.10, 1.10, 1.10, 1.10
    drawdown: 0, 0, 0.88/1.10-1=-0.20, 0.924/1.10-1=-0.16, 1.0164/1.10-1=-0.076
    max drawdown = -0.20 (worst point, right after the -20% month)
    """
    r = [0.10, -0.20, 0.05, 0.10]
    assert abs(max_drawdown(r) - (-0.20)) < TOL


def test_max_drawdown_all_positive_is_zero():
    """A series that never dips below its running peak has zero drawdown."""
    assert abs(max_drawdown([0.01, 0.02, 0.01]) - 0.0) < TOL


def test_pct_negative_months_hand_computed():
    """[0.01, -0.02, 0.03, -0.01, 0.00] -> 2 negative out of 5 = 0.4.
    A zero return is not negative."""
    r = [0.01, -0.02, 0.03, -0.01, 0.00]
    assert abs(pct_negative_months(r) - 0.4) < TOL


def test_trade_statistics_hand_computed():
    """
    Pair A: open@1, close@3 -> 1 round trip, duration 2.
    Pair B: never opened.
    Pair C: open@1, close@2, open@5, close@10 -> 2 round trips,
             durations 1 and 5.

    n_pairs = 3
    total round trips = 1 + 0 + 2 = 3 -> avg_round_trips_per_pair = 1.0
    durations = [2, 1, 5] -> mean = 8/3
    never opened = 1 (B) / 3 = 0.333...
    """
    pair_a = {"trades": [{"event": "open", "day": 1}, {"event": "close", "day": 3}]}
    pair_b = {"trades": []}
    pair_c = {
        "trades": [
            {"event": "open", "day": 1}, {"event": "close", "day": 2},
            {"event": "open", "day": 5}, {"event": "close", "day": 10},
        ]
    }

    stats = trade_statistics([pair_a, pair_b, pair_c])

    assert abs(stats["avg_round_trips_per_pair"] - 1.0) < TOL
    assert abs(stats["avg_holding_duration_days"] - 8 / 3) < TOL
    assert abs(stats["pct_pairs_never_opened"] - 1 / 3) < TOL


def test_trade_statistics_ignores_missed_events():
    """A wait-one-day "missed" event (never opened, PROTOCOL.md §2.4) must
    count toward pct_pairs_never_opened, not as a round trip."""
    pair = {"trades": [{"event": "missed", "day": 2}]}
    stats = trade_statistics([pair])
    assert stats["avg_round_trips_per_pair"] == 0.0
    assert stats["pct_pairs_never_opened"] == 1.0


def test_trade_statistics_empty_input():
    """No pairs at all -> all-zero stats, not an error."""
    stats = trade_statistics([])
    assert stats == {
        "avg_round_trips_per_pair": 0.0,
        "avg_holding_duration_days": 0.0,
        "pct_pairs_never_opened": 0.0,
    }


if __name__ == "__main__":
    test_annualized_sharpe_hand_computed()
    test_annualized_sharpe_zero_std_is_nan_not_crash()
    test_max_drawdown_hand_computed()
    test_max_drawdown_all_positive_is_zero()
    test_pct_negative_months_hand_computed()
    test_trade_statistics_hand_computed()
    test_trade_statistics_ignores_missed_events()
    test_trade_statistics_empty_input()
    print("test_diagnostics: all tests PASSED.")
