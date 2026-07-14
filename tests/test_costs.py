"""
test_costs.py -- apply_round_trip_cost/apply_cost_grid (PROTOCOL.md §5:
4*c per round-trip, applied on the closing day) and breakeven_cost_bp
(linear interpolation for c*), on hand-built fixtures with numbers
verifiable by hand.
"""
import sys
sys.path.insert(0, ".")

import numpy as np

from src.costs import apply_cost_grid, apply_round_trip_cost, breakeven_cost_bp

TOL = 1e-9


def _three_round_trip_fixture():
    """
    3 completed round trips over an 11-day trading period, gross payoff on
    each round trip's closing day: +0.02 (day 3), -0.01 (day 7), +0.015
    (day 11, closed via end_of_period rather than crossing -- still a full
    round trip per PROTOCOL.md §5). Gross total = 0.02 - 0.01 + 0.015 =
    0.025.
    """
    trades = [
        {"event": "open", "day": 1, "direction": "long1_short2", "spread": -0.1},
        {"event": "close", "day": 3, "reason": "crossing", "spread": 0.0},
        {"event": "open", "day": 5, "direction": "long2_short1", "spread": 0.1},
        {"event": "close", "day": 7, "reason": "crossing", "spread": 0.0},
        {"event": "open", "day": 9, "direction": "long1_short2", "spread": -0.1},
        {"event": "close", "day": 11, "reason": "end_of_period", "spread": -0.05},
    ]
    daily_payoff = np.zeros(11)
    daily_payoff[2] = 0.02    # day 3 -> index 2
    daily_payoff[6] = -0.01   # day 7 -> index 6
    daily_payoff[10] = 0.015  # day 11 -> index 10
    return {"trades": trades, "daily_payoff": daily_payoff}


def test_apply_round_trip_cost_hand_computed_at_10bp():
    """
    3 round trips, c=10bp=0.001 -> cost per round trip = 4*0.001=0.004,
    total cost = 3*0.004=0.012. Net total = 0.025 - 0.012 = 0.013 exactly.
    """
    pair_result = _three_round_trip_fixture()
    gross_total = float(np.sum(pair_result["daily_payoff"]))
    assert abs(gross_total - 0.025) < TOL

    net = apply_round_trip_cost(pair_result, cost_bp_per_side=10)

    expected_net_total = 0.025 - 3 * 4 * 0.001
    assert abs(expected_net_total - 0.013) < TOL, "sanity check on the hand arithmetic itself"
    assert abs(float(np.sum(net["daily_payoff"])) - expected_net_total) < TOL

    # cost lands exactly on each closing day, nowhere else
    assert abs(net["daily_payoff"][2] - (0.02 - 0.004)) < TOL
    assert abs(net["daily_payoff"][6] - (-0.01 - 0.004)) < TOL
    assert abs(net["daily_payoff"][10] - (0.015 - 0.004)) < TOL
    untouched_days = [i for i in range(11) if i not in (2, 6, 10)]
    for i in untouched_days:
        assert net["daily_payoff"][i] == 0.0


def test_apply_round_trip_cost_zero_leaves_payoff_unchanged():
    """c=0 -> daily_payoff numerically identical to the gross input (a
    copy, not the same object, but every value unchanged)."""
    pair_result = _three_round_trip_fixture()
    net = apply_round_trip_cost(pair_result, cost_bp_per_side=0)
    assert np.array_equal(net["daily_payoff"], pair_result["daily_payoff"])
    assert net["daily_payoff"] is not pair_result["daily_payoff"], "must return a copy, not mutate the input"


def test_apply_round_trip_cost_does_not_mutate_input():
    """The original pair_result's daily_payoff must be untouched after
    calling apply_round_trip_cost with a nonzero cost - callers need to
    reuse the same simulate_pair_* output across every grid level."""
    pair_result = _three_round_trip_fixture()
    original = pair_result["daily_payoff"].copy()
    apply_round_trip_cost(pair_result, cost_bp_per_side=40)
    assert np.array_equal(pair_result["daily_payoff"], original)


def test_apply_round_trip_cost_zero_round_trips_stays_zero_not_nan():
    """A pair that never opens (empty trade log, PROTOCOL.md's 'pairs
    never opened' case) has zero round trips: cost must leave its
    all-zero payoff at zero, never NaN, at any cost level."""
    pair_result = {"trades": [], "daily_payoff": np.zeros(20)}
    for c in (0, 5, 10, 20, 40):
        net = apply_round_trip_cost(pair_result, cost_bp_per_side=c)
        assert not np.isnan(net["daily_payoff"]).any()
        assert float(np.sum(net["daily_payoff"])) == 0.0


def test_apply_round_trip_cost_open_without_close_incurs_no_cost():
    """Defensive edge case (not expected from a real simulate_pair_*
    output, which always closes an open position by construction, but
    apply_round_trip_cost only ever looks at "close" events): a dangling
    "open" with no matching "close" contributes zero cost, no crash."""
    pair_result = {
        "trades": [{"event": "open", "day": 1, "direction": "long1_short2", "spread": -0.1}],
        "daily_payoff": np.array([0.0, 0.01, 0.02]),
    }
    net = apply_round_trip_cost(pair_result, cost_bp_per_side=40)
    assert np.array_equal(net["daily_payoff"], pair_result["daily_payoff"])


def test_apply_cost_grid_matches_manual_loop_over_pairs():
    """apply_cost_grid on a 2-pair portfolio must equal calling
    apply_round_trip_cost on each pair individually, same cost level."""
    pair_a = _three_round_trip_fixture()
    pair_b = {"trades": [], "daily_payoff": np.zeros(11)}
    pair_results = {"A": pair_a, "B": pair_b}

    out = apply_cost_grid(pair_results, cost_bp_per_side=20)

    assert np.array_equal(out["A"]["daily_payoff"], apply_round_trip_cost(pair_a, 20)["daily_payoff"])
    assert np.array_equal(out["B"]["daily_payoff"], apply_round_trip_cost(pair_b, 20)["daily_payoff"])


# ------------------------------------------------------------ break-even

def test_breakeven_cost_bp_exact_crossing_at_a_grid_point():
    """
    mean(c) = 0.002 - 0.0001*c (c in bp) evaluated at the frozen grid
    [0,5,10,20,40] gives [0.002, 0.0015, 0.001, 0.0, -0.002] - the
    crossing falls EXACTLY on the c=20bp grid point (mean=0.0 exactly),
    so interpolation between (10, 0.001) and (20, 0.0) must recover
    c*=20.0 exactly: c* = 10 + (0-0.001)*(20-10)/(0.0-0.001) = 10+10 = 20.
    """
    grid = (0, 5, 10, 20, 40)
    means = [0.002 - 0.0001 * c for c in grid]
    assert means == [0.002, 0.0015, 0.001, 0.0, -0.002]  # sanity on the hand arithmetic

    result = breakeven_cost_bp(grid, means)

    assert result["c_star_bp"] is not None
    assert abs(result["c_star_bp"] - 20.0) < TOL


def test_breakeven_cost_bp_interpolates_between_grid_points():
    """
    means = [0.002, 0.001, 0.0005, -0.0005, -0.002] at grid [0,5,10,20,40]:
    crossing is between c=10 (mean=0.0005) and c=20 (mean=-0.0005).
    c* = 10 + (0-0.0005)*(20-10)/(-0.0005-0.0005) = 10 + (-0.0005)*10/(-0.001)
       = 10 + 5 = 15.0 exactly.
    """
    grid = (0, 5, 10, 20, 40)
    means = [0.002, 0.001, 0.0005, -0.0005, -0.002]

    result = breakeven_cost_bp(grid, means)

    assert result["c_star_bp"] is not None
    assert abs(result["c_star_bp"] - 15.0) < TOL


def test_breakeven_cost_bp_never_crosses_zero_reports_none_not_forced():
    """All grid points positive and decreasing but never reaching zero:
    c_star_bp must be None (not a forced/extrapolated number), with an
    explanatory note."""
    grid = (0, 5, 10, 20, 40)
    means = [0.002, 0.0018, 0.0015, 0.001, 0.0005]

    result = breakeven_cost_bp(grid, means)

    assert result["c_star_bp"] is None
    assert "never crosses zero" in result["note"]


def test_breakeven_cost_bp_already_nonpositive_at_lowest_grid_point():
    """Mean return is already <=0 even at c=0 (gross): c* is reported as
    the first grid point itself (an upper bound), not interpolated below
    the grid's own range."""
    grid = (0, 5, 10, 20, 40)
    means = [-0.001, -0.002, -0.003, -0.004, -0.005]

    result = breakeven_cost_bp(grid, means)

    assert result["c_star_bp"] == 0.0
    assert "already <=0" in result["note"]


if __name__ == "__main__":
    test_apply_round_trip_cost_hand_computed_at_10bp()
    test_apply_round_trip_cost_zero_leaves_payoff_unchanged()
    test_apply_round_trip_cost_does_not_mutate_input()
    test_apply_round_trip_cost_zero_round_trips_stays_zero_not_nan()
    test_apply_round_trip_cost_open_without_close_incurs_no_cost()
    test_apply_cost_grid_matches_manual_loop_over_pairs()
    test_breakeven_cost_bp_exact_crossing_at_a_grid_point()
    test_breakeven_cost_bp_interpolates_between_grid_points()
    test_breakeven_cost_bp_never_crosses_zero_reports_none_not_forced()
    test_breakeven_cost_bp_already_nonpositive_at_lowest_grid_point()
    print("test_costs: all tests PASSED.")
