"""
costs.py -- Transaction-cost model (PROTOCOL.md §5): a flat per-round-trip
cost applied in AGGREGATION, never inside the trading engine itself
(src/trading.py's simulate_pair_* stay untouched -- trigger/crossing/
delisting mechanics are frozen GGR core; cost is a downstream overlay on
their output).

PROTOCOL.md §5, Level 2: "griglia esplicita c in {0,5,10,20,40} bp per lato
per trade; un round-trip di coppia = 4 trade -> costo 4c per round-trip."
config.COST_GRID_BP_PER_SIDE already declares the frozen grid (unused
until now) -- reused here as-is, not redefined.

A round trip is counted on ANY closing event, regardless of the reason
(crossing, delisting, end_of_period): PROTOCOL.md §5's "4 trade per
round-trip" is unconditional -- 2 trades to open, 2 to close -- since all
three closure paths in src/trading.py send the same 2 closing trades to
market, the only difference is WHY the position closed, not how many
trades that took.
"""
from __future__ import annotations

import numpy as np


def apply_round_trip_cost(pair_result: dict, cost_bp_per_side: float) -> dict:
    """
    Subtracts 4*c from the pair's daily_payoff on the day each round trip
    CLOSES (c = cost_bp_per_side / 10000, so 4*c is the round-trip cost in
    the same $-per-$1-notional units as daily_payoff). Returns a NEW dict
    (shallow copy of pair_result with only "daily_payoff" replaced) --
    does NOT mutate the input, so the same simulate_pair_* output can be
    reused across every level of the cost grid without re-simulating.

    Only daily_payoff is adjusted; daily_long_payoff/daily_short_payoff
    (the long/short alpha decomposition inputs, PROTOCOL.md §2.4) are left
    untouched and are NOT reconciled with the cost-adjusted net payoff --
    this cost model has no per-leg attribution (PROTOCOL.md §5 doesn't
    give one), so daily_long_payoff - daily_short_payoff no longer equals
    daily_payoff once a nonzero cost is applied. The long/short
    decomposition is not part of what this function is for.

    A pair with zero "close" events (never opened, or - not possible from
    a real simulate_pair_* output, but handled gracefully regardless -
    opened without ever closing) incurs zero cost: daily_payoff is
    returned as an unchanged copy, not NaN or an error.
    """
    c = cost_bp_per_side / 10000.0
    daily_payoff = np.array(pair_result["daily_payoff"], dtype=float, copy=True)
    for ev in pair_result["trades"]:
        if ev["event"] == "close":
            daily_payoff[ev["day"] - 1] -= 4 * c
    out = dict(pair_result)
    out["daily_payoff"] = daily_payoff
    return out


def apply_cost_grid(pair_results: dict[str, dict], cost_bp_per_side: float) -> dict[str, dict]:
    """apply_round_trip_cost over an entire portfolio-run's pair_results
    dict (same {pair_id: simulate_pair_* output} shape
    src/returns.aggregate_portfolio_run expects), at one cost level. Feed
    the result straight into aggregate_portfolio_run to get committed/
    employed returns net of cost, exactly as if the level-0 payoffs had
    been recomputed with cost baked in."""
    return {pid: apply_round_trip_cost(res, cost_bp_per_side) for pid, res in pair_results.items()}


def breakeven_cost_bp(cost_grid_bp: tuple[float, ...], mean_returns: list[float]) -> dict:
    """
    PROTOCOL.md §5: "costo di break-even c*" -- the per-side cost level at
    which the portfolio's mean monthly return crosses zero. Linear
    interpolation between the two adjacent grid points that bracket the
    sign change (config.COST_GRID_BP_PER_SIDE has only 5 points, not fine
    enough to read c* off directly).

    cost_grid_bp, mean_returns: parallel sequences, same length and order
    (typically config.COST_GRID_BP_PER_SIDE and the corresponding mean
    monthly returns computed via apply_cost_grid + aggregate_portfolio_run
    + compounding, one per grid point).

    Scans left to right for the first adjacent pair with mean_returns[i] >
    0 and mean_returns[i+1] <= 0 (increasing cost should push returns down
    monotonically in the typical case; scanning for the first sign change
    is a simple, declared, robust-enough rule if it isn't perfectly
    monotonic in practice). Returns {"c_star_bp": float, "note": str}.

    Two special cases, both reported explicitly rather than forcing a
    number:
      - already <= 0 at the very first (lowest) grid point: c_star_bp is
        that first point, noted as an upper bound (not interpolated -- the
        strategy isn't profitable even at the smallest tested cost).
      - never crosses zero anywhere in the grid (stays positive
        throughout): c_star_bp is None, noted as "> the grid's highest
        point" -- extrapolation beyond the frozen grid is not attempted.
    """
    grid = list(cost_grid_bp)
    means = list(mean_returns)
    if len(grid) != len(means):
        raise ValueError(f"cost_grid_bp and mean_returns must be the same length, got {len(grid)} and {len(means)}")
    if len(grid) < 2:
        raise ValueError("need at least 2 grid points to interpolate a break-even cost")

    if means[0] <= 0:
        return {
            "c_star_bp": float(grid[0]),
            "note": f"mean return is already <=0 at the lowest grid point (c={grid[0]}bp); "
                    "c* is reported as that point (an upper bound, not interpolated) -- the "
                    "strategy is not profitable even at the smallest cost level tested.",
        }

    for i in range(len(grid) - 1):
        if means[i] > 0 and means[i + 1] <= 0:
            c_lo, c_hi = grid[i], grid[i + 1]
            m_lo, m_hi = means[i], means[i + 1]
            c_star = c_lo + (0 - m_lo) * (c_hi - c_lo) / (m_hi - m_lo)
            return {
                "c_star_bp": float(c_star),
                "note": f"linearly interpolated between grid points c={c_lo}bp (mean={m_lo:.6f}) "
                        f"and c={c_hi}bp (mean={m_hi:.6f}), where the sign change occurs.",
            }

    return {
        "c_star_bp": None,
        "note": "mean return never crosses zero within the tested grid "
                f"({grid[0]}-{grid[-1]}bp per side) -- stays positive throughout, so c* > {grid[-1]}bp "
                "(extrapolation beyond the frozen grid not attempted, per PROTOCOL.md §5's explicit grid).",
    }
