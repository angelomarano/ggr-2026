"""
test_synthetic_pair.py — The PROTOCOL.md §6 test: synthetic pair with trade
and P&L computed BY HAND. If this doesn't pass, no real data.

Two independent episodes (one call to simulate_pair_same_day each, so state
resets):

EPISODE 1 — open long2/short1, close on CROSSING after 1 day.
  sigma=0.05 (threshold=0.10). returns_1=[+0.20, -1/12], returns_2=[0.00, 0.10].
  Day1: P1=1.20, P2=1.00, spread=+0.20 > 0.10 -> opens long leg2/short leg1.
  Day2: P1=1.20*(11/12)=1.10, P2=1.00*1.10=1.10, spread=0 -> crossing, closes.
  Day2 payoff = 1*r2 - 1*r1 = 0.10 - (-1/12) = 11/60 (by hand, verified).

EPISODE 2 — open long1/short2, held 2 days with weight compounding,
  forced closure at END OF PERIOD (no natural crossing).
  sigma=0.05. returns_1=[-0.20, 0.00, 0.30], returns_2=[0.00, 0.05, 0.02].
  Day1: P1=0.80, P2=1.00, spread=-0.20 < -0.10 -> opens long leg1/short leg2.
  Day2: payoff = w_long(1.0)*0.00 - w_short(1.0)*0.05 = -0.05
        weights after: w_long=1.0, w_short=1.05 (spread still -0.25, no cross)
  Day3: payoff = w_long(1.0)*0.30 - w_short(1.05)*0.02 = 0.30-0.021 = 0.279
        final spread = 1.04-1.071 = -0.031 (same sign -> NO natural crossing)
        -> forced closure at end of period (t==n).
  Total = -0.05 + 0.279 = 0.229 (by hand, verified).
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.trading import simulate_pair_same_day, simulate_pair_wait_one_day

TOL = 1e-9


def test_episode1_crossing_close():
    returns_1 = np.array([0.20, -1 / 12])
    returns_2 = np.array([0.00, 0.10])
    r = simulate_pair_same_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert abs(r["spread"][1] - 0.20) < TOL, "expected day1 spread 0.20"
    assert len(r["trades"]) == 2, f"expected 2 events (open+close), found {len(r['trades'])}"

    open_ev, close_ev = r["trades"]
    assert open_ev["event"] == "open" and open_ev["day"] == 1
    assert open_ev["direction"] == "long2_short1", "spread>threshold -> long the low leg (2), short the high one (1)"
    assert close_ev["event"] == "close" and close_ev["day"] == 2
    assert close_ev["reason"] == "crossing"
    assert abs(close_ev["spread"]) < TOL, "spread must be ~0 at crossing"

    expected_payoff_day2 = 11 / 60
    assert abs(r["daily_payoff"][1] - expected_payoff_day2) < TOL
    assert abs(r["daily_payoff"][0] - 0.0) < TOL, "no payoff on the opening day"
    assert abs(r["total_payoff"] - 11 / 60) < TOL

    # long/short leg decomposition: long leg is 2, short leg is 1.
    # long_contribution = w_long * r2[1] = 1.0 * 0.10 = 0.10
    # short_contribution = w_short * r1[1] = 1.0 * (-1/12)
    assert abs(r["daily_long_payoff"][1] - 0.10) < TOL
    assert abs(r["daily_short_payoff"][1] - (-1 / 12)) < TOL
    assert abs(r["daily_long_payoff"][1] - r["daily_short_payoff"][1] - expected_payoff_day2) < TOL


def test_episode2_weight_compounding_and_end_of_period_close():
    returns_1 = np.array([-0.20, 0.00, 0.30])
    returns_2 = np.array([0.00, 0.05, 0.02])
    r = simulate_pair_same_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert abs(r["spread"][1] - (-0.20)) < TOL
    assert len(r["trades"]) == 2

    open_ev, close_ev = r["trades"]
    assert open_ev["day"] == 1 and open_ev["direction"] == "long1_short2", \
        "spread<-threshold -> long the low leg (1), short the high one (2)"
    assert close_ev["day"] == 3 and close_ev["reason"] == "end_of_period", \
        "no natural crossing in this episode: closure MUST be forced at end of period"

    expected_day2 = -0.05           # 1.0*0.00 - 1.0*0.05
    expected_day3 = 0.30 - 1.05 * 0.02  # 0.279, weights compounded on the short leg
    assert abs(r["daily_payoff"][0] - 0.0) < TOL
    assert abs(r["daily_payoff"][1] - expected_day2) < TOL
    assert abs(r["daily_payoff"][2] - expected_day3) < TOL

    # long leg is 1, short leg is 2. Long contribution day3 = 1.0*0.30,
    # short contribution day3 = 1.05*0.02 (weight compounded from day2).
    assert abs(r["daily_long_payoff"][1] - 0.0) < TOL
    assert abs(r["daily_short_payoff"][1] - 0.05) < TOL
    assert abs(r["daily_long_payoff"][2] - 0.30) < TOL
    assert abs(r["daily_short_payoff"][2] - 1.05 * 0.02) < TOL
    assert abs(r["total_payoff"] - 0.229) < TOL


def test_no_signal_no_trade():
    """Spread below threshold for the whole period: zero trades, zero payoff."""
    returns_1 = np.array([0.01, -0.01, 0.005])
    returns_2 = np.array([0.00, 0.00, 0.00])
    r = simulate_pair_same_day(returns_1, returns_2, sigma=0.05, k=2.0)
    assert len(r["trades"]) == 0
    assert abs(r["total_payoff"]) < TOL


def test_wait_one_day_confirmed_signal_opens_next_day():
    """
    Signal on day1 (spread=0.20>0.10), CONFIRMED on day2 (spread still 0.20,
    unchanged) -> opens on day2 (not day1), no payoff on the opening day.
    Day3: large swing -> crossing and closure, payoff computed by hand with
    weights still 1.0 (only one day elapsed since opening).
      returns_1 = [+0.20, 0.00, -0.20], returns_2 = [0.00, 0.00, +0.30]
      Day1: P1=1.20, P2=1.00, spread=0.20 -> SIGNAL (does not open yet).
      Day2: P1=1.20, P2=1.00, spread=0.20 still > 0.10 -> CONFIRMED, opens
            long2/short1 on day2, weights=1.
      Day3: P1=1.20*0.80=0.96, P2=1.00*1.30=1.30, spread=-0.34 (sign flipped
            -> crossing). payoff = 1*r2 - 1*r1 = 0.30-(-0.20)=0.50.
    """
    returns_1 = np.array([0.20, 0.00, -0.20])
    returns_2 = np.array([0.00, 0.00, 0.30])
    r = simulate_pair_wait_one_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert len(r["trades"]) == 2, f"expected 2 events (open+close), found {r['trades']}"
    open_ev, close_ev = r["trades"]
    assert open_ev["event"] == "open" and open_ev["day"] == 2, \
        "opens on day2 (the day after the signal), not day1"
    assert open_ev["signal_day"] == 1, "the signal must be tracked on the day it was observed"
    assert open_ev["direction"] == "long2_short1"
    assert close_ev["event"] == "close" and close_ev["day"] == 3 and close_ev["reason"] == "crossing"

    assert abs(r["daily_payoff"][0] - 0.0) < TOL, "no payoff on pre-opening days"
    assert abs(r["daily_payoff"][1] - 0.0) < TOL, "no payoff on the opening day"
    assert abs(r["daily_payoff"][2] - 0.50) < TOL
    assert abs(r["total_payoff"] - 0.50) < TOL


def test_same_day_delisting_forces_close_at_last_valid_price():
    """
    Mid-period delisting (PROTOCOL.md §1.4/§2.2), OPEN position:
      returns_1 = [+0.20, +0.10, NaN, 0.00], returns_2 = [0.00, -0.05, 0.00, 0.00]
      sigma=0.05 (threshold=0.10).
      Day1: P1=1.20, P2=1.00, spread=0.20>0.10 -> opens long2/short1.
      Day2: r_long=r2=-0.05, r_short=r1=0.10.
            payoff = 1*(-0.05) - 1*(0.10) = -0.15.
            P1=1.20*1.10=1.32, P2=1.00*0.95=0.95, spread=0.37 (no natural
            crossing: same sign as before).
      Day3: returns_1[2]=NaN -> ticker 1 delists here. The last day with
            valid prices for BOTH legs is day2: the (still open) position is
            FORCIBLY closed there, reason="delisting", NOT on day3 nor at
            end of period (n=4). Day4 (after the delisting) is never
            processed at all (the loop stops at day2):
            daily_payoff[2]=daily_payoff[3]=0.
      Expected total = 0 (day1, opening) + (-0.15) (day2) = -0.15.
    """
    returns_1 = np.array([0.20, 0.10, np.nan, 0.00])
    returns_2 = np.array([0.00, -0.05, 0.00, 0.00])
    r = simulate_pair_same_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert len(r["trades"]) == 2
    open_ev, close_ev = r["trades"]
    assert open_ev["day"] == 1 and open_ev["direction"] == "long2_short1"
    assert close_ev["event"] == "close" and close_ev["day"] == 2, \
        "forced closure on day2 = last valid price, not day3 (NaN) nor end of period"
    assert close_ev["reason"] == "delisting"

    assert abs(r["daily_payoff"][0] - 0.0) < TOL
    assert abs(r["daily_payoff"][1] - (-0.15)) < TOL
    assert abs(r["daily_payoff"][2] - 0.0) < TOL, "day3 (post-delisting) not processed: payoff 0"
    assert abs(r["daily_payoff"][3] - 0.0) < TOL, "day4 (post-delisting) not processed: payoff 0"
    assert abs(r["total_payoff"] - (-0.15)) < TOL


def test_same_day_delisting_without_open_position_just_stops():
    """
    Delisting WITHOUT an open position (spread always below threshold before
    the NaN): the pair simply stops generating signals from that point,
    no trade, no payoff, no crash.
      returns_1 = [0.01, NaN, 0.01], returns_2 = [0.00, 0.00, 0.00], sigma=0.05.
    """
    returns_1 = np.array([0.01, np.nan, 0.01])
    returns_2 = np.array([0.00, 0.00, 0.00])
    r = simulate_pair_same_day(returns_1, returns_2, sigma=0.05, k=2.0)
    assert r["trades"] == []
    assert abs(r["total_payoff"]) < TOL


def test_wait_one_day_delisting_forces_close_at_last_valid_price():
    """
    Same scenario as test_same_day_delisting_forces_close_at_last_valid_price
    but with wait-one-day execution (confirmation one day after the signal):
      returns_1 = [+0.20, 0.00, +0.10, NaN], returns_2 = [0.00, 0.00, -0.05, 0.00]
      sigma=0.05 (threshold=0.10).
      Day1: spread=0.20>0.10 -> SIGNAL (does not open).
      Day2: spread unchanged (0.20, still >0.10) -> CONFIRMED, opens
            long2/short1 on day2 (no payoff, opening day).
      Day3: r_long=r2=-0.05, r_short=r1=0.10 -> payoff = -0.05-0.10=-0.15.
            spread=0.37, no natural crossing.
      Day4: returns_1[3]=NaN -> delisting. Last valid day = day3:
            forced closure there (reason="delisting"), not at end of period.
      Expected total = -0.15 (day3 only).
    """
    returns_1 = np.array([0.20, 0.00, 0.10, np.nan])
    returns_2 = np.array([0.00, 0.00, -0.05, 0.00])
    r = simulate_pair_wait_one_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert len(r["trades"]) == 2
    open_ev, close_ev = r["trades"]
    assert open_ev["day"] == 2 and open_ev["signal_day"] == 1
    assert close_ev["day"] == 3 and close_ev["reason"] == "delisting"

    assert abs(r["daily_payoff"][1] - 0.0) < TOL, "day2 = opening, no payoff"
    assert abs(r["daily_payoff"][2] - (-0.15)) < TOL
    assert abs(r["daily_payoff"][3] - 0.0) < TOL, "day4 (post-delisting) not processed"
    assert abs(r["total_payoff"] - (-0.15)) < TOL


def test_wait_one_day_delisting_without_open_position_just_stops():
    """Delisting WITHOUT an open position or an unresolved pending signal:
    no trade, no payoff, no crash."""
    returns_1 = np.array([0.01, np.nan, 0.01])
    returns_2 = np.array([0.00, 0.00, 0.00])
    r = simulate_pair_wait_one_day(returns_1, returns_2, sigma=0.05, k=2.0)
    assert r["trades"] == []
    assert abs(r["total_payoff"]) < TOL


def test_same_day_open_on_last_valid_day_closes_same_iteration():
    """
    Edge case found during the audit (not specific to delisting: it also
    arises at an ordinary end of period): the signal fires ONLY on the last
    valid day of the period (no following day to mark the position to
    market). Without a forced closure in the same iteration, the trade log
    would contain an "open" event NEVER matched by a "close" (is_open would
    stay True past the function's return) - a phantom trade that would
    corrupt any downstream diagnostics on round-trip/duration (PROTOCOL.md
    §2.3). Here n=2, signal only on day2:
      returns_1 = [0.00, 0.20], returns_2 = [0.00, 0.00], sigma=0.05.
      Day1: spread=0, no signal.
      Day2: P1=1.20, P2=1.00, spread=0.20>0.10 -> opens long2/short1 AND
            closes in the SAME iteration (end of period, zero duration).
    Expected: exactly 2 events (open+close, both on day2), zero total payoff
    (no following day to mark to market).
    """
    returns_1 = np.array([0.00, 0.20])
    returns_2 = np.array([0.00, 0.00])
    r = simulate_pair_same_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert len(r["trades"]) == 2, f"expected open AND close, not a phantom trade: {r['trades']}"
    open_ev, close_ev = r["trades"]
    assert open_ev["event"] == "open" and open_ev["day"] == 2
    assert close_ev["event"] == "close" and close_ev["day"] == 2
    assert close_ev["reason"] == "end_of_period"
    assert abs(r["total_payoff"]) < TOL


def test_wait_one_day_open_on_last_valid_day_closes_same_iteration():
    """
    Same edge case as test_same_day_open_on_last_valid_day_closes_same_iteration
    but with wait-one-day execution: the CONFIRMATION (not just the signal)
    falls on the last valid day.
      returns_1 = [0.00, 0.20, 0.20], returns_2 = [0.00, 0.00, 0.00], sigma=0.05.
      Day1: no signal. Day2: spread=0.20>0.10 -> SIGNAL (pending).
      Day3 (last day, n=3): spread still 0.20+0.20=0.44>0.10 ->
      CONFIRMED, opens AND closes in the same iteration (zero duration).
    """
    returns_1 = np.array([0.00, 0.20, 0.20])
    returns_2 = np.array([0.00, 0.00, 0.00])
    r = simulate_pair_wait_one_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert len(r["trades"]) == 2, f"expected open AND close, not a phantom trade: {r['trades']}"
    open_ev, close_ev = r["trades"]
    assert open_ev["event"] == "open" and open_ev["day"] == 3 and open_ev["signal_day"] == 2
    assert close_ev["event"] == "close" and close_ev["day"] == 3
    assert close_ev["reason"] == "end_of_period"
    assert abs(r["total_payoff"]) < TOL


def test_wait_one_day_missed_opportunity_reverts_before_execution():
    """
    Signal on day1 (spread=0.20>0.10), but on day2 the spread has already
    fallen back below the threshold (0.02 < 0.10, same sign) before
    execution -> the trade does NOT open: missed opportunity, "missed"
    event, zero payoff, no position ever opened.
      returns_1 = [+0.20, -0.15], returns_2 = [0.00, 0.00]
      Day1: P1=1.20, P2=1.00, spread=0.20 -> SIGNAL.
      Day2: P1=1.20*0.85=1.02, P2=1.00, spread=0.02 < threshold -> MISSED.
    """
    returns_1 = np.array([0.20, -0.15])
    returns_2 = np.array([0.00, 0.00])
    r = simulate_pair_wait_one_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert len(r["trades"]) == 1, f"expected only the missed event, found {r['trades']}"
    missed_ev = r["trades"][0]
    assert missed_ev["event"] == "missed" and missed_ev["day"] == 2
    assert missed_ev["signal_day"] == 1
    assert missed_ev["direction"] == "long2_short1"
    assert abs(r["total_payoff"]) < TOL, "missed opportunity: no position, no payoff"


if __name__ == "__main__":
    test_episode1_crossing_close()
    test_episode2_weight_compounding_and_end_of_period_close()
    test_no_signal_no_trade()
    test_wait_one_day_confirmed_signal_opens_next_day()
    test_wait_one_day_missed_opportunity_reverts_before_execution()
    test_same_day_delisting_forces_close_at_last_valid_price()
    test_same_day_delisting_without_open_position_just_stops()
    test_wait_one_day_delisting_forces_close_at_last_valid_price()
    test_wait_one_day_delisting_without_open_position_just_stops()
    test_same_day_open_on_last_valid_day_closes_same_iteration()
    test_wait_one_day_open_on_last_valid_day_closes_same_iteration()
    print("All synthetic tests PASSED.")
    print("  Episode 1 (crossing close):        total_payoff =", 11/60)
    print("  Episode 2 (end-of-period, compound): total_payoff =", 0.229)
    print("  Wait-one-day confirmed (open+close): total_payoff =", 0.50)
    print("  Wait-one-day missed opportunity:      total_payoff =", 0.0)
