"""
trading.py — GGR engine: normalized price index, opening trigger, closing on
crossing, weight evolution (mark-to-market, buy-and-hold within the trade).
Two execution variants (PROTOCOL.md §2.1): SAME-DAY and WAIT-ONE-DAY.

Convention (PROTOCOL.md, §2.2 + task W2): both legs re-normalize to 1 on the
first day of the trading period; sigma is estimated ONLY on the formation
period and passed in as a frozen external input.
"""
from __future__ import annotations

import numpy as np


def build_price_index(returns: np.ndarray) -> np.ndarray:
    """P[0]=1 (re-normalization anchor); P[t]=P[t-1]*(1+returns[t-1])."""
    P = np.empty(len(returns) + 1)
    P[0] = 1.0
    for t, r in enumerate(returns, start=1):
        P[t] = P[t - 1] * (1 + r)
    return P


def _last_valid_day(returns_1: np.ndarray, returns_2: np.ndarray) -> int:
    """
    Last day (1-indexed) on which BOTH legs have a valid return (PROTOCOL.md
    §1.4/§2.2: a stock that delists mid-trading-period stops having prices,
    not stops generating noise).

    If returns_1 or returns_2 have a NaN starting at 0-indexed position m,
    the price for day m+1 is undefined for that leg: the last day with valid
    prices for BOTH legs is day m (1-indexed) — which numerically coincides
    with the 0-indexed position of the first NaN, because returns[j] is the
    return that carries P[j] to P[j+1].
    No NaN -> the whole period is valid, return n.
    """
    nan_mask = np.isnan(returns_1) | np.isnan(returns_2)
    if not nan_mask.any():
        return len(returns_1)
    return int(np.argmax(nan_mask))


def simulate_pair_same_day(
    returns_1: np.ndarray,
    returns_2: np.ndarray,
    sigma: float,
    k: float = 2.0,
) -> dict:
    """
    Simulates a GGR pair over the trading period, same-day execution.

    returns_1, returns_2: simple daily returns, index 0 = day 1 of the
        trading period (the return that carries the anchor P=1 to the
        first observed price).
    sigma: standard deviation of the spread estimated ON THE FORMATION
        period (external input).
    k: threshold in standard deviations (default 2, frozen by protocol).

    Mid-period delisting (a NaN in returns_1/returns_2 from a certain day
    onward, PROTOCOL.md §1.4/§2.2): if a position is open when prices run
    out, it is forcibly closed at the LAST valid price (event "close",
    reason="delisting"); if no position is open, the pair simply stops
    generating signals from that day onward (the loop stops there, no
    further trade is possible without valid prices for both legs).

    Returns: P1, P2, spread (index 0..n), daily_payoff (index 0..n-1,
    payoff realized on day t+1), daily_long_payoff/daily_short_payoff
    (each leg's contribution to the payoff, w_long*r_long and w_short*r_short:
    daily_payoff = daily_long_payoff - daily_short_payoff; used for the
    long/short alpha decomposition, PROTOCOL.md §2.4), trade log, cumulative
    payoff.
    """
    n = len(returns_1)
    assert len(returns_2) == n, "the two series must have the same length"
    last_day = _last_valid_day(returns_1, returns_2)

    P1 = build_price_index(returns_1)
    P2 = build_price_index(returns_2)
    spread = P1 - P2
    threshold = k * sigma

    is_open = False
    long_leg = None  # 1 or 2: which leg is long
    w_long = w_short = 0.0
    daily_payoff = np.zeros(n)
    daily_long_payoff = np.zeros(n)
    daily_short_payoff = np.zeros(n)
    trades: list[dict] = []

    for t in range(1, last_day + 1):
        if is_open:
            if long_leg == 2:
                r_long, r_short = returns_2[t - 1], returns_1[t - 1]
            else:
                r_long, r_short = returns_1[t - 1], returns_2[t - 1]
            long_contribution = w_long * r_long
            short_contribution = w_short * r_short
            daily_long_payoff[t - 1] = long_contribution
            daily_short_payoff[t - 1] = short_contribution
            daily_payoff[t - 1] = long_contribution - short_contribution
            w_long *= 1 + r_long
            w_short *= 1 + r_short

            crossed = (spread[t] == 0) or (np.sign(spread[t]) != np.sign(spread[t - 1]))
            if crossed or t == last_day:
                reason = "crossing" if crossed else ("delisting" if last_day < n else "end_of_period")
                trades.append({
                    "event": "close", "day": t, "spread": spread[t], "reason": reason,
                })
                is_open, long_leg = False, None
            continue

        if spread[t] > threshold:
            is_open, long_leg, w_long, w_short = True, 2, 1.0, 1.0
            trades.append({"event": "open", "day": t, "direction": "long2_short1", "spread": spread[t]})
        elif spread[t] < -threshold:
            is_open, long_leg, w_long, w_short = True, 1, 1.0, 1.0
            trades.append({"event": "open", "day": t, "direction": "long1_short2", "spread": spread[t]})

        if is_open and t == last_day:
            # Opening on the LAST valid day (end of period or delisting):
            # no following day to mark the position to market, so it closes
            # in the SAME iteration (zero duration, zero payoff) instead of
            # leaving an "open" without a matching "close" in the trade log
            # (bug: without this, is_open would stay True past the function's
            # return and downstream round-trip/duration would be computed on
            # a phantom trade that never closed).
            reason = "delisting" if last_day < n else "end_of_period"
            trades.append({"event": "close", "day": t, "spread": spread[t], "reason": reason})
            is_open, long_leg = False, None

    return {
        "P1": P1, "P2": P2, "spread": spread,
        "daily_payoff": daily_payoff,
        "daily_long_payoff": daily_long_payoff, "daily_short_payoff": daily_short_payoff,
        "trades": trades,
        "total_payoff": float(daily_payoff.sum()),
    }


def simulate_pair_wait_one_day(
    returns_1: np.ndarray,
    returns_2: np.ndarray,
    sigma: float,
    k: float = 2.0,
) -> dict:
    """
    Simulates a GGR pair over the trading period, wait-one-day execution.

    Signal observed on day t (|spread_t| > k*sigma); execution attempted on
    day t+1. If on day t+1 the spread has already fallen back below the
    threshold or crossed zero (i.e. it is no longer past the threshold in
    the same direction as the signal), the trade does NOT open: a missed
    opportunity, logged as a "missed" event (no payoff, no open position).
    Otherwise the position opens on day t+1 (weights=1, no payoff that day,
    same as a same-day opening) and from there on follows exactly the same
    mechanics as simulate_pair_same_day (mark-to-market, crossing, end of
    period).

    returns_1, returns_2, sigma, k: see simulate_pair_same_day.

    Mid-period delisting: same handling as simulate_pair_same_day (see its
    docstring) — forced closure at the last valid price if a position is
    open, otherwise the pair simply stops generating signals (and any
    "pending" signal awaiting confirmation silently expires, no differently
    from a signal never confirmed by the ordinary end of the period).

    Returns: same schema as simulate_pair_same_day (including
    daily_long_payoff/daily_short_payoff); "open" events also carry
    "signal_day" (the day the signal was observed, one day before opening).
    """
    n = len(returns_1)
    assert len(returns_2) == n, "the two series must have the same length"
    last_day = _last_valid_day(returns_1, returns_2)

    P1 = build_price_index(returns_1)
    P2 = build_price_index(returns_2)
    spread = P1 - P2
    threshold = k * sigma

    is_open = False
    long_leg = None  # 1 or 2: which leg is long
    w_long = w_short = 0.0
    pending: dict | None = None  # signal observed, awaiting execution the next day
    daily_payoff = np.zeros(n)
    daily_long_payoff = np.zeros(n)
    daily_short_payoff = np.zeros(n)
    trades: list[dict] = []

    for t in range(1, last_day + 1):
        if is_open:
            if long_leg == 2:
                r_long, r_short = returns_2[t - 1], returns_1[t - 1]
            else:
                r_long, r_short = returns_1[t - 1], returns_2[t - 1]
            long_contribution = w_long * r_long
            short_contribution = w_short * r_short
            daily_long_payoff[t - 1] = long_contribution
            daily_short_payoff[t - 1] = short_contribution
            daily_payoff[t - 1] = long_contribution - short_contribution
            w_long *= 1 + r_long
            w_short *= 1 + r_short

            crossed = (spread[t] == 0) or (np.sign(spread[t]) != np.sign(spread[t - 1]))
            if crossed or t == last_day:
                reason = "crossing" if crossed else ("delisting" if last_day < n else "end_of_period")
                trades.append({
                    "event": "close", "day": t, "spread": spread[t], "reason": reason,
                })
                is_open, long_leg = False, None
            continue

        if pending is not None:
            direction = pending["direction"]
            if direction == "long2_short1" and spread[t] > threshold:
                is_open, long_leg, w_long, w_short = True, 2, 1.0, 1.0
                trades.append({
                    "event": "open", "day": t, "direction": "long2_short1",
                    "spread": spread[t], "signal_day": pending["signal_day"],
                })
                pending = None
                if t == last_day:
                    # See comment in simulate_pair_same_day: opening on the
                    # last valid day -> closes in the same iteration, no
                    # "open" without a matching "close".
                    reason = "delisting" if last_day < n else "end_of_period"
                    trades.append({"event": "close", "day": t, "spread": spread[t], "reason": reason})
                    is_open, long_leg = False, None
                continue
            if direction == "long1_short2" and spread[t] < -threshold:
                is_open, long_leg, w_long, w_short = True, 1, 1.0, 1.0
                trades.append({
                    "event": "open", "day": t, "direction": "long1_short2",
                    "spread": spread[t], "signal_day": pending["signal_day"],
                })
                pending = None
                if t == last_day:
                    reason = "delisting" if last_day < n else "end_of_period"
                    trades.append({"event": "close", "day": t, "spread": spread[t], "reason": reason})
                    is_open, long_leg = False, None
                continue
            trades.append({
                "event": "missed", "day": t, "direction": direction,
                "spread": spread[t], "signal_day": pending["signal_day"],
            })
            pending = None
            # the day is not "consumed": the same bar can generate a new signal

        if spread[t] > threshold:
            pending = {"direction": "long2_short1", "signal_day": t}
        elif spread[t] < -threshold:
            pending = {"direction": "long1_short2", "signal_day": t}

    return {
        "P1": P1, "P2": P2, "spread": spread,
        "daily_payoff": daily_payoff,
        "daily_long_payoff": daily_long_payoff, "daily_short_payoff": daily_short_payoff,
        "trades": trades,
        "total_payoff": float(daily_payoff.sum()),
    }
