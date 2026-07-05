"""
trading.py — Motore GGR: indice di prezzo normalizzato, trigger di apertura,
chiusura al crossing, evoluzione dei pesi (mark-to-market, buy-and-hold entro
il trade). Due varianti di esecuzione (PROTOCOL.md §2.1): SAME-DAY e
WAIT-ONE-DAY.

Convenzione (PROTOCOL.md, §2.2 + task W2): entrambe le gambe si
ri-normalizzano a 1 al primo giorno del trading period; sigma e' stimata
SOLO sul formation e passata come input esterno, congelata.
"""
from __future__ import annotations

import numpy as np


def build_price_index(returns: np.ndarray) -> np.ndarray:
    """P[0]=1 (ancora di rinormalizzazione); P[t]=P[t-1]*(1+returns[t-1])."""
    P = np.empty(len(returns) + 1)
    P[0] = 1.0
    for t, r in enumerate(returns, start=1):
        P[t] = P[t - 1] * (1 + r)
    return P


def simulate_pair_same_day(
    returns_1: np.ndarray,
    returns_2: np.ndarray,
    sigma: float,
    k: float = 2.0,
) -> dict:
    """
    Simula una coppia GGR sul trading period, esecuzione same-day.

    returns_1, returns_2: rendimenti giornalieri semplici, indice 0 = giorno
        1 del trading period (il rendimento che porta da P=1 dell'ancora
        al primo prezzo osservato).
    sigma: deviazione standard dello spread stimata SUL FORMATION (esterna).
    k: soglia in deviazioni standard (default 2, congelato da protocollo).

    Ritorna: P1, P2, spread (indice 0..n), daily_payoff (indice 0..n-1,
    payoff realizzato il giorno t+1), trade log, payoff cumulato.
    """
    n = len(returns_1)
    assert len(returns_2) == n, "le due serie devono avere la stessa lunghezza"

    P1 = build_price_index(returns_1)
    P2 = build_price_index(returns_2)
    spread = P1 - P2
    threshold = k * sigma

    is_open = False
    long_leg = None  # 1 o 2: quale gamba e' long
    w_long = w_short = 0.0
    daily_payoff = np.zeros(n)
    trades: list[dict] = []

    for t in range(1, n + 1):
        if is_open:
            if long_leg == 2:
                r_long, r_short = returns_2[t - 1], returns_1[t - 1]
            else:
                r_long, r_short = returns_1[t - 1], returns_2[t - 1]
            payoff = w_long * r_long - w_short * r_short
            daily_payoff[t - 1] = payoff
            w_long *= 1 + r_long
            w_short *= 1 + r_short

            crossed = (spread[t] == 0) or (np.sign(spread[t]) != np.sign(spread[t - 1]))
            if crossed or t == n:
                trades.append({
                    "event": "close", "day": t, "spread": spread[t],
                    "reason": "crossing" if crossed else "end_of_period",
                })
                is_open, long_leg = False, None
            continue

        if spread[t] > threshold:
            is_open, long_leg, w_long, w_short = True, 2, 1.0, 1.0
            trades.append({"event": "open", "day": t, "direction": "long2_short1", "spread": spread[t]})
        elif spread[t] < -threshold:
            is_open, long_leg, w_long, w_short = True, 1, 1.0, 1.0
            trades.append({"event": "open", "day": t, "direction": "long1_short2", "spread": spread[t]})

    return {
        "P1": P1, "P2": P2, "spread": spread,
        "daily_payoff": daily_payoff, "trades": trades,
        "total_payoff": float(daily_payoff.sum()),
    }


def simulate_pair_wait_one_day(
    returns_1: np.ndarray,
    returns_2: np.ndarray,
    sigma: float,
    k: float = 2.0,
) -> dict:
    """
    Simula una coppia GGR sul trading period, esecuzione wait-one-day.

    Segnale osservato al giorno t (|spread_t| > k*sigma); esecuzione tentata
    al giorno t+1. Se al giorno t+1 lo spread e' gia' rientrato sotto soglia
    o ha attraversato lo zero (cioe' non e' piu' oltre soglia nello stesso
    verso del segnale), il trade NON si apre: occasione persa, loggata come
    evento "missed" (nessun payoff, nessuna posizione aperta). Altrimenti la
    posizione apre al giorno t+1 (pesi=1, nessun payoff quel giorno, come
    all'apertura same-day) e da li' in poi segue esattamente la stessa
    meccanica di simulate_pair_same_day (mark-to-market, crossing, fine
    periodo).

    returns_1, returns_2, sigma, k: vedi simulate_pair_same_day.

    Ritorna: stesso schema di simulate_pair_same_day; gli eventi "open"
    portano anche "signal_day" (il giorno in cui il segnale e' stato
    osservato, un giorno prima dell'apertura).
    """
    n = len(returns_1)
    assert len(returns_2) == n, "le due serie devono avere la stessa lunghezza"

    P1 = build_price_index(returns_1)
    P2 = build_price_index(returns_2)
    spread = P1 - P2
    threshold = k * sigma

    is_open = False
    long_leg = None  # 1 o 2: quale gamba e' long
    w_long = w_short = 0.0
    pending: dict | None = None  # segnale osservato, in attesa di esecuzione il giorno dopo
    daily_payoff = np.zeros(n)
    trades: list[dict] = []

    for t in range(1, n + 1):
        if is_open:
            if long_leg == 2:
                r_long, r_short = returns_2[t - 1], returns_1[t - 1]
            else:
                r_long, r_short = returns_1[t - 1], returns_2[t - 1]
            payoff = w_long * r_long - w_short * r_short
            daily_payoff[t - 1] = payoff
            w_long *= 1 + r_long
            w_short *= 1 + r_short

            crossed = (spread[t] == 0) or (np.sign(spread[t]) != np.sign(spread[t - 1]))
            if crossed or t == n:
                trades.append({
                    "event": "close", "day": t, "spread": spread[t],
                    "reason": "crossing" if crossed else "end_of_period",
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
                continue
            if direction == "long1_short2" and spread[t] < -threshold:
                is_open, long_leg, w_long, w_short = True, 1, 1.0, 1.0
                trades.append({
                    "event": "open", "day": t, "direction": "long1_short2",
                    "spread": spread[t], "signal_day": pending["signal_day"],
                })
                pending = None
                continue
            trades.append({
                "event": "missed", "day": t, "direction": direction,
                "spread": spread[t], "signal_day": pending["signal_day"],
            })
            pending = None
            # non si "consuma" il giorno: la stessa barra puo' generare un nuovo segnale

        if spread[t] > threshold:
            pending = {"direction": "long2_short1", "signal_day": t}
        elif spread[t] < -threshold:
            pending = {"direction": "long1_short2", "signal_day": t}

    return {
        "P1": P1, "P2": P2, "spread": spread,
        "daily_payoff": daily_payoff, "trades": trades,
        "total_payoff": float(daily_payoff.sum()),
    }
