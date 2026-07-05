"""
test_returns.py — aggregate_portfolio_run (committed/employed capital),
pair_employed_mask (episodi multipli), compound_to_monthly e
combine_overlapping_portfolios (media alla Jegadeesh-Titman), con numeri
calcolati a mano su un piccolo portafoglio fittizio.
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
    Un solo episodio: apre al giorno 2, chiude al giorno 4 (n_days=5).
    Employed = giorni (2,4] = {3,4} -> indici 0-based {2,3}.
    """
    trades = [
        {"event": "open", "day": 2},
        {"event": "close", "day": 4, "reason": "crossing"},
    ]
    mask = pair_employed_mask(trades, n_days=5)
    assert list(mask) == [False, False, True, True, False]


def test_pair_employed_mask_multiple_episodes():
    """
    Due episodi nello stesso run (riapertura dopo convergenza, PROTOCOL.md
    §2.1): open@1-close@2, poi open@3-close@5 (n_days=5).
    Employed atteso: giorno2 (episodio1) e giorni4,5 (episodio2).
    Indici 0-based: {1, 3, 4} -> [F,T,F,T,T].
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
    Portafoglio di 2 coppie (n_selected=2), n_days=2:
      Pair A: identica all'episodio1 di test_synthetic_pair.py (same-day).
        returns_1=[0.20,-1/12], returns_2=[0.00,0.10], sigma=0.05.
        Apre giorno1, chiude giorno2 per crossing. daily_payoff=[0, 11/60].
        trades=[open@1, close@2] -> employed = [False, True].
      Pair B: mai apre (spread sempre sotto soglia). daily_payoff=[0,0],
        trades=[] -> employed=[False, False].

    Attesi:
      payoff_sum = [0, 11/60]
      n_open     = [0, 1]
      committed_return = payoff_sum / 2 = [0, 11/120]
      employed_return  = [0.0 (n_open=0 -> 0, non NaN), (11/60)/1]
    """
    pair_A = simulate_pair_same_day(
        np.array([0.20, -1 / 12]), np.array([0.00, 0.10]), sigma=0.05, k=2.0
    )
    pair_B = simulate_pair_same_day(
        np.array([0.01, -0.01]), np.array([0.00, 0.00]), sigma=0.05, k=2.0
    )
    assert pair_B["trades"] == [], "pair B non deve mai aprire (pre-condizione del test)"

    agg = aggregate_portfolio_run({"A": pair_A, "B": pair_B}, n_days=2, n_selected=2)

    assert np.allclose(agg["payoff_sum"], [0.0, 11 / 60])
    assert list(agg["n_open"]) == [0, 1]
    assert np.allclose(agg["committed_return"], [0.0, 11 / 120])
    assert abs(agg["employed_return"].iloc[0] - 0.0) < TOL, "n_open=0 -> return 0.0, mai NaN"
    assert abs(agg["employed_return"].iloc[1] - 11 / 60) < TOL


def test_aggregate_portfolio_run_zero_selected_no_division_error():
    """n_selected=0 (nessuna coppia selezionabile, es. golden set troppo
    piccolo): committed_return deve essere zero ovunque, non un errore."""
    agg = aggregate_portfolio_run({}, n_days=3, n_selected=0)
    assert np.allclose(agg["committed_return"], [0.0, 0.0, 0.0])
    assert np.allclose(agg["employed_return"], [0.0, 0.0, 0.0])


def test_aggregate_portfolio_run_ignores_wait_one_day_missed_event():
    """
    Audit PROTOCOL.md §2.1 (esecuzione wait-one-day sempre presente accanto
    a same-day): un risultato con un evento "missed" (occasione persa,
    nessuna posizione aperta) deve aggregarsi come una coppia mai aperta —
    payoff zero, n_open zero, nessun crash su un evento diverso da open/close.
      returns_1=[0.20,-0.15], returns_2=[0.00,0.00], sigma=0.05 -> segnale
      al giorno1, rientrato sotto soglia al giorno2 -> trades=[missed].
    """
    pair_missed = simulate_pair_wait_one_day(
        np.array([0.20, -0.15]), np.array([0.00, 0.00]), sigma=0.05, k=2.0
    )
    assert pair_missed["trades"][0]["event"] == "missed", "pre-condizione del test"

    agg = aggregate_portfolio_run({"X": pair_missed}, n_days=2, n_selected=1)
    assert np.allclose(agg["payoff_sum"], [0.0, 0.0])
    assert list(agg["n_open"]) == [0, 0]
    assert np.allclose(agg["committed_return"], [0.0, 0.0])
    assert np.allclose(agg["employed_return"], [0.0, 0.0])


def test_compound_to_monthly_groups_by_calendar_month():
    """
    3 giorni: 2 in gennaio, 1 in febbraio.
      Gennaio: r = [0.01, 0.02] -> (1.01*1.02)-1 = 0.0302
      Febbraio: r = [-0.01] -> -0.01
    """
    daily = [0.01, 0.02, -0.01]
    dates = pd.to_datetime(["2020-01-15", "2020-01-20", "2020-02-05"])
    monthly = compound_to_monthly(daily, dates)

    assert list(monthly.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
    assert abs(monthly.iloc[0] - (1.01 * 1.02 - 1)) < TOL
    assert abs(monthly.iloc[1] - (-0.01)) < TOL


def test_combine_overlapping_portfolios_simple_average():
    """
    run1: Gen=0.01, Feb=0.02
    run2:        Feb=0.04, Mar=0.03
    Attesi: Gen=0.01 (solo run1), Feb=(0.02+0.04)/2=0.03 (media dei 2 attivi),
    Mar=0.03 (solo run2).
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
    test_aggregate_portfolio_run_zero_selected_no_division_error()
    test_aggregate_portfolio_run_ignores_wait_one_day_missed_event()
    test_compound_to_monthly_groups_by_calendar_month()
    test_combine_overlapping_portfolios_simple_average()
    print("test_returns: tutti i test PASSATI.")
