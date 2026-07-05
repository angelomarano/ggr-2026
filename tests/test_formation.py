"""
test_formation.py — normalized_price_indices, ssd_matrix, rank_pairs (tie-break
deterministico, esclusione sigma=0) e select_portfolios (universo piccolo),
su un piccolo universo fittizio con SSD calcolate a mano.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pytest

from src.formation import normalized_price_indices, rank_pairs, select_portfolios, ssd_matrix
from src.trading import build_price_index

TOL = 1e-9


def test_normalized_price_indices_matches_build_price_index():
    """Ogni colonna di normalized_price_indices deve coincidere ESATTAMENTE
    con build_price_index applicata alla stessa serie (riuso, non duplicazione)."""
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [0.0, 0.05, -0.01]})
    out = normalized_price_indices(returns)
    assert np.allclose(out["A"].to_numpy(), build_price_index(returns["A"].to_numpy()))
    assert np.allclose(out["B"].to_numpy(), build_price_index(returns["B"].to_numpy()))
    assert out["A"].iloc[0] == 1.0 and out["B"].iloc[0] == 1.0


def test_ssd_and_rank_hand_computed_with_tie_break():
    """
    3 titoli, 1 giorno di formation, r=25% (0.25 = 2^-2, esattamente
    rappresentabile in float64: garantisce un TIE bit-esatto, non solo
    numerico, tra SSD(A,B) e SSD(A,C)):
      A: [0.00]  -> P*_A = [1, 1.00]
      B: [+0.25] -> P*_B = [1, 1.25]
      C: [-0.25] -> P*_C = [1, 0.75]

    SSD(A,B) = 0^2 + (1.00-1.25)^2 = 0.0625
    SSD(A,C) = 0^2 + (1.00-0.75)^2 = 0.0625   (TIE esatto con A,B)
    SSD(B,C) = 0^2 + (1.25-0.75)^2 = 0.25

    sigma(A,B) = std([0,-0.25], ddof=0) = 0.125
    sigma(A,C) = std([0, 0.25], ddof=0) = 0.125
    sigma(B,C) = std([0, 0.50], ddof=0) = 0.25

    Tie-break dichiarato (docstring formation.py): a parita' di ssd, ordine
    alfabetico su (ticker_1, ticker_2) -> (A,B) prima di (A,C).
    Rank atteso: 1=(A,B), 2=(A,C), 3=(B,C).
    """
    returns = pd.DataFrame({"A": [0.00], "B": [0.25], "C": [-0.25]})
    price_index = normalized_price_indices(returns)

    ssd = ssd_matrix(price_index)
    assert abs(ssd.loc["A", "B"] - 0.0625) < TOL
    assert abs(ssd.loc["A", "C"] - 0.0625) < TOL
    assert abs(ssd.loc["B", "C"] - 0.25) < TOL
    assert ssd.loc["A", "B"] == ssd.loc["A", "C"], "il tie deve essere bit-esatto, non solo entro tolleranza"

    ranked = rank_pairs(price_index)
    assert len(ranked) == 3, "3 titoli -> C(3,2)=3 coppie, nessuna esclusa (sigma!=0 ovunque)"

    r1, r2, r3 = ranked.loc[1], ranked.loc[2], ranked.loc[3]
    assert (r1["ticker_1"], r1["ticker_2"]) == ("A", "B")
    assert (r2["ticker_1"], r2["ticker_2"]) == ("A", "C")
    assert (r3["ticker_1"], r3["ticker_2"]) == ("B", "C")
    assert abs(r1["sigma"] - 0.125) < TOL
    assert abs(r2["sigma"] - 0.125) < TOL
    assert abs(r3["sigma"] - 0.25) < TOL

    # "un titolo puo' comparire in piu' coppie" (PROTOCOL.md §2.1): nessuna
    # deduplicazione, A compare sia in rank1 che in rank2.
    assert (ranked["ticker_1"] == "A").sum() == 2


def test_sigma_zero_pair_excluded_with_warning():
    """
    D ed E hanno rendimenti IDENTICI -> spread D-E costante a zero in tutto
    il formation -> sigma=0 -> la coppia (D,E) va esclusa dal ranking con un
    warning esplicito (non deve propagare una soglia degenere k*0=0 a valle).
    F ha rendimenti diversi: le coppie (D,F) ed (E,F) restano nel ranking.
    """
    returns = pd.DataFrame({
        "D": [0.01, -0.02, 0.03],
        "E": [0.01, -0.02, 0.03],
        "F": [0.05, 0.01, -0.04],
    })
    price_index = normalized_price_indices(returns)

    with pytest.warns(UserWarning, match="sigma"):
        ranked = rank_pairs(price_index)

    pairs = set(zip(ranked["ticker_1"], ranked["ticker_2"]))
    assert ("D", "E") not in pairs, "coppia sigma=0 deve essere esclusa dal ranking"
    assert ("D", "F") in pairs and ("E", "F") in pairs
    assert len(ranked) == 2


def test_select_portfolios_small_universe_truncates_gracefully():
    """
    Universo di 4 titoli -> C(4,2)=6 coppie candidate totali, ben sotto i 20
    del portafoglio primario e ben sotto le 120 del controllo. Verifica che
    select_portfolios NON sollevi eccezioni e restituisca sotto-tabelle piu'
    corte del nominale invece di un errore o un padding artificiale:
      - top_5: al massimo 5 righe (qui 5, disponibili)
      - top_20: troncato a 6 (tutte le coppie disponibili, < 20 richieste)
      - control (101-120): VUOTO, nessuna coppia arriva a rank>=101
    """
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({
        t: rng.normal(0, 0.02, size=20) for t in ["W", "X", "Y", "Z"]
    })
    price_index = normalized_price_indices(returns)
    portfolios = select_portfolios(price_index)

    assert len(portfolios["top_5"]) == 5
    assert len(portfolios["top_20"]) == 6, "solo 6 coppie disponibili su un universo di 4 titoli"
    assert len(portfolios["control"]) == 0, "nessuna coppia raggiunge rank 101-120 su soli 6 candidati"
    for key in ("top_5", "top_20", "control"):
        assert "sigma" in portfolios[key].columns


def test_select_portfolios_control_partially_populated():
    """
    Caso limite intermedio tra "pieno" e "vuoto" (audit PROTOCOL.md §2.1):
    15 titoli -> C(15,2)=105 coppie candidate. top_20 e' pieno (20 righe),
    ma control (101-120) e' PARZIALMENTE popolato: solo i rank 101-105
    esistono (5 righe), non 20 e non 0. Nessun crash, nessun padding.
    """
    rng = np.random.default_rng(1)
    returns = pd.DataFrame({f"T{i:02d}": rng.normal(0, 0.02, size=20) for i in range(15)})
    price_index = normalized_price_indices(returns)
    portfolios = select_portfolios(price_index)

    assert len(portfolios["top_20"]) == 20
    assert len(portfolios["control"]) == 5, "105 coppie totali -> solo rank 101-105 esistono"
    assert list(portfolios["control"].index) == [101, 102, 103, 104, 105]


if __name__ == "__main__":
    test_normalized_price_indices_matches_build_price_index()
    test_ssd_and_rank_hand_computed_with_tie_break()
    test_select_portfolios_small_universe_truncates_gracefully()
    test_select_portfolios_control_partially_populated()
    print("test_formation: test puri OK (il test sigma=0 richiede pytest.warns)")
