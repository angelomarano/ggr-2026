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

import data.prices as prices_mod
from src.formation import (
    load_formation_returns,
    load_trading_returns,
    normalized_price_indices,
    rank_pairs,
    select_portfolios,
    ssd_matrix,
)
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


def _write_fake_calendar_and_prices(tmp_path):
    """6 consecutive trading days. AAA has a full price history; BBB is
    missing its last 2 days (mid-window delisting); CCC has no file at all."""
    dates = pd.to_datetime(
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"]
    )
    pd.DataFrame({"Close": range(len(dates))}, index=dates).to_parquet(tmp_path / "_idx_GSPC.parquet")

    aaa = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=dates, name="Adj Close")
    pd.DataFrame({"Adj Close": aaa}).to_parquet(tmp_path / "AAA.parquet")

    bbb = pd.Series([50.0, 51.0, 52.0, 53.0, np.nan, np.nan], index=dates, name="Adj Close")
    pd.DataFrame({"Adj Close": bbb}).to_parquet(tmp_path / "BBB.parquet")

    return dates


def test_load_formation_returns_exact_row_count_and_completeness_filter(tmp_path, monkeypatch):
    """
    Requesting the window [dates[1], dates[5]] (5 reference days) must
    return exactly 5 rows, using dates[0] as the pct_change anchor - not 4
    (a naive reindex+pct_change over the window alone loses the first row).
    AAA has full history -> kept, 5 hand-computable returns. BBB has NaN in
    the last 2 days of the window -> excluded entirely (formation requires
    completeness). CCC has no cached file at all -> excluded, no crash.
    """
    dates = _write_fake_calendar_and_prices(tmp_path)
    monkeypatch.setattr(prices_mod, "RAW", tmp_path)

    with pytest.warns(UserWarning):
        returns = load_formation_returns(["AAA", "BBB", "CCC"], dates[1], dates[5], price_dir=tmp_path)

    assert len(returns) == 5, "5 reference days requested -> exactly 5 return rows, not 4"
    assert list(returns.columns) == ["AAA"], "BBB (incomplete) and CCC (no file) must be excluded"
    manual = [101 / 100 - 1, 102 / 101 - 1, 103 / 102 - 1, 104 / 103 - 1, 105 / 104 - 1]
    assert np.allclose(returns["AAA"].to_numpy(), manual)


def test_load_trading_returns_keeps_nan_for_delisting(tmp_path, monkeypatch):
    """
    Same fixture and window as the formation test above, but the trading
    loader must PRESERVE BBB's mid-window NaN instead of dropping the
    ticker - this is exactly the delisting case src/trading.py's
    simulate_pair_* functions are built to handle. CCC (no file) is still
    skipped, since there is nothing to simulate for it either way.
    """
    dates = _write_fake_calendar_and_prices(tmp_path)
    monkeypatch.setattr(prices_mod, "RAW", tmp_path)

    returns = load_trading_returns(["AAA", "BBB", "CCC"], dates[1], dates[5], price_dir=tmp_path)

    assert len(returns) == 5
    assert list(returns.columns) == ["AAA", "BBB"]
    assert not returns["BBB"].iloc[:3].isna().any(), "first 3 days of the window have valid BBB prices"
    assert returns["BBB"].iloc[3:].isna().all(), "last 2 days: BBB price missing -> NaN return, preserved"


def test_load_returns_window_raises_when_no_anchor_day_exists(tmp_path, monkeypatch):
    """
    Edge case hit in production (Gate 1 replication, run 2003-01): if the
    requested window starts on the very first date in the cached price
    history, there is no earlier trading day to anchor the first return.
    Both loaders must raise a clear ValueError rather than silently
    computing a wrong or truncated result.
    """
    dates = _write_fake_calendar_and_prices(tmp_path)
    monkeypatch.setattr(prices_mod, "RAW", tmp_path)

    with pytest.raises(ValueError, match="no reference trading day before"):
        load_formation_returns(["AAA"], dates[0], dates[5], price_dir=tmp_path)


if __name__ == "__main__":
    test_normalized_price_indices_matches_build_price_index()
    test_ssd_and_rank_hand_computed_with_tie_break()
    test_select_portfolios_small_universe_truncates_gracefully()
    test_select_portfolios_control_partially_populated()
    print("test_formation: pure tests OK (sigma=0, loader tests need pytest fixtures)")
