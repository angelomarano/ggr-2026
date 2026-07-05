"""
test_inference.py — newey_west_mean_test, stationary_bootstrap_ci,
factor_regression.

Nota sul metodo di verifica (vedi anche il modulo docstring di inference.py):
uno stimatore HAC (Newey-West) e' una somma pesata di autocovarianze
campionarie fino a `lags` ritardi con pesi di Bartlett — calcolarlo "a mano"
in un test significherebbe reimplementare la stessa formula che il modulo
gia' delega a statsmodels, senza validarla contro una verita' indipendente:
non aggiungerebbe nessuna garanzia. La verifica che sostituisce il calcolo a
mano qui e': (1) su un caso SENZA rumore, l'OLS deve recuperare i
coefficienti ESATTI (questo e' verificabile a mano, R^2=1); (2) su un caso
CON rumore, l'output della funzione deve coincidere con una chiamata DIRETTA
e indipendente a statsmodels con gli stessi parametri (stesso cov_type,
stessi maxlags) — questo verifica che il nostro wrapper passi i parametri
giusti e legga gli attributi giusti (params/bse/tvalues nell'ordine
corretto), che e' la superficie di bug plausibile in un wrapper, mentre la
correttezza della formula HAC in se' e' responsabilita' di statsmodels.
Lo stesso ragionamento si applica al block bootstrap: la sua distribuzione
e' intrinsecamente casuale, quindi si verifica (a) un caso degenere
(serie costante) il cui risultato E' calcolabile a mano esattamente, e (b)
la riproducibilita' bit-esatta a parita' di seed.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.inference import (
    decile_matched_bootstrap_pairs,
    decile_of_returns,
    factor_regression,
    long_short_leg_regression,
    newey_west_mean_test,
    stationary_bootstrap_ci,
)

TOL = 1e-9


def test_newey_west_mean_matches_direct_statsmodels_call():
    """Caso di riferimento: 24 rendimenti mensili pseudo-casuali (seed
    fisso). L'output di newey_west_mean_test deve coincidere ESATTAMENTE con
    una chiamata diretta a sm.OLS(y, costante).fit(cov_type='HAC', ...)."""
    rng = np.random.default_rng(42)
    y = rng.normal(0.005, 0.02, size=24)

    out = newey_west_mean_test(y, lags=6)

    ref = sm.OLS(y, np.ones((len(y), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    assert abs(out["mean"] - ref.params[0]) < TOL
    assert abs(out["se"] - ref.bse[0]) < TOL
    assert abs(out["t_stat"] - ref.tvalues[0]) < TOL
    assert abs(out["p_value"] - ref.pvalues[0]) < TOL
    assert out["n"] == 24


def test_newey_west_mean_sign_and_magnitude_sanity():
    """Serie con media chiaramente positiva e bassa varianza -> t-stat
    grande e positivo (sanity check indipendente dal confronto diretto)."""
    y = np.array([0.01, 0.012, 0.009, 0.011, 0.010, 0.0105, 0.0095, 0.0115] * 3)
    out = newey_west_mean_test(y, lags=6)
    assert out["mean"] > 0.009
    assert out["t_stat"] > 5


def test_stationary_bootstrap_constant_series_exact():
    """
    Caso degenere calcolabile A MANO: se la serie e' costante (tutti i
    valori = c), QUALUNQUE ricampionamento (qualunque blocco, qualunque
    indice) produce una media ricampionata identica a c. Quindi l'IC deve
    collassare esattamente a un punto: ci_low == ci_high == mean == c, per
    costruzione, indipendentemente dalla casualita' del bootstrap.
    """
    c = 0.0123
    y = np.full(20, c)
    out = stationary_bootstrap_ci(y, mean_block_months=6, n_reps=200, seed=1)
    assert abs(out["mean"] - c) < TOL
    assert abs(out["ci_low"] - c) < TOL
    assert abs(out["ci_high"] - c) < TOL


def test_stationary_bootstrap_reproducible_with_same_seed():
    """A parita' di seed, la replica bootstrap deve essere bit-riproducibile
    (nessuna sorgente di casualita' non seedata)."""
    rng = np.random.default_rng(7)
    y = rng.normal(0.01, 0.03, size=36)
    out1 = stationary_bootstrap_ci(y, n_reps=500, seed=123)
    out2 = stationary_bootstrap_ci(y, n_reps=500, seed=123)
    assert out1["ci_low"] == out2["ci_low"]
    assert out1["ci_high"] == out2["ci_high"]
    assert np.array_equal(out1["boot_means"], out2["boot_means"])


def test_stationary_bootstrap_ci_contains_sample_mean_on_symmetric_data():
    """Sanity check strutturale (non un calcolo a mano): su dati simmetrici
    a bassa asimmetria e abbastanza repliche, l'IC percentile deve contenere
    la media campionaria."""
    rng = np.random.default_rng(99)
    y = rng.normal(0.01, 0.02, size=48)
    out = stationary_bootstrap_ci(y, n_reps=2000, seed=99)
    assert out["ci_low"] < out["mean"] < out["ci_high"]


def test_factor_regression_exact_recovery_without_noise():
    """
    Caso SENZA rumore, calcolabile a mano: y = alpha_true + sum(beta_true_k *
    factor_k), ESATTAMENTE, per 30 mesi. L'OLS deve recuperare alpha e i
    loadings ESATTI (entro tolleranza numerica) e R^2 = 1.
    """
    rng = np.random.default_rng(5)
    months = pd.date_range("2010-01-01", periods=30, freq="MS")
    factor_cols = ["Mkt-RF", "SMB", "HML", "Mom", "ST_Rev"]
    factors = pd.DataFrame(
        rng.normal(0, 0.02, size=(30, 5)), index=months, columns=factor_cols
    )
    alpha_true = 0.004
    betas_true = {"Mkt-RF": 0.8, "SMB": 0.3, "HML": -0.2, "Mom": 0.1, "ST_Rev": -0.4}
    y = pd.Series(
        alpha_true + sum(betas_true[c] * factors[c] for c in factor_cols), index=months
    )

    out = factor_regression(y, factors, factor_cols=tuple(factor_cols))

    assert abs(out["alpha"] - alpha_true) < 1e-8
    for c in factor_cols:
        assert abs(out["loadings"][c] - betas_true[c]) < 1e-8
    assert abs(out["r_squared"] - 1.0) < 1e-8
    assert out["n_obs"] == 30


def test_factor_regression_matches_direct_statsmodels_call_with_noise():
    """Caso CON rumore: l'output deve coincidere con una chiamata diretta e
    indipendente a statsmodels (stesso motivo del test NW: verifica il
    wiring del wrapper, non la formula HAC in se')."""
    rng = np.random.default_rng(11)
    months = pd.date_range("2015-01-01", periods=40, freq="MS")
    factor_cols = ["Mkt-RF", "SMB", "HML", "Mom", "ST_Rev"]
    factors = pd.DataFrame(
        rng.normal(0, 0.02, size=(40, 5)), index=months, columns=factor_cols
    )
    betas_true = {"Mkt-RF": 0.6, "SMB": -0.1, "HML": 0.2, "Mom": 0.05, "ST_Rev": -0.3}
    noise = rng.normal(0, 0.001, size=40)
    y = pd.Series(
        0.002 + sum(betas_true[c] * factors[c] for c in factor_cols) + noise, index=months
    )

    out = factor_regression(y, factors, factor_cols=tuple(factor_cols))

    X = sm.add_constant(factors[factor_cols].to_numpy())
    ref = sm.OLS(y.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

    assert abs(out["alpha"] - ref.params[0]) < TOL
    assert abs(out["alpha_t"] - ref.tvalues[0]) < TOL
    for i, c in enumerate(factor_cols, start=1):
        assert abs(out["loadings"][c] - ref.params[i]) < TOL
        assert abs(out["loadings_t"][c] - ref.tvalues[i]) < TOL


def test_factor_regression_drops_months_missing_in_factors():
    """Due mesi presenti in excess_returns ma assenti in factors (Ken French
    non ancora aggiornata) vanno scartati (inner join), non devono produrre
    NaN ne' un errore. Dati non degeneri (10 mesi con fattori che variano
    davvero) per evitare un sistema rank-deficient come nel test precedente
    a rumore zero, qui serve solo a verificare l'allineamento, non il fit."""
    rng = np.random.default_rng(3)
    months = pd.date_range("2020-01-01", periods=10, freq="MS")
    factor_cols = ["Mkt-RF", "SMB", "HML", "Mom", "ST_Rev"]
    factors = pd.DataFrame(
        rng.normal(0, 0.02, size=(8, 5)), index=months[:8], columns=factor_cols
    )  # mancano gli ultimi 2 mesi
    y = pd.Series(rng.normal(0.01, 0.005, size=10), index=months)

    out = factor_regression(y, factors, factor_cols=tuple(factor_cols))
    assert out["n_obs"] == 8
    assert not np.isnan(out["alpha"])
    assert not np.isnan(out["r_squared"])


def test_long_short_leg_regression_recovers_known_drift():
    """
    Decomposizione alpha long/short (PROTOCOL.md §2.4, punto 2): costruita
    con un drift NOTO per costruzione, entro tolleranza (non esatto: c'e'
    rumore, quindi non e' un caso a mano come il recupero esatto senza
    rumore di factor_regression, ma il drift iniettato e' comunque noto e
    la stima deve avvicinarvisi entro una tolleranza dettata dall'errore
    standard atteso, non entro epsilon numerico).

    200 mesi, rumore idiosincratico std=0.008 (=> SE atteso della media
    ~0.008/sqrt(200)=0.00057, la tolleranza 0.0015 e' ~2.6 SE, ampiamente
    sufficiente per un seed fisso):
      long_returns  = rumore puro attorno a +0.0002 (alpha atteso ~0,
                      "non significativo" nello spirito di GGR Tabella 7)
      short_returns = rumore attorno a -0.005 (drift negativo NOTO, atteso
                      "significativo": e' il leg che genera il profitto
                      quando lo si shorta)
    """
    rng = np.random.default_rng(21)
    months = pd.date_range("2000-01-01", periods=200, freq="MS")
    factor_cols = ["Mkt-RF", "SMB", "HML", "Mom", "ST_Rev"]
    factors = pd.DataFrame(
        rng.normal(0, 0.02, size=(200, 5)), index=months, columns=factor_cols
    )

    long_drift = 0.0002
    short_drift = -0.005
    long_returns = pd.Series(long_drift + rng.normal(0, 0.008, size=200), index=months)
    short_returns = pd.Series(short_drift + rng.normal(0, 0.008, size=200), index=months)

    out = long_short_leg_regression(long_returns, short_returns, factors, factor_cols=tuple(factor_cols))

    assert abs(out["long"]["alpha"] - long_drift) < 0.0015
    assert abs(out["short"]["alpha"] - short_drift) < 0.0015
    assert out["short"]["alpha"] < out["long"]["alpha"], \
        "la gamba short deve mostrare il drift negativo, la long deve restare vicina a zero"
    assert out["long"]["n_obs"] == 200 and out["short"]["n_obs"] == 200


def test_decile_of_returns_hand_computed():
    """
    Universo fittizio di 20 titoli, rendimenti del mese precedente = 0..19
    (equispaziati) -> con 10 decili, pd.qcut deve assegnare ESATTAMENTE 2
    titoli per decile (T00,T01)->decile1, (T02,T03)->decile2, ...,
    (T18,T19)->decile10. Decili noti a mano, nessuna ambiguita'.
    """
    tickers = [f"T{i:02d}" for i in range(20)]
    prior_returns = pd.Series(np.arange(20, dtype=float), index=tickers)
    deciles = decile_of_returns(prior_returns, n_deciles=10)

    assert (deciles.value_counts() == 2).all(), "10 decili x 2 titoli ciascuno, nessuno sbilanciato"
    assert deciles.loc["T00"] == deciles.loc["T01"] == 1
    assert deciles.loc["T18"] == deciles.loc["T19"] == 10
    assert deciles.loc["T00"] != deciles.loc["T02"], "decili adiacenti devono restare distinti"


def test_decile_matched_bootstrap_respects_decile_constraint():
    """
    Falsificazione bootstrap (PROTOCOL.md §2.4, punto 1): su 50 repliche di
    2 coppie vere, OGNI titolo fittizio sostituito deve appartenere ALLO
    STESSO decile del titolo vero che rimpiazza — verificato su TUTTE le
    repliche, non solo in media (il vincolo e' per costruzione, deve valere
    sempre, non statisticamente).
    """
    tickers = [f"T{i:02d}" for i in range(20)]
    prior_returns = pd.Series(np.arange(20, dtype=float), index=tickers)
    deciles = decile_of_returns(prior_returns, n_deciles=10)

    selected_pairs = [("T00", "T19"), ("T05", "T14")]  # decile1 e decile10; decile3 e decile8
    reps = decile_matched_bootstrap_pairs(
        selected_pairs, prior_returns, n_deciles=10, n_reps=50, seed=0
    )

    assert len(reps) == 50
    for rep in reps:
        assert len(rep) == len(selected_pairs)
        for (t1_true, t2_true), (t1_fake, t2_fake) in zip(selected_pairs, rep):
            assert deciles.loc[t1_fake] == deciles.loc[t1_true], \
                f"{t1_fake} non e' nello stesso decile di {t1_true}"
            assert deciles.loc[t2_fake] == deciles.loc[t2_true], \
                f"{t2_fake} non e' nello stesso decile di {t2_true}"


def test_decile_matched_bootstrap_reproducible_with_same_seed():
    """A parita' di seed, l'assegnazione dei titoli fittizi deve essere
    bit-riproducibile (nessuna sorgente di casualita' non seedata)."""
    tickers = [f"T{i:02d}" for i in range(20)]
    prior_returns = pd.Series(np.arange(20, dtype=float), index=tickers)
    selected_pairs = [("T00", "T19"), ("T05", "T14")]

    reps1 = decile_matched_bootstrap_pairs(selected_pairs, prior_returns, n_reps=30, seed=42)
    reps2 = decile_matched_bootstrap_pairs(selected_pairs, prior_returns, n_reps=30, seed=42)
    assert reps1 == reps2


def test_decile_matched_bootstrap_singleton_decile_returns_itself():
    """Caso limite: se n_deciles == n_titoli, ogni titolo e' l'unico membro
    del proprio decile -> l'unica sostituzione possibile e' il titolo
    stesso, nessun crash (nessuna alternativa nel pool)."""
    tickers = [f"T{i:02d}" for i in range(10)]
    prior_returns = pd.Series(np.arange(10, dtype=float), index=tickers)
    selected_pairs = [("T02", "T07")]

    reps = decile_matched_bootstrap_pairs(
        selected_pairs, prior_returns, n_deciles=10, n_reps=20, seed=1
    )
    assert all(rep == [("T02", "T07")] for rep in reps)


if __name__ == "__main__":
    test_newey_west_mean_matches_direct_statsmodels_call()
    test_newey_west_mean_sign_and_magnitude_sanity()
    test_stationary_bootstrap_constant_series_exact()
    test_stationary_bootstrap_reproducible_with_same_seed()
    test_stationary_bootstrap_ci_contains_sample_mean_on_symmetric_data()
    test_factor_regression_exact_recovery_without_noise()
    test_factor_regression_matches_direct_statsmodels_call_with_noise()
    test_factor_regression_drops_months_missing_in_factors()
    test_long_short_leg_regression_recovers_known_drift()
    test_decile_of_returns_hand_computed()
    test_decile_matched_bootstrap_respects_decile_constraint()
    test_decile_matched_bootstrap_reproducible_with_same_seed()
    test_decile_matched_bootstrap_singleton_decile_returns_itself()
    print("test_inference: tutti i test PASSATI.")
