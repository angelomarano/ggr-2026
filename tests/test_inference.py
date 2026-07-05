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

from src.inference import factor_regression, newey_west_mean_test, stationary_bootstrap_ci

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


if __name__ == "__main__":
    test_newey_west_mean_matches_direct_statsmodels_call()
    test_newey_west_mean_sign_and_magnitude_sanity()
    test_stationary_bootstrap_constant_series_exact()
    test_stationary_bootstrap_reproducible_with_same_seed()
    test_stationary_bootstrap_ci_contains_sample_mean_on_symmetric_data()
    test_factor_regression_exact_recovery_without_noise()
    test_factor_regression_matches_direct_statsmodels_call_with_noise()
    test_factor_regression_drops_months_missing_in_factors()
    print("test_inference: tutti i test PASSATI.")
