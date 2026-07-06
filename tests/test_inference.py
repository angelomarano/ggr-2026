"""
test_inference.py — newey_west_mean_test, stationary_bootstrap_ci,
factor_regression.

Note on the verification method (see also inference.py's module docstring):
a HAC (Newey-West) estimator is a weighted sum of sample autocovariances up
to `lags` lags with Bartlett weights — computing it "by hand" in a test
would mean reimplementing the same formula the module already delegates to
statsmodels, without validating it against an independent truth: it
wouldn't add any guarantee. The verification used here instead is: (1) on a
NOISE-FREE case, the OLS must recover the EXACT coefficients (this is
verifiable by hand, R^2=1); (2) on a case WITH noise, the function's output
must match a DIRECT, independent call to statsmodels with the same
parameters (same cov_type, same maxlags) — this verifies that our wrapper
passes the right parameters and reads the right attributes (params/bse/
tvalues in the correct order), which is the plausible bug surface in a
wrapper, while the correctness of the HAC formula itself is statsmodels'
responsibility. The same reasoning applies to the block bootstrap: its
distribution is inherently random, so we verify (a) a degenerate case
(constant series) whose result CAN be computed exactly by hand, and (b)
bit-exact reproducibility for the same seed.
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
    """Reference case: 24 pseudo-random monthly returns (fixed seed). The
    output of newey_west_mean_test must EXACTLY match a direct call to
    sm.OLS(y, constant).fit(cov_type='HAC', ...)."""
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
    """Series with clearly positive mean and low variance -> large positive
    t-stat (sanity check independent of the direct comparison)."""
    y = np.array([0.01, 0.012, 0.009, 0.011, 0.010, 0.0105, 0.0095, 0.0115] * 3)
    out = newey_west_mean_test(y, lags=6)
    assert out["mean"] > 0.009
    assert out["t_stat"] > 5


def test_stationary_bootstrap_constant_series_exact():
    """
    Degenerate case computable BY HAND: if the series is constant (all
    values = c), ANY resampling (any block, any index) produces a
    resampled mean identical to c. So the CI must collapse exactly to a
    point: ci_low == ci_high == mean == c, by construction, regardless of
    the bootstrap's randomness.
    """
    c = 0.0123
    y = np.full(20, c)
    out = stationary_bootstrap_ci(y, mean_block_months=6, n_reps=200, seed=1)
    assert abs(out["mean"] - c) < TOL
    assert abs(out["ci_low"] - c) < TOL
    assert abs(out["ci_high"] - c) < TOL


def test_stationary_bootstrap_reproducible_with_same_seed():
    """For the same seed, the bootstrap replication must be bit-reproducible
    (no unseeded source of randomness)."""
    rng = np.random.default_rng(7)
    y = rng.normal(0.01, 0.03, size=36)
    out1 = stationary_bootstrap_ci(y, n_reps=500, seed=123)
    out2 = stationary_bootstrap_ci(y, n_reps=500, seed=123)
    assert out1["ci_low"] == out2["ci_low"]
    assert out1["ci_high"] == out2["ci_high"]
    assert np.array_equal(out1["boot_means"], out2["boot_means"])


def test_stationary_bootstrap_ci_contains_sample_mean_on_symmetric_data():
    """Structural sanity check (not a hand computation): on symmetric,
    low-skew data with enough replications, the percentile CI must contain
    the sample mean."""
    rng = np.random.default_rng(99)
    y = rng.normal(0.01, 0.02, size=48)
    out = stationary_bootstrap_ci(y, n_reps=2000, seed=99)
    assert out["ci_low"] < out["mean"] < out["ci_high"]


def test_factor_regression_exact_recovery_without_noise():
    """
    NOISE-FREE case, computable by hand: y = alpha_true + sum(beta_true_k *
    factor_k), EXACTLY, for 30 months. The OLS must recover the EXACT alpha
    and loadings (within numerical tolerance) and R^2 = 1.
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
    """Case WITH noise: the output must match a direct, independent call to
    statsmodels (same reasoning as the NW test: verifies the wrapper's
    wiring, not the HAC formula itself)."""
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
    """Two months present in excess_returns but absent from factors (Ken
    French not yet updated) must be dropped (inner join), and must not
    produce a NaN or an error. Non-degenerate data (10 months with factors
    that actually vary) to avoid a rank-deficient system like the previous
    zero-noise test; here it's only meant to verify the alignment, not the
    fit."""
    rng = np.random.default_rng(3)
    months = pd.date_range("2020-01-01", periods=10, freq="MS")
    factor_cols = ["Mkt-RF", "SMB", "HML", "Mom", "ST_Rev"]
    factors = pd.DataFrame(
        rng.normal(0, 0.02, size=(8, 5)), index=months[:8], columns=factor_cols
    )  # last 2 months are missing
    y = pd.Series(rng.normal(0.01, 0.005, size=10), index=months)

    out = factor_regression(y, factors, factor_cols=tuple(factor_cols))
    assert out["n_obs"] == 8
    assert not np.isnan(out["alpha"])
    assert not np.isnan(out["r_squared"])


def test_long_short_leg_regression_recovers_known_drift():
    """
    Long/short alpha decomposition (PROTOCOL.md §2.4, point 2): built with a
    drift KNOWN by construction, within tolerance (not exact: there is
    noise, so this isn't a hand-computed case like factor_regression's
    exact noise-free recovery, but the injected drift is still known and
    the estimate must approach it within a tolerance dictated by the
    expected standard error, not within numerical epsilon).

    200 months, idiosyncratic noise std=0.008 (=> expected SE of the mean
    ~0.008/sqrt(200)=0.00057, the 0.0015 tolerance is ~2.6 SE, comfortably
    enough for a fixed seed):
      long_returns  = pure noise around +0.0002 (alpha expected ~0,
                      "not significant" in the spirit of GGR Table 7)
      short_returns = noise around -0.005 (KNOWN negative drift, expected
                      to be "significant": it's the leg that generates the
                      profit when shorted)
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
        "the short leg must show the negative drift, the long leg must stay near zero"
    assert out["long"]["n_obs"] == 200 and out["short"]["n_obs"] == 200


def test_decile_of_returns_hand_computed():
    """
    Fictitious universe of 20 tickers, prior-month returns = 0..19
    (evenly spaced) -> with 10 deciles, pd.qcut must assign EXACTLY 2
    tickers per decile (T00,T01)->decile1, (T02,T03)->decile2, ...,
    (T18,T19)->decile10. Deciles known by hand, no ambiguity.
    """
    tickers = [f"T{i:02d}" for i in range(20)]
    prior_returns = pd.Series(np.arange(20, dtype=float), index=tickers)
    deciles = decile_of_returns(prior_returns, n_deciles=10)

    assert (deciles.value_counts() == 2).all(), "10 deciles x 2 tickers each, none unbalanced"
    assert deciles.loc["T00"] == deciles.loc["T01"] == 1
    assert deciles.loc["T18"] == deciles.loc["T19"] == 10
    assert deciles.loc["T00"] != deciles.loc["T02"], "adjacent deciles must stay distinct"


def test_decile_matched_bootstrap_respects_decile_constraint():
    """
    Bootstrap falsification (PROTOCOL.md §2.4, point 1): over 50
    replications of 2 real pairs, EVERY substituted fictitious ticker must
    belong to THE SAME decile as the real ticker it replaces — verified on
    ALL replications, not just on average (the constraint holds by
    construction, it must always hold, not just statistically).
    """
    tickers = [f"T{i:02d}" for i in range(20)]
    prior_returns = pd.Series(np.arange(20, dtype=float), index=tickers)
    deciles = decile_of_returns(prior_returns, n_deciles=10)

    selected_pairs = [("T00", "T19"), ("T05", "T14")]  # decile1 & decile10; decile3 & decile8
    reps = decile_matched_bootstrap_pairs(
        selected_pairs, prior_returns, n_deciles=10, n_reps=50, seed=0
    )

    assert len(reps) == 50
    for rep in reps:
        assert len(rep) == len(selected_pairs)
        for (t1_true, t2_true), (t1_fake, t2_fake) in zip(selected_pairs, rep):
            assert deciles.loc[t1_fake] == deciles.loc[t1_true], \
                f"{t1_fake} is not in the same decile as {t1_true}"
            assert deciles.loc[t2_fake] == deciles.loc[t2_true], \
                f"{t2_fake} is not in the same decile as {t2_true}"


def test_decile_matched_bootstrap_reproducible_with_same_seed():
    """For the same seed, the fictitious-ticker assignment must be
    bit-reproducible (no unseeded source of randomness)."""
    tickers = [f"T{i:02d}" for i in range(20)]
    prior_returns = pd.Series(np.arange(20, dtype=float), index=tickers)
    selected_pairs = [("T00", "T19"), ("T05", "T14")]

    reps1 = decile_matched_bootstrap_pairs(selected_pairs, prior_returns, n_reps=30, seed=42)
    reps2 = decile_matched_bootstrap_pairs(selected_pairs, prior_returns, n_reps=30, seed=42)
    assert reps1 == reps2


def test_decile_matched_bootstrap_singleton_decile_returns_itself():
    """Edge case: if n_deciles == n_tickers, each ticker is the only member
    of its own decile -> the only possible substitution is the ticker
    itself, no crash (no alternative in the pool)."""
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
    print("test_inference: all tests PASSED.")
