"""
inference.py — GGR statistical inference: Newey-West t-test on the mean,
stationary block bootstrap for the 95% CI, factor regression (alpha,
loadings, NW t-stat) against data/factors.py.

PROTOCOL.md §2.1/§2.3: "Inference: Newey-West 6 lags on the monthly series".
§4/H1: "95% CI with stationary block bootstrap (mean block 6 months, 10,000
replications)".

Newey-West: we do NOT reimplement the HAC estimator by hand. A HAC t-test on
a series' mean is algebraically an OLS regression of y on a single constant
with HAC covariance — the standard approach (Newey & West, 1987), which we
delegate to statsmodels (sm.OLS(..., cov_type="HAC")).

Stationary block bootstrap (Politis & Romano, 1994): there is no direct
equivalent in statsmodels/scipy in this project's environment, so we
implement it here: blocks of GEOMETRIC length (mean = mean_block_months)
sampled with circular wraparound (the "stationary" property: unlike a
fixed-block bootstrap, the resampled series stays weakly stationary if the
original is).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config


def newey_west_mean_test(returns, lags: int = config.NW_LAGS) -> dict:
    """
    t-test on the mean of a monthly return series, with Newey-West (HAC)
    standard errors at `lags` lags (default 6, frozen by PROTOCOL.md §2.1).
    Implemented as an OLS of y on a constant with cov_type="HAC": this is
    the standard way to get a HAC t-test on the mean, not an ad hoc estimator.

    Returns: mean, se, t_stat, p_value (two-sided, H0: mean=0), n.
    """
    y = np.asarray(returns, dtype=float)
    n = len(y)
    X = np.ones((n, 1))
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "mean": float(model.params[0]),
        "se": float(model.bse[0]),
        "t_stat": float(model.tvalues[0]),
        "p_value": float(model.pvalues[0]),
        "n": n,
    }


def stationary_bootstrap_ci(
    returns,
    mean_block_months: int = config.BLOCK_BOOTSTRAP_MEAN_BLOCK_MONTHS,
    n_reps: int = config.BLOCK_BOOTSTRAP_REPS,
    ci: float = 0.95,
    seed: int = config.SEED,
) -> dict:
    """
    Stationary bootstrap CI (Politis & Romano 1994) on the series' mean.
    Each replication reconstructs a synthetic series of length n by
    concatenating blocks of random geometric length (parameter p =
    1/mean_block_months, so E[block length] = mean_block_months), sampled
    with circular wraparound over the original index (random start in
    [0,n), then consecutive indices modulo n). n_reps independent
    replications (seeded rng: reproducible). CI = (1-ci)/2 and 1-(1-ci)/2
    percentiles of the bootstrap means distribution.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(returns, dtype=float)
    n = len(y)
    p = 1.0 / mean_block_months

    boot_means = np.empty(n_reps)
    for b in range(n_reps):
        idx = np.empty(n, dtype=int)
        pos = 0
        while pos < n:
            start = rng.integers(0, n)
            block_len = min(int(rng.geometric(p)), n - pos)
            idx[pos : pos + block_len] = (start + np.arange(block_len)) % n
            pos += block_len
        boot_means[b] = y[idx].mean()

    alpha = 1 - ci
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(y.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_reps": n_reps,
        "boot_means": boot_means,
    }


def factor_regression(
    excess_returns,
    factors: pd.DataFrame,
    factor_cols: tuple[str, ...] = ("Mkt-RF", "SMB", "HML", "Mom", "ST_Rev"),
    lags: int = config.NW_LAGS,
) -> dict:
    """
    Factor regression (PROTOCOL.md §2.4.3): monthly excess returns (already
    net of RF, computed upstream) on FF3 + Momentum + ST-Reversal,
    Newey-West standard errors at `lags` lags.

    excess_returns: Series indexed by month (start-of-month Timestamp),
        same format as data/factors.py.
    factors: DataFrame as returned by data.factors.load_factors.

    Alignment: inner join on the index (month). A month present in
    excess_returns but absent from factors (e.g. the Ken French Data
    Library not yet updated to the latest month) is dropped, it doesn't
    produce a NaN in the regression.

    Returns: alpha, alpha_se, alpha_t, loadings (dict factor->beta),
    loadings_se, loadings_t, n_obs, r_squared.
    """
    y_series = excess_returns if isinstance(excess_returns, pd.Series) else pd.Series(excess_returns)
    aligned = pd.concat(
        [y_series.rename("y"), factors[list(factor_cols)]], axis=1, join="inner"
    ).dropna()

    y = aligned["y"].to_numpy()
    # explicit has_constant="add": if for a short/degenerate window a factor
    # came out with zero variance, sm.add_constant's default would mistake
    # it for a constant already present and skip adding a new one, silently
    # misaligning params (5 values) against names (6 names). "add" always
    # guarantees one extra column for the alpha.
    X = sm.add_constant(aligned[list(factor_cols)].to_numpy(), has_constant="add")
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    names = ["alpha", *factor_cols]
    params = dict(zip(names, model.params))
    se = dict(zip(names, model.bse))
    tvals = dict(zip(names, model.tvalues))

    return {
        "alpha": params["alpha"],
        "alpha_se": se["alpha"],
        "alpha_t": tvals["alpha"],
        "loadings": {c: params[c] for c in factor_cols},
        "loadings_se": {c: se[c] for c in factor_cols},
        "loadings_t": {c: tvals[c] for c in factor_cols},
        "n_obs": len(y),
        "r_squared": float(model.rsquared),
    }


def long_short_leg_regression(
    long_returns,
    short_returns,
    factors: pd.DataFrame,
    factor_cols: tuple[str, ...] = ("Mkt-RF", "SMB", "HML", "Mom", "ST_Rev"),
    lags: int = config.NW_LAGS,
) -> dict:
    """
    Long/short alpha decomposition (PROTOCOL.md §2.4, point 2 — mandatory
    falsification test): the SAME factor regression as factor_regression,
    applied separately to the monthly returns of the portfolio's LONG leg
    alone and SHORT leg alone (no new formula: it's a double call to
    factor_regression, one per leg).

    long_returns, short_returns: Series indexed by month, same format as
        excess_returns in factor_regression (typically obtained by
        aggregating to portfolio level, with the same logic as
        src/returns.py, the payoffs of each pair's long/short leg alone).

    Expected (GGR Table 7, top-20 — Gate 1 qualitative invariant,
    PROTOCOL.md §3): SHORT leg alpha negative and significant (shorting a
    stock with negative alpha contributes positively to the strategy), LONG
    leg alpha near zero and not significant: the pair trade's profitability
    comes from the short leg, not the long one. If the replication came out
    the other way around, it's a red flag for a sign or construction error
    (PROTOCOL.md §2.4.2), not a result to accept at face value.

    Returns: {"long": factor_regression(...), "short": factor_regression(...)}.

    Note: alpha_long - alpha_short is NOT expected to equal the primary
    portfolio's net factor_regression alpha, and that is not a bug. Both
    legs here are compounded monthly as independent stand-alone
    sub-portfolios (the standard approach for this decomposition - each leg
    is its own hypothetical fund with its own monthly return), while the
    primary portfolio compounds the already-netted daily series. Compounding
    is non-linear - (1+a)(1+b)-1 - [(1+c)(1+d)-1] != (1+a-c)(1+b-d)-1 in
    general - so the two aggregation paths diverge by construction, not by
    error. Verified empirically (pair-day level and pre-compounding daily
    portfolio level both match exactly): the divergence appears only after
    monthly compounding, typically on the order of a few bp/month.
    """
    return {
        "long": factor_regression(long_returns, factors, factor_cols, lags),
        "short": factor_regression(short_returns, factors, factor_cols, lags),
    }


def decile_of_returns(returns: pd.Series, n_deciles: int = 10) -> pd.Series:
    """
    Assigns each ticker (index of `returns`) the decile its return falls in
    (1 = lowest decile, n_deciles = highest), via pd.qcut on equal-frequency
    bins. `returns` is typically the universe's return in the MONTH BEFORE
    the formation period (PROTOCOL.md §2.4, point 1: decile-matching is done
    on the prior return, not on quantities computed in the trading period,
    to avoid introducing look-ahead).

    duplicates="drop": if several tickers have identical returns such that
    n_deciles distinct bins aren't possible, fewer bins are produced instead
    of a ValueError — a data edge case, not a pipeline error.
    """
    return pd.qcut(returns, n_deciles, labels=False, duplicates="drop") + 1


def _sample_same_decile(ticker: str, deciles: pd.Series, rng: np.random.Generator) -> str:
    """A random ticker from the same decile as `ticker` (including `ticker`
    itself if it's the only member of its decile: no alternative possible)."""
    d = deciles.loc[ticker]
    candidates = deciles.index[deciles == d].to_numpy()
    return str(candidates[rng.integers(len(candidates))])


def decile_matched_bootstrap_pairs(
    selected_pairs: list[tuple[str, str]],
    prior_month_returns: pd.Series,
    n_deciles: int = 10,
    n_reps: int = config.RANDOM_PAIRS_BOOTSTRAP_REPS,
    seed: int = config.SEED,
) -> list[list[tuple[str, str]]]:
    """
    Decile-matched random-pairs bootstrap falsification (PROTOCOL.md §2.4,
    point 1 — mandatory falsification test, not optional): expected
    bootstrap return ~0 or negative; if it came out positive, either the
    pipeline has a bug or the apparent profit is actually masked reversal,
    not convergence of cointegrated pairs.

    For n_reps repetitions, replaces EVERY real pair (ticker_1, ticker_2) in
    `selected_pairs` with a pair of fictitious tickers, each drawn RANDOMLY
    from the same return decile in the MONTH BEFORE the formation period
    (prior_month_returns, indexed by ticker) as the respective original
    leg — ticker_1 replaced by a ticker from ticker_1's decile, ticker_2 by
    a ticker from ticker_2's decile, independently.

    "Same trigger/event dates" (PROTOCOL.md): this function ONLY generates
    the fictitious-ticker assignment for each replication; the caller is
    responsible for re-running the simulation (simulate_pair_same_day or
    _wait_one_day) on the substituted tickers' returns BUT over the same
    trading period/window as the real pair — no new dates are generated
    here, the existing simulation is reused with different inputs.

    selected_pairs: the run's real pairs (format (ticker_1, ticker_2)).
    prior_month_returns: Series ticker -> return in the month before the
        formation period, over the FULL universe (not just the selected
        pairs: needed to build the deciles and draw the substitutes).

    Returns: a list of n_reps lists of fictitious pairs (same order and
    length as selected_pairs, one list per replication).
    """
    deciles = decile_of_returns(prior_month_returns, n_deciles)
    rng = np.random.default_rng(seed)

    reps: list[list[tuple[str, str]]] = []
    for _ in range(n_reps):
        rep_pairs = [
            (_sample_same_decile(t1, deciles, rng), _sample_same_decile(t2, deciles, rng))
            for t1, t2 in selected_pairs
        ]
        reps.append(rep_pairs)
    return reps
