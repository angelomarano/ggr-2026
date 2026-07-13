"""
test_selection_cluster.py — PCA loadings, OPTICS clustering with its
k-means/silhouette fallback, intra-cluster SSD (Variant A) and
Engle-Granger (Variant B) ranking, the brute-force EG comparator, and
Benjamini-Hochberg/Yekutieli, on small fictitious universes with
hand-computable numbers where feasible.

Note on PCA (see also pca_loadings' docstring): principal components are
unique only up to sign (and, for repeated eigenvalues, up to rotation), so
individual loading VALUES cannot be hand-verified the way an OLS
coefficient can. What IS exactly hand-verifiable, and what
test_pca_loadings_recovers_known_group_structure checks, is a STRUCTURAL
property: tickers built as exact positive scalar multiples of the same
underlying factor become, after per-ticker standardization, bit-identical
series - so their PCA loadings must coincide exactly (distance ~0), while
two tickers built from different, uncorrelated factors must not.

Note on Engle-Granger p-values: like the HAC estimator tested in
test_inference.py, EG/ADF p-values are asymptotic and not hand-computable
to an exact number. What's checked instead, matching that precedent, is
the qualitative behavior the test is designed to exercise - a clearly low
p-value for a synthetically cointegrated pair, a clearly high one for two
independent random walks - with a fixed seed and a large enough sample
that the result isn't a coin flip.
"""
import sys
sys.path.insert(0, ".")

import math

import numpy as np
import pandas as pd

import config
from src.formation import normalized_price_indices
from src.selection_cluster import (
    _half_life,
    _optics_is_degenerate,
    benjamini_hochberg,
    benjamini_yekutieli,
    brute_force_cointegration_screen,
    cluster_tickers,
    cointegration_intra_cluster_ranking,
    engle_granger_pair,
    intra_cluster_pairs,
    pca_loadings,
    select_pairs_via_clustering,
    ssd_intra_cluster_ranking,
)

TOL = 1e-9


# ---------------------------------------------------------------- (a) PCA

def test_pca_loadings_recovers_known_group_structure():
    """
    6 tickers, 2 groups, no idiosyncratic noise: A1/A2/A3 are exact
    positive scalar multiples of one factor, B1/B2/B3 of a different,
    uncorrelated factor. Per-ticker standardization erases the scalar
    multiples entirely, so within a group the standardized return series
    are BIT-IDENTICAL - their PCA loadings must therefore coincide
    exactly (Euclidean distance ~0 in the 4-component loading space),
    while the two groups, built from independent factors, must land at a
    clearly different point.
    """
    rng = np.random.default_rng(11)
    n_days = 120
    factor_a = rng.normal(0, 0.01, size=n_days)
    factor_b = rng.normal(0, 0.01, size=n_days)

    returns = pd.DataFrame({
        "A1": 1.0 * factor_a, "A2": 2.0 * factor_a, "A3": 0.5 * factor_a,
        "B1": 1.0 * factor_b, "B2": 3.0 * factor_b, "B3": 0.25 * factor_b,
    })

    loadings = pca_loadings(returns, n_components=4)
    assert list(loadings.columns) == ["PC1", "PC2", "PC3", "PC4"]

    def dist(t1, t2):
        return float(np.linalg.norm(loadings.loc[t1] - loadings.loc[t2]))

    assert dist("A1", "A2") < 1e-6, "exact scalar multiples of the same factor must collapse to one point"
    assert dist("A1", "A3") < 1e-6
    assert dist("B1", "B2") < 1e-6
    assert dist("B1", "B3") < 1e-6
    assert dist("A1", "B1") > 1.0, "independent factors must land at a clearly different point"


# ---------------------------------------------------------- (b) clustering

def test_optics_degenerate_hand_cases():
    """
    _optics_is_degenerate on hand-built label arrays (default thresholds:
    noise_share > 80%, largest non-noise cluster share > 50%):
      - 9 noise + 1 real point: noise_share=0.9 > 0.8 -> degenerate.
      - a mega-cluster of 8 plus a cluster of 2 (no noise): largest share
        = 8/10 = 0.8 > 0.5 -> degenerate.
      - an exact 50/50 split (no noise): largest share = 5/10 = 0.5, NOT
        strictly greater than 0.5 -> not degenerate (boundary case).
      - a healthy case with some noise (2 noise, two clusters of 4):
        noise_share=0.2, largest share=4/8=0.5 -> not degenerate.
    """
    assert _optics_is_degenerate(np.array([-1] * 9 + [0])) is True
    assert _optics_is_degenerate(np.array([0] * 8 + [1] * 2)) is True
    assert _optics_is_degenerate(np.array([0] * 5 + [1] * 5)) is False
    assert _optics_is_degenerate(np.array([-1, -1] + [0] * 4 + [1] * 4)) is False


def test_cluster_tickers_finds_obvious_clusters():
    """
    10 tickers, 2 groups of 5, each a shared factor plus small
    idiosyncratic noise (realistic, not exact duplicates like the PCA
    test above). OPTICS (default min_samples=3, xi=0.05) must find
    exactly 2 clusters and assign every A-ticker the same label, every
    B-ticker a different (but also shared) label, no ticker dropped as
    noise.
    """
    rng = np.random.default_rng(21)
    n_days = 150
    factor_a = rng.normal(0, 0.01, size=n_days)
    factor_b = rng.normal(0, 0.01, size=n_days)
    noise = 0.001

    returns = pd.DataFrame({
        **{f"A{i}": factor_a + rng.normal(0, noise, n_days) for i in range(5)},
        **{f"B{i}": factor_b + rng.normal(0, noise, n_days) for i in range(5)},
    })

    loadings = pca_loadings(returns, n_components=4)
    result = cluster_tickers(loadings, kmeans_k_range=(2, 4))

    assert result["method"] == "optics"
    labels = result["labels"]
    assert len(labels) == 10, "no ticker should be dropped as noise in this clean case"

    a_labels = {labels[f"A{i}"] for i in range(5)}
    b_labels = {labels[f"B{i}"] for i in range(5)}
    assert len(a_labels) == 1, "all A-tickers must share one cluster label"
    assert len(b_labels) == 1, "all B-tickers must share one cluster label"
    assert a_labels != b_labels, "A and B must land in different clusters"


def test_cluster_tickers_falls_back_on_independent_tickers():
    """
    12 mutually independent tickers (no shared factor structure at all):
    OPTICS has no density structure to find, so it must degenerate (per
    _optics_is_degenerate) and cluster_tickers must fall back to k-means.
    Unlike the OPTICS path, the fallback assigns every ticker a label -
    there is no "noise" concept in k-means.
    """
    rng = np.random.default_rng(33)
    n_days = 150
    returns = pd.DataFrame({f"T{i:02d}": rng.normal(0, 0.01, n_days) for i in range(12)})

    loadings = pca_loadings(returns, n_components=4)
    result = cluster_tickers(loadings, kmeans_k_range=(2, 4))

    assert result["method"] == "kmeans_fallback"
    assert len(result["labels"]) == 12, "k-means fallback must label every ticker, none dropped"


def test_cluster_tickers_kmeans_fallback_reproducible_with_same_seed():
    """Same seed, same k-means fallback labels (KMeans' own random_state is
    wired to cluster_tickers' seed parameter, config.SEED by default) - no
    unseeded source of randomness, matching the reproducibility precedent
    in test_inference.py (stationary bootstrap, decile-matched bootstrap)."""
    rng = np.random.default_rng(33)
    n_days = 150
    returns = pd.DataFrame({f"T{i:02d}": rng.normal(0, 0.01, n_days) for i in range(12)})
    loadings = pca_loadings(returns, n_components=4)

    result1 = cluster_tickers(loadings, kmeans_k_range=(2, 4), seed=123)
    result2 = cluster_tickers(loadings, kmeans_k_range=(2, 4), seed=123)

    assert result1["method"] == result2["method"] == "kmeans_fallback"
    pd.testing.assert_series_equal(result1["labels"], result2["labels"])


def test_intra_cluster_pairs_only_within_same_label():
    """Hand-built cluster assignment: {A,B} -> 0, {C,D} -> 1, {E} -> 2
    (singleton, contributes nothing). Expected candidate pairs: exactly
    (A,B) and (C,D) - never (A,C), (A,E), etc."""
    clusters = pd.Series({"A": 0, "B": 0, "C": 1, "D": 1, "E": 2})
    pairs = set(intra_cluster_pairs(clusters))
    assert pairs == {("A", "B"), ("C", "D")}


# ------------------------------------------------------- (a bis) SSD arm

def test_ssd_intra_cluster_ranking_never_crosses_clusters():
    """
    5 tickers, 2 clusters: {A,B,C} (cluster 0) and {D,E} (cluster 1).
    A/B/C reuse the EXACT hand-computed SSD fixture from
    test_formation.py (r=0.25=2^-2, exactly representable in float64):
      SSD(A,B) = 0.0625, SSD(A,C) = 0.0625, SSD(B,C) = 0.25.
    D/E are deliberately built so that D-E's own SSD is large (99.980001,
    computed below) while E's price path is almost identical to A's -
    if the implementation ever leaked cross-cluster comparisons, the
    pair (A,E) would show up with a suspiciously tiny SSD; correct
    behavior must never even consider it, since A and E are in different
    clusters.
    P*_D = [1, 11.0], P*_E = [1, 1.001]:
      SSD(D,E) = (11.0-1.001)^2 = 9.999^2 = 99.980001
      sigma(D,E) = std([0, 9.999], ddof=0) = 9.999/2 = 4.9995
    """
    returns = pd.DataFrame({
        "A": [0.00], "B": [0.25], "C": [-0.25], "D": [10.0], "E": [0.001],
    })
    price_index = normalized_price_indices(returns)
    clusters = pd.Series({"A": 0, "B": 0, "C": 0, "D": 1, "E": 1})

    ranked = ssd_intra_cluster_ranking(price_index, clusters, top_n=10)

    pairs = list(zip(ranked["ticker_1"], ranked["ticker_2"]))
    assert pairs == [("A", "B"), ("A", "C"), ("B", "C"), ("D", "E")], (
        "must contain exactly the 3 intra-cluster-0 pairs and the 1 "
        "intra-cluster-1 pair, ranked by ascending SSD, and NEVER a "
        "cross-cluster pair like (A, E) despite its tiny would-be SSD"
    )
    assert abs(ranked.iloc[0]["ssd"] - 0.0625) < TOL
    assert abs(ranked.iloc[1]["ssd"] - 0.0625) < TOL
    assert abs(ranked.iloc[2]["ssd"] - 0.25) < TOL
    assert abs(ranked.iloc[3]["ssd"] - 99.980001) < 1e-6
    assert abs(ranked.iloc[3]["sigma"] - 4.9995) < 1e-6


# ---------------------------------------------------- (c) Engle-Granger

def _synthetic_cointegrated_pair(seed: int, n: int = 600, phi: float = 0.9):
    """P2 = P1 * exp(u_t), u ~ AR(1) stationary with the given phi (same
    design as PROTOCOL.md §6's synthetic cointegration test)."""
    rng = np.random.default_rng(seed)
    log_p1 = np.cumsum(rng.normal(0, 0.01, size=n))
    p1 = np.exp(log_p1)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = phi * u[t - 1] + rng.normal(0, 0.015)
    p2 = p1 * np.exp(u)
    return p1, p2


def _synthetic_independent_pair(seed: int, n: int = 600):
    """Two independent log-random-walks, no shared drift or factor."""
    rng = np.random.default_rng(seed)
    log_a = np.cumsum(rng.normal(0, 0.01, size=n))
    log_b = np.cumsum(rng.normal(0, 0.01, size=n))
    return np.exp(log_a), np.exp(log_b)


def test_engle_granger_pair_cointegrated_pair_low_pvalue():
    """Synthetically cointegrated pair (P2 = P1*exp(u_t), u AR(1), phi=0.9
    -> theoretical half-life -ln(2)/ln(0.9) = 6.58 days): expected a low
    p-value (reject the no-cointegration null) and a half-life inside
    PROTOCOL.md's [5, 60] day filter range."""
    p1, p2 = _synthetic_cointegrated_pair(seed=2)
    price_index = pd.DataFrame({"P1": p1, "P2": p2})

    result = engle_granger_pair(price_index, "P1", "P2")

    assert result["p_value"] < 0.01
    assert result["half_life_days"] is not None
    assert config.HALF_LIFE_RANGE_DAYS[0] <= result["half_life_days"] <= config.HALF_LIFE_RANGE_DAYS[1]


def test_engle_granger_pair_independent_walks_high_pvalue():
    """Two independent log-random-walks (no shared drift or factor):
    expected a high p-value (fail to reject the no-cointegration null)."""
    q1, r1 = _synthetic_independent_pair(seed=101)
    price_index = pd.DataFrame({"Q1": q1, "R1": r1})

    result = engle_granger_pair(price_index, "Q1", "R1")

    assert result["p_value"] > 0.3


# --------------------------------------------------------- (d) half-life

def test_half_life_hand_computed_ar1():
    """
    Noiseless AR(1) with a KNOWN coefficient phi=0.9: x_t = phi^t
    (x_0=1), so the OLS slope of x[1:] on x[:-1] (no intercept) recovers
    phi EXACTLY (a perfect noiseless linear relationship). Expected
    half-life = -ln(2)/ln(0.9), computed independently here with
    math.log to cross-check _half_life's own formula.
    """
    phi = 0.9
    x = np.array([phi ** t for t in range(50)])
    expected = -math.log(2) / math.log(phi)

    result = _half_life(x)

    assert result is not None
    assert abs(result - expected) < TOL
    assert abs(result - 6.578813478960585) < 1e-9  # value spelled out for the reader


def test_half_life_none_outside_valid_phi_range():
    """phi <= 0 (oscillating, sign flips every step) and phi >= 1 (not
    mean-reverting) must both return None - the half-life is undefined in
    either case (see _half_life's docstring)."""
    n = 50
    oscillating = np.array([(-0.5) ** t for t in range(n)])  # phi = -0.5
    explosive = np.array([1.05 ** t for t in range(n)])       # phi = 1.05
    unit_root = np.array([1.0 for _ in range(n)])             # phi = 1.0 exactly

    assert _half_life(oscillating) is None
    assert _half_life(explosive) is None
    assert _half_life(unit_root) is None


def test_cointegration_intra_cluster_ranking_filters_and_ranks():
    """
    4 tickers, 2 clusters: {P1,P2} (cluster 0, cointegrated by
    construction, same fixture as the engle_granger_pair tests above) and
    {Q1,R1} (cluster 1, independent random walks). Only (P1,P2) must
    survive the p-value/half-life filter and appear in the ranking;
    (Q1,R1) must be excluded (high p-value), and no cross-cluster pair
    like (P1,Q1) is ever even tested (intra_cluster_pairs restriction).
    """
    p1, p2 = _synthetic_cointegrated_pair(seed=2)
    q1, r1 = _synthetic_independent_pair(seed=101)

    price_index = pd.DataFrame({"P1": p1, "P2": p2, "Q1": q1, "R1": r1})
    clusters = pd.Series({"P1": 0, "P2": 0, "Q1": 1, "R1": 1})

    ranked = cointegration_intra_cluster_ranking(price_index, clusters, top_n=10)

    assert len(ranked) == 1
    assert (ranked.iloc[0]["ticker_1"], ranked.iloc[0]["ticker_2"]) == ("P1", "P2")
    assert ranked.iloc[0]["p_value"] < config.EG_PVALUE_MAX


# ------------------------------------------------------------ (e) BH/BY

def test_benjamini_hochberg_and_yekutieli_hand_computed():
    """
    n=5, alpha=0.05, p-values chosen so each of the first 4 lands EXACTLY
    on its own BH threshold i/n*alpha (thresholds are 0.01, 0.02, 0.03,
    0.04, 0.05 for i=1..5):
      p = [0.01, 0.02, 0.03, 0.04, 0.50]
    BH: p_(i) <= i/n*alpha holds for i=1..4 (equality) and fails for i=5
    (0.50 > 0.05) -> BH rejects ranks 1-4, keeps rank 5.
    BY divides every threshold by c(5) = 1+1/2+1/3+1/4+1/5 = 137/60 ~=
    2.28333, so threshold_i = i/(5*137/60)*0.05 ~= i*0.0043796 for
    i=1..5 (~0.00438, 0.00876, 0.01314, 0.01752, 0.02190) - EVERY p-value
    here exceeds its own (much stricter) BY threshold, so BY rejects
    NOTHING: a hand-verifiable illustration of BY's extra conservatism
    under dependence, which is exactly why PROTOCOL.md §4/H5 step 6 asks
    for it alongside BH.
    """
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.50])

    bh = benjamini_hochberg(p, alpha=0.05)
    by = benjamini_yekutieli(p, alpha=0.05)

    assert list(bh) == [True, True, True, True, False]
    assert list(by) == [False, False, False, False, False]


def test_benjamini_yekutieli_does_reject_an_extreme_p_value():
    """
    Sanity check that BY isn't vacuously always-empty: with a p-value
    small enough to clear even BY's stricter threshold
    (i=1: alpha/(n*c(n)) = 0.05/(5*137/60) ~= 0.0043796), it must survive.
    p = [0.0001, 0.5, 0.6, 0.7, 0.8], n=5: only the first (0.0001 <
    0.0043796) passes.
    """
    p = np.array([0.0001, 0.5, 0.6, 0.7, 0.8])
    by = benjamini_yekutieli(p, alpha=0.05)
    assert list(by) == [True, False, False, False, False]


def test_benjamini_hochberg_no_survivors_when_nothing_passes():
    """All p-values well above alpha/n for every rank: BH rejects nothing."""
    p = np.array([0.9, 0.8, 0.7])
    bh = benjamini_hochberg(p, alpha=0.05)
    assert list(bh) == [False, False, False]


# --------------------------------------------- brute-force comparator

def test_brute_force_cointegration_screen_expected_false_positives_hand_computed():
    """
    Same 4-ticker fixture as test_cointegration_intra_cluster_ranking_
    filters_and_ranks, but run brute-force (no clustering restriction):
    C(4,2) = 6 pairs tested, so expected_false_positives = 6 * 0.05 = 0.3
    EXACTLY (trivial to verify by hand, and a direct instance of
    PROTOCOL.md §4/H5 step 6's "falsi positivi attesi sotto ipotesi
    nulla globale" figure, just at N=4 instead of the ~500-ticker,
    ~125,000-pair full universe). Only (P1, P2) should survive the
    p-value/half-life filter AND Benjamini-Hochberg.
    """
    p1, p2 = _synthetic_cointegrated_pair(seed=2)
    q1, r1 = _synthetic_independent_pair(seed=101)

    price_index = pd.DataFrame({"P1": p1, "P2": p2, "Q1": q1, "R1": r1})

    result = brute_force_cointegration_screen(price_index)

    assert result["n_tests"] == 6
    assert abs(result["expected_false_positives"] - 0.3) < TOL
    assert result["n_bh_survivors"] == 1, "only (P1, P2) is extreme enough to survive BH"
    assert result["n_by_survivors"] == 1, "(P1, P2)'s p-value is extreme enough to survive BY too"
    assert len(result["top_n"]) == 1
    assert (result["top_n"].iloc[0]["ticker_1"], result["top_n"].iloc[0]["ticker_2"]) == ("P1", "P2")


def test_brute_force_cointegration_screen_empty_universe():
    """A single-ticker universe has zero pairs to test: n_tests=0,
    expected_false_positives=0.0, no crash (mirrors formation.py's
    "no exception, no artificial padding" convention for small universes)."""
    price_index = pd.DataFrame({"A": [1.0, 1.01, 1.02]})
    result = brute_force_cointegration_screen(price_index)
    assert result["n_tests"] == 0
    assert result["expected_false_positives"] == 0.0
    assert len(result["top_n"]) == 0


# ------------------------------------------------------------ orchestrator

def test_select_pairs_via_clustering_smoke():
    """Light integration check: the full pipeline (PCA -> cluster ->
    Variant A + Variant B) runs end to end on the same clean 2-group
    fixture as test_cluster_tickers_finds_obvious_clusters, and Variant A
    never returns a cross-cluster pair."""
    rng = np.random.default_rng(21)
    n_days = 150
    factor_a = rng.normal(0, 0.01, size=n_days)
    factor_b = rng.normal(0, 0.01, size=n_days)
    noise = 0.001

    returns = pd.DataFrame({
        **{f"A{i}": factor_a + rng.normal(0, noise, n_days) for i in range(5)},
        **{f"B{i}": factor_b + rng.normal(0, noise, n_days) for i in range(5)},
    })
    price_index = normalized_price_indices(returns)

    out = select_pairs_via_clustering(returns, price_index, n_components=4, top_n=20)

    assert out["clustering"]["method"] == "optics"
    labels = out["clustering"]["labels"]
    for _, row in out["variant_a"].iterrows():
        assert labels[row["ticker_1"]] == labels[row["ticker_2"]], (
            "Variant A must never rank a cross-cluster pair"
        )


if __name__ == "__main__":
    test_pca_loadings_recovers_known_group_structure()
    test_optics_degenerate_hand_cases()
    test_cluster_tickers_finds_obvious_clusters()
    test_cluster_tickers_falls_back_on_independent_tickers()
    test_cluster_tickers_kmeans_fallback_reproducible_with_same_seed()
    test_intra_cluster_pairs_only_within_same_label()
    test_ssd_intra_cluster_ranking_never_crosses_clusters()
    test_engle_granger_pair_cointegrated_pair_low_pvalue()
    test_engle_granger_pair_independent_walks_high_pvalue()
    test_half_life_hand_computed_ar1()
    test_half_life_none_outside_valid_phi_range()
    test_cointegration_intra_cluster_ranking_filters_and_ranks()
    test_benjamini_hochberg_and_yekutieli_hand_computed()
    test_benjamini_yekutieli_does_reject_an_extreme_p_value()
    test_benjamini_hochberg_no_survivors_when_nothing_passes()
    test_brute_force_cointegration_screen_expected_false_positives_hand_computed()
    test_brute_force_cointegration_screen_empty_universe()
    test_select_pairs_via_clustering_smoke()
    print("test_selection_cluster: all tests PASSED.")
