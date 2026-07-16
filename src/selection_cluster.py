"""
selection_cluster.py — H5 alternative pair-selection pipeline (PROTOCOL.md
§4/H5): PCA on standardized formation-period returns, OPTICS clustering
(k-means/silhouette fallback if OPTICS degenerates), intra-cluster
candidate pairs ranked two ways (SSD, Engle-Granger cointegration), and a
brute-force Engle-Granger comparator with Benjamini-Hochberg/Yekutieli
multiple-testing correction. Everything downstream of the formation period
only; nothing here ever touches trading-period data.

Design choice, declared here (PROTOCOL.md doesn't specify the exact input):
every function in this module that needs price *levels* (PCA, SSD, EG)
takes the SAME `price_index` object as src/formation.py's
normalized_price_indices (P*_i0 = 1, P*_it = prod(1+r)), not raw Adj
Close. For SSD this is required (formation.rank_pairs already assumes
it). For PCA it doesn't matter which representation of "returns" is used
as long as it's standardized before PCA (done here). For Engle-Granger,
log(P*_it) is a valid log-price level for cointegration purposes: it
differs from the real log(Adj Close) only by a per-ticker additive
constant (log of the ticker's actual starting price), which the EG
regression's intercept absorbs exactly. Reusing one shared price
representation across PCA/SSD/EG avoids requiring the caller to also load
raw price levels separately for this module alone.

OPTICS degeneracy threshold (PROTOCOL.md §8 names the failure modes -
"all noise or one mega-cluster" - but not the exact trigger): declared
here as noise_share > 80% OR the largest non-noise cluster holding > 50%
of the non-noise points. Both thresholds are arbitrary defaults chosen to
catch the two named failure modes and are documented, not tuned against
any result.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import OPTICS, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import coint

import config
from src.formation import rank_pairs


def standardize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker standardization (PROTOCOL.md §4/H5 step 1): each column
    (ticker) is rescaled to mean 0, std 1 over the formation days, using
    its OWN mean/std (sklearn.preprocessing.StandardScaler's default is
    exactly this: per-feature/per-column standardization with population
    std, ddof=0). Not reimplemented by hand."""
    scaler = StandardScaler()
    values = scaler.fit_transform(returns.to_numpy())
    return pd.DataFrame(values, index=returns.index, columns=returns.columns)


def pca_loadings(
    returns: pd.DataFrame, n_components: int = config.PCA_N_COMPONENTS
) -> pd.DataFrame:
    """
    PCA on standardized formation-period returns (PROTOCOL.md §4/H5 steps
    1-2, ratified n_components=10, declared sensitivity {5, 15} via
    config.PCA_SENSITIVITY). Not reimplemented by hand: delegated to
    sklearn.decomposition.PCA with svd_solver="full" (deterministic, no
    randomness to seed).

    Orientation (Avellaneda & Lee 2010 / Sarmento & Horta clustering
    setup): tickers are the SAMPLES, formation days are the FEATURES, so
    the standardized (days x tickers) matrix is transposed to (tickers x
    days) before fitting. PCA then finds the common latent factors driving
    daily return co-movement across tickers, and each ticker gets a
    "loading" = its coordinates in the reduced n_components-dimensional
    factor space. These loadings, not the raw returns, are what gets
    clustered downstream (cluster_tickers).

    Returns a DataFrame indexed by ticker, columns PC1..PCn_components.
    """
    standardized = standardize_returns(returns)
    pca = PCA(n_components=n_components, svd_solver="full")
    loadings = pca.fit_transform(standardized.T.to_numpy())
    columns = [f"PC{i + 1}" for i in range(n_components)]
    return pd.DataFrame(loadings, index=returns.columns, columns=columns)


def _optics_is_degenerate(
    labels: np.ndarray,
    noise_share_max: float = 0.80,
    largest_cluster_share_max: float = 0.50,
) -> bool:
    """
    Pure threshold check, factored out for direct testing (see module
    docstring for the rationale behind the two default thresholds).
    labels: OPTICS' raw labels_ array (-1 = noise, >=0 = cluster id).
    Degenerate if:
      - more than noise_share_max of all points are noise, OR
      - among the non-noise points, the single largest cluster holds more
        than largest_cluster_share_max of them.
    An empty `labels` array, or one where every point is noise, is
    degenerate by definition (nothing to cluster).
    """
    labels = np.asarray(labels)
    n = len(labels)
    if n == 0:
        return True
    noise_share = float(np.mean(labels == -1))
    if noise_share > noise_share_max:
        return True
    non_noise = labels[labels != -1]
    if len(non_noise) == 0:
        return True
    counts = np.bincount(non_noise)
    largest_cluster_share = counts.max() / len(non_noise)
    return bool(largest_cluster_share > largest_cluster_share_max)


def cluster_tickers(
    loadings: pd.DataFrame,
    min_samples: int = config.OPTICS_MIN_SAMPLES,
    xi: float = config.OPTICS_XI,
    kmeans_k_range: tuple[int, int] = config.KMEANS_K_RANGE,
    seed: int = config.SEED,
) -> dict:
    """
    OPTICS primary (PROTOCOL.md §4/H5 step 3, ratified min_samples=3,
    xi=0.05), k-means/silhouette fallback if OPTICS degenerates
    (_optics_is_degenerate; PROTOCOL.md §8 names the failure modes without
    fixing the trigger threshold, fixed and documented here). Neither
    algorithm is reimplemented by hand: sklearn.cluster.OPTICS /
    sklearn.cluster.KMeans / sklearn.metrics.silhouette_score.

    OPTICS path: noise points (label -1) are EXCLUDED entirely from the
    returned labels (PROTOCOL.md §4/H5 step 3: "i punti 'noise' sono
    esclusi").
    K-means fallback path: k is chosen by grid search over
    kmeans_k_range=[lo, hi] (ratified [5, 30]), picking the k with the
    highest silhouette score; every ticker gets a label (no noise concept
    in k-means).

    Returns {"labels": pd.Series (ticker -> int cluster id),
    "method": "optics" or "kmeans_fallback"}.
    """
    tickers = loadings.index.to_numpy()
    X = loadings.to_numpy()

    optics = OPTICS(min_samples=min_samples, xi=xi)
    raw_labels = optics.fit_predict(X)

    if not _optics_is_degenerate(raw_labels):
        mask = raw_labels != -1
        labels = pd.Series(raw_labels[mask], index=tickers[mask], name="cluster")
        return {"labels": labels, "method": "optics"}

    lo, hi = kmeans_k_range
    n = len(tickers)
    best_k, best_score, best_labels = None, -1.0, None
    for k in range(lo, min(hi, n - 1) + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        km_labels = km.fit_predict(X)
        if len(set(km_labels)) < 2:
            continue
        score = silhouette_score(X, km_labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, km_labels
    if best_labels is None:
        raise ValueError(
            f"OPTICS degenerated and the k-means fallback found no valid k in "
            f"[{lo}, {hi}] for {n} tickers (need at least 2 clusters and k < n)."
        )
    labels = pd.Series(best_labels, index=tickers, name="cluster")
    return {"labels": labels, "method": "kmeans_fallback"}


def intra_cluster_pairs(clusters: pd.Series) -> list[tuple[str, str]]:
    """All ticker pairs (i, j), i<j alphabetically, that share the same
    cluster label (PROTOCOL.md §4/H5 step 4: candidate pairs are
    restricted to intra-cluster only, never across clusters)."""
    pairs: list[tuple[str, str]] = []
    for _, members in clusters.groupby(clusters):
        ordered = sorted(members.index)
        pairs.extend(itertools.combinations(ordered, 2))
    return pairs


def ssd_intra_cluster_ranking(
    price_index: pd.DataFrame,
    clusters: pd.Series,
    top_n: int = config.TOP_PAIRS,
) -> pd.DataFrame:
    """
    Variant A (PROTOCOL.md §4/H5 step 5): the same SSD ranking as
    formation.rank_pairs, with the candidate universe restricted to
    intra-cluster pairs - a clean, apples-to-apples comparison against the
    GGR-SSD arm. Reuses formation.rank_pairs directly (no duplicated SSD
    logic): each cluster's own tickers are ranked in isolation by calling
    formation.rank_pairs on that cluster's price_index sub-table, so a
    pair is never scored against a rival from a different cluster. The
    per-cluster tables are then concatenated and globally re-sorted by
    SSD, exactly like formation.select_portfolios' top_n slice.

    Clusters with fewer than 2 members contribute no pairs. Returns an
    empty DataFrame (same columns) if no cluster has 2+ members.
    """
    tables = []
    for label, members in clusters.groupby(clusters):
        member_list = sorted(members.index)
        if len(member_list) < 2:
            continue
        ranked = rank_pairs(price_index[member_list])
        tables.append(ranked.assign(cluster=label))

    columns = ["ticker_1", "ticker_2", "ssd", "sigma", "cluster"]
    if not tables:
        return pd.DataFrame(columns=columns)

    combined = pd.concat(tables, ignore_index=True)
    combined = combined.sort_values(
        ["ssd", "ticker_1", "ticker_2"], kind="stable"
    ).reset_index(drop=True)
    combined.index = combined.index + 1
    combined.index.name = "rank"
    return combined.iloc[:top_n]


def _half_life(residuals: np.ndarray) -> float | None:
    """
    Half-life of mean reversion for an AR(1)/OU spread (PROTOCOL.md §4/H5
    step 5: "filtro half-life AR(1) dello spread in [5, 60] giorni").

    Fits resid_t = phi * resid_{t-1} + eps_t by OLS with NO intercept
    (the residuals already come from a regression that included a
    constant, so they are exactly mean-zero over the sample by
    construction - a standard simplification, e.g. Chan, "Algorithmic
    Trading", half-life derivation). Under this AR(1)/OU dynamic,
    E[resid_t | resid_0] = phi^t * resid_0, so the half-life is the t
    solving phi^t = 0.5:

        half_life = -ln(2) / ln(phi)

    Returns None if phi is outside (0, 1): phi <= 0 means the spread
    oscillates in sign step to step rather than decaying monotonically
    (the "time to halve" reading of half-life doesn't apply), phi >= 1
    means the spread is not mean-reverting (unit root or explosive) and
    the half-life is undefined/infinite.
    """
    resid = np.asarray(residuals, dtype=float)
    y = resid[1:]
    x = resid[:-1]
    phi = float(np.dot(x, y) / np.dot(x, x))
    if not (0 < phi < 1):
        return None
    return -np.log(2) / np.log(phi)


def engle_granger_pair(price_index: pd.DataFrame, i: str, j: str) -> dict:
    """
    Engle-Granger two-step cointegration test on log-prices (PROTOCOL.md
    §4/H5 step 5, Variant B). Delegated to
    statsmodels.tsa.stattools.coint (ADF on the OLS residual, MacKinnon
    p-value), not reimplemented: EG is itself a two-step OLS + ADF
    procedure, and statsmodels already implements the finite-sample
    MacKinnon critical-value/p-value surface that a hand implementation
    would otherwise have to reproduce from tables. trend="c" is passed
    explicitly (matches statsmodels' own default, stated here so the
    residual regression below - needed for half-life, which coint() does
    not expose - uses the identical specification: OLS of log(P_i) on a
    constant and log(P_j)).

    See module docstring for why `price_index` (formation.py's normalized
    P*_it, P*_i0=1) is a valid log-price level for this test.

    Returns {"t_stat", "p_value", "half_life_days", "residual_std"} (see
    _half_life for when half_life_days is None).

    residual_std = resid.std(ddof=0), the spread's own scale on the
    log-price axis this test operates on - NOT the SSD-style sigma from
    formation.spread_sigma (std of a *normalized price-index* spread,
    P*_i - P*_j). The two are different units on a different scale (log
    vs. normalized-price-level), so a sigma computed one way must never be
    fed into a trigger meant to be compared against a spread computed the
    other way. No extra regression: resid is already computed above for
    _half_life, this just reports its dispersion too.
    """
    log_i = np.log(price_index[i].to_numpy())
    log_j = np.log(price_index[j].to_numpy())
    t_stat, p_value, _ = coint(log_i, log_j, trend="c")
    resid = sm.OLS(log_i, sm.add_constant(log_j)).fit().resid
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "half_life_days": _half_life(resid),
        "residual_std": float(resid.std(ddof=0)),
    }


def _passes_eg_filter(
    table: pd.DataFrame,
    p_value_max: float,
    half_life_range: tuple[float, float],
) -> pd.Series:
    """Shared p-value + half-life filter (PROTOCOL.md §4/H5 step 5):
    p <= p_value_max AND half_life_days is defined (see _half_life) AND
    within half_life_range, inclusive on both ends."""
    lo, hi = half_life_range
    return (
        (table["p_value"] <= p_value_max)
        & table["half_life_days"].notna()
        & (table["half_life_days"] >= lo)
        & (table["half_life_days"] <= hi)
    )


def cointegration_intra_cluster_ranking(
    price_index: pd.DataFrame,
    clusters: pd.Series,
    p_value_max: float = config.EG_PVALUE_MAX,
    half_life_range: tuple[float, float] = config.HALF_LIFE_RANGE_DAYS,
    top_n: int = config.TOP_PAIRS,
) -> pd.DataFrame:
    """
    Variant B (PROTOCOL.md §4/H5 step 5): Engle-Granger + half-life filter,
    restricted to intra-cluster candidate pairs (reuses intra_cluster_pairs
    + engle_granger_pair, no duplicated EG logic). Pairs failing the
    p-value or half-life filter are dropped; survivors are ranked by
    ascending p-value and truncated to top_n.

    Returns a DataFrame (rank-indexed like formation.rank_pairs) with
    columns ticker_1, ticker_2, t_stat, p_value, half_life_days,
    residual_std (the EG residual's own std, see engle_granger_pair - the
    scale a caller must use for this pair's trigger, NOT
    formation.spread_sigma's SSD-style sigma). Empty (same columns) if no
    intra-cluster pair survives the filter.
    """
    pairs = intra_cluster_pairs(clusters)
    columns = ["ticker_1", "ticker_2", "t_stat", "p_value", "half_life_days", "residual_std"]
    if not pairs:
        return pd.DataFrame(columns=columns)

    rows = [
        {"ticker_1": i, "ticker_2": j, **engle_granger_pair(price_index, i, j)}
        for i, j in pairs
    ]
    table = pd.DataFrame(rows, columns=columns)
    survivors = table[_passes_eg_filter(table, p_value_max, half_life_range)]
    ranked = survivors.sort_values("p_value", kind="stable").reset_index(drop=True)
    ranked.index = ranked.index + 1
    ranked.index.name = "rank"
    return ranked.iloc[:top_n]


def benjamini_hochberg(p_values, alpha: float = config.FDR_ALPHA) -> np.ndarray:
    """
    Benjamini-Hochberg (1995) step-up FDR procedure, valid under
    independence or positive dependence (PRDS) among the tests -
    PROTOCOL.md §4/H5 step 6 flags this assumption as violated here (pair
    cointegration tests on overlapping tickers/factors are not
    independent) and requires reporting Benjamini-Yekutieli alongside it.

    Implemented directly (not via statsmodels.stats.multitest) because the
    procedure is a short, exactly-specified rank-based rule (sort
    p-values ascending, find the largest rank i with p_(i) <= (i/n)*alpha,
    reject all ranks <= i) that is easy to verify by hand on a small
    p-value vector (see tests) - keeping it here makes the exact formula
    auditable in one place next to benjamini_yekutieli, whose only
    difference is a single correction factor.

    Returns a boolean array, same order/length as p_values: True = reject
    H0 (survives the correction) at level alpha.
    """
    return _step_up_procedure(p_values, alpha, correction_factor=1.0)


def benjamini_yekutieli(p_values, alpha: float = config.FDR_ALPHA) -> np.ndarray:
    """
    Benjamini-Yekutieli (2001) step-up FDR procedure: identical to
    benjamini_hochberg, except every threshold is divided by
    c(n) = sum_{k=1}^n 1/k (the n-th harmonic number), which makes the FDR
    bound valid under ARBITRARY dependence between tests, not just PRDS -
    the robustness check PROTOCOL.md §4/H5 step 6 requires because
    pairwise cointegration tests sharing tickers/common factors are
    dependent in an unknown way. Always at least as conservative as BH
    (c(n) >= 1 for n >= 1).
    """
    n = len(np.asarray(p_values))
    c_n = np.sum(1.0 / np.arange(1, n + 1)) if n > 0 else 1.0
    return _step_up_procedure(p_values, alpha, correction_factor=c_n)


def _step_up_procedure(p_values, alpha: float, correction_factor: float) -> np.ndarray:
    """Shared Benjamini step-up logic: threshold_i = (i / (n * correction_factor)) * alpha
    for the i-th smallest p-value (1-indexed); reject all ranks up to and
    including the LARGEST rank whose p-value is <= its own threshold.
    correction_factor=1.0 gives Benjamini-Hochberg; correction_factor =
    the n-th harmonic number gives Benjamini-Yekutieli."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)

    order = np.argsort(p, kind="stable")
    ranked_p = p[order]
    thresholds = (np.arange(1, n + 1) / (n * correction_factor)) * alpha
    below = ranked_p <= thresholds

    reject = np.zeros(n, dtype=bool)
    if below.any():
        max_rank = np.max(np.nonzero(below)[0])
        reject_sorted = np.zeros(n, dtype=bool)
        reject_sorted[: max_rank + 1] = True
        reject[order] = reject_sorted
    return reject


def brute_force_cointegration_screen(
    price_index: pd.DataFrame,
    p_value_max: float = config.EG_PVALUE_MAX,
    half_life_range: tuple[float, float] = config.HALF_LIFE_RANGE_DAYS,
    fdr_alpha: float = config.FDR_ALPHA,
    top_n: int = config.TOP_PAIRS,
) -> dict:
    """
    Brute-force comparator (PROTOCOL.md §4/H5 step 6): the same
    Engle-Granger + half-life screen as Variant B, run on EVERY pair of
    the formation universe (not just intra-cluster candidates) - the
    baseline that clustering's discovery quality is compared against.
    Reuses engle_granger_pair; no duplicated EG logic.

    Returns:
      "all_pairs": DataFrame, one row per pair tested - ticker_1,
        ticker_2, t_stat, p_value, half_life_days, residual_std (the EG
        residual's own std, see engle_granger_pair - the scale a caller
        must use for this pair's trigger, NOT formation.spread_sigma's
        SSD-style sigma), passes_filter (p and half-life filter,
        PROTOCOL.md §4/H5 step 5), bh_survivor,
        by_survivor (Benjamini-Hochberg / Benjamini-Yekutieli at
        fdr_alpha, computed over the FULL p-value distribution of all
        tested pairs - BH/BY need every p-value to control the FDR
        correctly, not just the ones already passing the raw filter).
      "n_tests": total pairs tested, i.e. C(len(tickers), 2).
      "expected_false_positives": n_tests * fdr_alpha - the number of
        spuriously "significant" pairs expected under a global null of no
        true cointegration anywhere (PROTOCOL.md §4/H5 step 6, point 2:
        "falsi positivi attesi sotto ipotesi nulla globale"; ~6,200 for
        the ~125,000 pairs of the full S&P 500 universe at alpha=0.05).
      "n_bh_survivors" / "n_by_survivors": count of pairs surviving
        Benjamini-Hochberg / Benjamini-Yekutieli at fdr_alpha
        (PROTOCOL.md §4/H5 step 6: "conteggio sopravvissuti"; by <= bh
        always, since BY is strictly more conservative - the gap between
        the two is the point of reporting both, "commentare la
        differenza").
      "top_n": survivors of BOTH the raw filter and Benjamini-Hochberg,
        ranked by ascending p-value, truncated to top_n.
    """
    tickers = sorted(price_index.columns)
    columns = ["ticker_1", "ticker_2", "t_stat", "p_value", "half_life_days", "residual_std"]
    rows = [
        {"ticker_1": i, "ticker_2": j, **engle_granger_pair(price_index, i, j)}
        for i, j in itertools.combinations(tickers, 2)
    ]
    all_pairs = pd.DataFrame(rows, columns=columns)
    n_tests = len(all_pairs)

    if n_tests == 0:
        all_pairs = all_pairs.assign(passes_filter=[], bh_survivor=[], by_survivor=[])
        return {
            "all_pairs": all_pairs,
            "n_tests": 0,
            "expected_false_positives": 0.0,
            "n_bh_survivors": 0,
            "n_by_survivors": 0,
            "top_n": all_pairs,
        }

    p_values = all_pairs["p_value"].to_numpy()
    all_pairs["passes_filter"] = _passes_eg_filter(all_pairs, p_value_max, half_life_range)
    all_pairs["bh_survivor"] = benjamini_hochberg(p_values, fdr_alpha)
    all_pairs["by_survivor"] = benjamini_yekutieli(p_values, fdr_alpha)

    survivors = all_pairs[all_pairs["passes_filter"] & all_pairs["bh_survivor"]]
    ranked = survivors.sort_values("p_value", kind="stable").reset_index(drop=True)

    return {
        "all_pairs": all_pairs,
        "n_tests": n_tests,
        "expected_false_positives": n_tests * fdr_alpha,
        "n_bh_survivors": int(all_pairs["bh_survivor"].sum()),
        "n_by_survivors": int(all_pairs["by_survivor"].sum()),
        "top_n": ranked.iloc[:top_n],
    }


def select_pairs_via_clustering(
    returns: pd.DataFrame,
    price_index: pd.DataFrame,
    n_components: int = config.PCA_N_COMPONENTS,
    top_n: int = config.TOP_PAIRS,
) -> dict:
    """
    Full H5 pipeline (PROTOCOL.md §4/H5 steps 1-5): pca_loadings ->
    cluster_tickers -> {Variant A, Variant B}. Orchestrator only, no new
    logic (mirrors formation.select_pairs_for_formation's role for the
    GGR-SSD arm).

    returns: formation-period simple daily returns (e.g.
        formation.load_formation_returns's output).
    price_index: the SAME formation period's normalized price index
        (formation.normalized_price_indices(returns)) - passed separately
        so a caller that also runs the GGR-SSD arm can reuse one
        already-computed price_index instead of paying for it twice.

    Returns {"clustering": cluster_tickers(...)'s output, "variant_a":
    ssd_intra_cluster_ranking(...), "variant_b":
    cointegration_intra_cluster_ranking(...)}.
    """
    loadings = pca_loadings(returns, n_components)
    clustering = cluster_tickers(loadings)
    labels = clustering["labels"]
    variant_a = ssd_intra_cluster_ranking(price_index, labels, top_n)
    variant_b = cointegration_intra_cluster_ranking(price_index, labels, top_n=top_n)
    return {"clustering": clustering, "variant_a": variant_a, "variant_b": variant_b}
