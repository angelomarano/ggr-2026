"""
formation.py — GGR formation period: normalized price index per ticker,
pairwise SSD matrix, top-N selection and 101-120 control.

PROTOCOL.md §2.2:
- P*_it = prod(1+r) over the formation period, P*_i0=1 (build_price_index
  from trading.py, reused here, not duplicated).
- SSD_ij = sum_t (P*_it - P*_jt)^2 over the formation days.
- sigma_ij = std(spread) over the formation period (spread = P*_i - P*_j),
  passed as a frozen external input to simulate_pair_same_day/wait_one_day.

SSD tie-break (unspecified by GGR, choice declared here): on a tie,
ascending alphabetical order on the ticker pair (ticker_1 first, then
ticker_2). Deterministic and reproducible: itertools.combinations over
alphabetically sorted tickers already produces pairs in this order, and a
stable sort (kind="stable") on ["ssd", "ticker_1", "ticker_2"] preserves the
alphabetical ordering on ties in ssd.

sigma=0 (constant spread over the formation period, e.g. two identical
series or degenerate data): the pair is EXCLUDED from the ranking with a
warning. Rationale: the opening threshold is k*sigma; with sigma=0 the
threshold is 0 and any nonzero spread (including numerical noise) would
trigger an opening, a degenerate behavior not interpretable as a signal.
Excluding it upstream avoids propagating the pathological case to trading.py.
"""
from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data.prices import RAW as PRICES_DIR, _reference_days
from src.trading import build_price_index


def normalized_price_indices(returns: pd.DataFrame) -> pd.DataFrame:
    """returns: DataFrame (rows=formation days, columns=ticker), simple
    daily returns. Returns a DataFrame (rows 0..n, columns=ticker) with the
    normalized index P*_i0=1, one column per ticker via build_price_index
    (reused, no duplicated logic)."""
    out = {tkr: build_price_index(returns[tkr].to_numpy()) for tkr in returns.columns}
    return pd.DataFrame(out)


def ssd_matrix(price_index: pd.DataFrame) -> pd.DataFrame:
    """SSD_ij = sum_t (P*_it - P*_jt)^2 over all rows of price_index
    (including day 0 = anchor, where every series equals 1 and thus
    contributes 0 to the sum: it doesn't alter the ranking, included for
    simplicity of the vectorized computation)."""
    tickers = price_index.columns.tolist()
    P = price_index.to_numpy()  # rows=days, columns=ticker
    diff = P[:, :, None] - P[:, None, :]
    ssd = np.sum(diff ** 2, axis=0)
    return pd.DataFrame(ssd, index=tickers, columns=tickers)


def spread_sigma(price_index: pd.DataFrame, i: str, j: str) -> float:
    """sigma of the spread P*_i - P*_j over the formation period (ddof=0: GGR
    estimates sigma over the whole observed window, not from a sample used
    to infer a larger population)."""
    spread = price_index[i] - price_index[j]
    return float(spread.std(ddof=0))


def rank_pairs(price_index: pd.DataFrame) -> pd.DataFrame:
    """
    All pairs i<j sorted by ascending SSD. Columns: ticker_1, ticker_2, ssd,
    sigma. Index = rank (1-indexed, consistent with the protocol's
    "pairs 101-120" notation). Pairs with sigma=0 are excluded with a
    warning (see module docstring).
    """
    tickers = sorted(price_index.columns)
    P = price_index[tickers]
    ssd = ssd_matrix(P)

    rows = []
    for i, j in itertools.combinations(tickers, 2):
        sigma = spread_sigma(P, i, j)
        if sigma == 0.0:
            warnings.warn(
                f"pair ({i},{j}) excluded from ranking: the spread's sigma "
                "over the formation period is zero (constant spread, degenerate trigger)."
            )
            continue
        rows.append({"ticker_1": i, "ticker_2": j, "ssd": ssd.loc[i, j], "sigma": sigma})

    ranked = pd.DataFrame(rows, columns=["ticker_1", "ticker_2", "ssd", "sigma"])
    ranked = ranked.sort_values(["ssd", "ticker_1", "ticker_2"], kind="stable").reset_index(drop=True)
    ranked.index = ranked.index + 1
    ranked.index.name = "rank"
    return ranked


def select_portfolios(
    price_index: pd.DataFrame,
    top_n_small: int = config.TOP_PAIRS_SMALL,
    top_n: int = config.TOP_PAIRS,
    control_range: tuple[int, int] = config.CONTROL_PAIRS_RANGE,
) -> dict[str, pd.DataFrame]:
    """
    Returns {"top_5", "top_20", "control"}: sub-tables of rank_pairs, each
    with the "sigma" column (estimated on the same formation period, to be
    passed frozen to the trading period).

    If the number of available pairs (after excluding sigma=0) is below
    top_n or control_range[0], the sub-tables come out shorter than nominal
    (possibly empty for "control"): this is not an error, it's a universe
    insufficient to fill the requested portfolio (can happen on a reduced
    golden set; PROTOCOL.md doesn't explicitly exclude it but assumes a
    large universe) — no exception, no artificial padding.
    """
    ranked = rank_pairs(price_index)
    lo, hi = control_range
    return {
        "top_5": ranked.iloc[:top_n_small],
        "top_20": ranked.iloc[:top_n],
        "control": ranked.loc[(ranked.index >= lo) & (ranked.index <= hi)],
    }


def _max_consecutive_frozen_run(prices: pd.Series) -> int:
    """
    Longest run of consecutive bit-identical values in `prices` (exact
    equality, not approximate). NaN never extends a run (NaN != NaN), so a
    gap never counts as "frozen". Returns 0 for an empty series, 1 if no
    two consecutive values are ever equal.
    """
    values = prices.to_numpy()
    n = len(values)
    if n == 0:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, n):
        if values[i] == values[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def _load_returns_window(
    tickers: list[str], start, end, price_dir: Path, require_complete: bool,
) -> pd.DataFrame:
    """
    Shared loader for both formation and trading windows: daily simple
    returns for `tickers` over every reference trading day in [start, end]
    inclusive. Uses the reference trading day immediately before `start` as
    the pct_change anchor, so the result has exactly one row per reference
    day in [start, end] - not one fewer, which a naive reindex+pct_change
    over [start, end] alone would give (the first day would have no prior
    price to compute a return from).

    require_complete=True (formation): a ticker missing from the cache, or
    with any NaN return in the window, is dropped with a warning (GGR's
    "no-trade days" filter, PROTOCOL.md §1.2.3).
    require_complete=False (trading): nothing is dropped for missing data;
    gaps (mid-period delisting) stay as NaN so src/trading.py's explicit
    NaN handling applies. A ticker absent from the cache entirely is still
    skipped (there is nothing to simulate for it).
    """
    days = _reference_days(start, end)
    wide_days = _reference_days(pd.Timestamp(start) - pd.Timedelta(days=20), end)
    anchor_pos = wide_days.get_indexer([days[0]])[0]
    if anchor_pos <= 0:
        raise ValueError(f"no reference trading day before {start} to anchor the first return")
    full_days = wide_days[anchor_pos - 1 : anchor_pos].append(days)

    prices: dict[str, pd.Series] = {}
    for t in tickers:
        p = price_dir / f"{t}.parquet"
        if not p.exists():
            if require_complete:
                warnings.warn(f"{t}: no price file in cache, excluded from formation.")
            continue
        s = pd.read_parquet(p, columns=["Adj Close"]).reindex(full_days)["Adj Close"]
        prices[t] = s

    price_df = pd.DataFrame(prices)
    # fill_method=None: pandas' default forward-fills NaN prices before
    # differencing, which would turn a real gap (delisting) into a fake
    # zero-return day instead of the NaN that must reach trading.py / the
    # formation completeness filter.
    returns = price_df.pct_change(fill_method=None).iloc[1:]
    if require_complete:
        incomplete = [c for c in returns.columns if returns[c].isna().any()]
        for t in incomplete:
            warnings.warn(f"{t}: incomplete history in the formation period, excluded.")
        returns = returns.drop(columns=incomplete)

        # Post-hoc data-quality filter (config.MAX_ABS_DAILY_RETURN, see
        # DEVIATIONS.md): a small number of tickers have severely corrupted
        # Yahoo data (recycled ticker symbols reassigned to illiquid
        # OTC/penny-stock entities after the original company delisted),
        # producing implausible daily returns. Causal by construction: this
        # only ever looks at `returns`, which was built solely from
        # [start, end] above - never at data outside this run's own
        # formation window.
        extreme = [c for c in returns.columns if (returns[c].abs() > config.MAX_ABS_DAILY_RETURN).any()]
        for t in extreme:
            warnings.warn(
                f"{t}: daily return beyond {config.MAX_ABS_DAILY_RETURN:.0%} "
                "in the formation period, excluded (likely corrupted data)."
            )
        returns = returns.drop(columns=extreme)

        # Second post-hoc data-quality filter (config.MAX_CONSECUTIVE_FROZEN_DAYS,
        # see DEVIATIONS.md): some corrupted tickers have long runs of
        # bit-identical ("frozen"/stale) Adj Close within the formation
        # window instead of an outright jump - a flat price has zero daily
        # return, so it never trips the extreme-return filter above, but it
        # produces an artificially low SSD against any other low-volatility
        # ticker (a constant normalized price index trivially "matches"
        # anything that barely moves). Deliberately price-only, not Volume:
        # the Volume field on these same tickers is itself an unreliable,
        # seemingly artifactual signal. Causal: computed only on the
        # formation-window's own rows (the anchor day before `start` is
        # excluded), never on data outside [start, end].
        window_prices = price_df.iloc[1:]
        frozen = [
            c for c in returns.columns
            if _max_consecutive_frozen_run(window_prices[c]) > config.MAX_CONSECUTIVE_FROZEN_DAYS
        ]
        for t in frozen:
            warnings.warn(
                f"{t}: price frozen for more than {config.MAX_CONSECUTIVE_FROZEN_DAYS} consecutive "
                "days in the formation period, excluded (likely corrupted data)."
            )
        returns = returns.drop(columns=frozen)
    return returns


def load_formation_returns(
    tickers: list[str], formation_start, formation_end, price_dir: Path = PRICES_DIR,
) -> pd.DataFrame:
    """
    Simple daily returns (Adj Close) of `tickers` in the formation period
    [formation_start, formation_end], aligned to the market calendar
    (proxy: ^GSPC index cached in price_dir, reusing
    data.prices._reference_days, no duplication). A ticker with incomplete
    history in this window or absent from the cache is excluded with a
    warning (the universe's "no-trade days" filter is data/prices.py's
    responsibility; here the same logic is applied locally, defensively,
    on the requested subset of tickers).

    One row per reference trading day in [formation_start, formation_end]
    (see _load_returns_window).
    """
    return _load_returns_window(tickers, formation_start, formation_end, price_dir, require_complete=True)


def load_trading_returns(
    tickers: list[str], trading_start, trading_end, price_dir: Path = PRICES_DIR,
) -> pd.DataFrame:
    """
    Same as load_formation_returns but for the TRADING period: no
    completeness filter, mid-period NaN (delisting) is preserved rather
    than dropping the ticker, since src/trading.py's simulate_pair_same_day
    / simulate_pair_wait_one_day handle a NaN mid-series explicitly
    (PROTOCOL.md §1.4/§2.2). Dropping incomplete tickers here would defeat
    that handling by silently excluding exactly the cases it exists for.
    """
    return _load_returns_window(tickers, trading_start, trading_end, price_dir, require_complete=False)


def select_pairs_for_formation(
    tickers: list[str],
    formation_start,
    formation_end,
    price_dir: Path = PRICES_DIR,
    top_n_small: int = config.TOP_PAIRS_SMALL,
    top_n: int = config.TOP_PAIRS,
    control_range: tuple[int, int] = config.CONTROL_PAIRS_RANGE,
) -> dict[str, pd.DataFrame]:
    """Full pipeline: tickers + formation window -> {top_5, top_20,
    control}. Orchestrates load_formation_returns + normalized_price_indices
    + select_portfolios."""
    returns = load_formation_returns(tickers, formation_start, formation_end, price_dir)
    price_index = normalized_price_indices(returns)
    return select_portfolios(price_index, top_n_small, top_n, control_range)
