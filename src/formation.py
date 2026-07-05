"""
formation.py — Formation period GGR: indice di prezzo normalizzato per
titolo, matrice SSD pairwise, selezione top-N e controllo 101-120.

PROTOCOL.md §2.2:
- P*_it = prod(1+r) sul formation, P*_i0=1 (build_price_index di trading.py,
  riusata qui, non duplicata).
- SSD_ij = sum_t (P*_it - P*_jt)^2 sui giorni del formation.
- sigma_ij = std(spread) nel formation (spread = P*_i - P*_j), da passare
  come input esterno e congelato a simulate_pair_same_day/wait_one_day.

Tie-break SSD (non specificato da GGR, scelta dichiarata qui): a pari
merito, ordine alfabetico crescente sulla coppia di ticker (prima ticker_1,
poi ticker_2). Deterministico e riproducibile: itertools.combinations su
ticker ordinati alfabeticamente gia' produce le coppie in quest'ordine, e un
sort stabile (kind="stable") su ["ssd", "ticker_1", "ticker_2"] preserva
l'ordinamento alfabetico a parita' di ssd.

sigma=0 (spread costante nel formation, es. due serie identiche o dati
degeneri): la coppia viene ESCLUSA dal ranking con un warning. Motivazione:
la soglia di apertura e' k*sigma; con sigma=0 la soglia e' 0 e qualunque
spread diverso da zero (compreso il rumore numerico) farebbe scattare
un'apertura, un comportamento degenere e non interpretabile come segnale.
Escludere a monte evita di propagare il caso patologico a trading.py.
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
    """returns: DataFrame (righe=giorni del formation, colonne=ticker),
    rendimenti giornalieri semplici. Ritorna DataFrame (righe 0..n, colonne
    =ticker) con l'indice normalizzato P*_i0=1, una colonna per ticker via
    build_price_index (riuso, nessuna duplicazione della logica)."""
    out = {tkr: build_price_index(returns[tkr].to_numpy()) for tkr in returns.columns}
    return pd.DataFrame(out)


def ssd_matrix(price_index: pd.DataFrame) -> pd.DataFrame:
    """SSD_ij = sum_t (P*_it - P*_jt)^2 su tutte le righe di price_index
    (incluso il giorno 0 = ancora, dove tutte le serie valgono 1 e quindi
    contribuiscono 0 alla somma: non altera il ranking, incluso per
    semplicita' del calcolo vettorizzato)."""
    tickers = price_index.columns.tolist()
    P = price_index.to_numpy()  # righe=giorni, colonne=ticker
    diff = P[:, :, None] - P[:, None, :]
    ssd = np.sum(diff ** 2, axis=0)
    return pd.DataFrame(ssd, index=tickers, columns=tickers)


def spread_sigma(price_index: pd.DataFrame, i: str, j: str) -> float:
    """sigma dello spread P*_i - P*_j nel formation (ddof=0: GGR stima sigma
    sull'intera finestra osservata, non su un campione da cui inferire una
    popolazione piu' ampia)."""
    spread = price_index[i] - price_index[j]
    return float(spread.std(ddof=0))


def rank_pairs(price_index: pd.DataFrame) -> pd.DataFrame:
    """
    Tutte le coppie i<j ordinate per SSD crescente. Colonne: ticker_1,
    ticker_2, ssd, sigma. Indice = rank (1-indexed, coerente con la notazione
    "coppie 101-120" del protocollo). Le coppie con sigma=0 sono escluse con
    un warning (vedi docstring di modulo).
    """
    tickers = sorted(price_index.columns)
    P = price_index[tickers]
    ssd = ssd_matrix(P)

    rows = []
    for i, j in itertools.combinations(tickers, 2):
        sigma = spread_sigma(P, i, j)
        if sigma == 0.0:
            warnings.warn(
                f"coppia ({i},{j}) esclusa dal ranking: sigma dello spread "
                "nel formation e' zero (spread costante, trigger degenere)."
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
    Ritorna {"top_5", "top_20", "control"}: sotto-tabelle di rank_pairs,
    ciascuna con la colonna "sigma" (stimata sullo stesso formation period,
    da passare congelata al trading period).

    Se il numero di coppie disponibili (dopo l'esclusione sigma=0) e'
    inferiore a top_n o control_range[0], le sotto-tabelle risultano piu'
    corte del nominale (anche vuote per "control"): non e' un errore, e' un
    universo insufficiente a riempire il portafoglio richiesto (puo'
    succedere su un golden-set ridotto, PROTOCOL.md non lo esclude
    esplicitamente ma presuppone un universo ampio) — nessuna eccezione,
    nessun padding artificiale.
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
                warnings.warn(f"{t}: nessun file prezzi in cache, escluso dal formation.")
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
            warnings.warn(f"{t}: storico incompleto nel formation period, escluso.")
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
                f"{t}: rendimento giornaliero oltre {config.MAX_ABS_DAILY_RETURN:.0%} "
                "nel formation period, escluso (dato verosimilmente corrotto)."
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
                f"{t}: prezzo congelato per oltre {config.MAX_CONSECUTIVE_FROZEN_DAYS} giorni "
                "consecutivi nel formation period, escluso (dato verosimilmente corrotto)."
            )
        returns = returns.drop(columns=frozen)
    return returns


def load_formation_returns(
    tickers: list[str], formation_start, formation_end, price_dir: Path = PRICES_DIR,
) -> pd.DataFrame:
    """
    Rendimenti giornalieri semplici (Adj Close) di `tickers` nel formation
    period [formation_start, formation_end], allineati al calendario di
    mercato (proxy: indice ^GSPC cachato in price_dir, riuso di
    data.prices._reference_days, nessuna duplicazione). Un ticker con
    storico incompleto in questa finestra o assente dalla cache viene
    escluso con un warning (il filtro "no-trade days" dell'universo e'
    responsabilita' di data/prices.py; qui si applica la stessa logica
    localmente, a difesa, sul sotto-insieme di ticker richiesto).

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
    """Pipeline completa: ticker + finestra di formation -> {top_5, top_20,
    control}. Orchestratore di load_formation_returns + normalized_price_indices
    + select_portfolios."""
    returns = load_formation_returns(tickers, formation_start, formation_end, price_dir)
    price_index = normalized_price_indices(returns)
    return select_portfolios(price_index, top_n_small, top_n, control_range)
