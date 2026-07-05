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
    """
    days = _reference_days(formation_start, formation_end)
    prices: dict[str, pd.Series] = {}
    for t in tickers:
        p = price_dir / f"{t}.parquet"
        if not p.exists():
            warnings.warn(f"{t}: nessun file prezzi in cache, escluso dal formation.")
            continue
        s = pd.read_parquet(p, columns=["Adj Close"]).reindex(days)["Adj Close"]
        if s.isna().any():
            warnings.warn(f"{t}: storico incompleto nel formation period, escluso.")
            continue
        prices[t] = s
    price_df = pd.DataFrame(prices)
    return price_df.pct_change().iloc[1:]


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
