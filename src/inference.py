"""
inference.py — Inferenza statistica GGR: t-test Newey-West sulla media,
block bootstrap stazionario per l'IC 95%, regressione fattoriale (alpha,
loadings, t-stat NW) contro data/factors.py.

PROTOCOL.md §2.1/§2.3: "Inferenza: Newey-West 6 lag su serie mensile".
§4/H1: "CI 95% con block bootstrap stazionario (blocco medio 6 mesi, 10.000
repliche)".

Newey-West: NON reimplementiamo lo stimatore HAC a mano. Un t-test HAC sulla
media di una serie e' algebricamente una regressione OLS di y su una sola
costante con covarianza HAC — e' l'approccio standard (Newey & West, 1987) e
lo deleghiamo a statsmodels (sm.OLS(..., cov_type="HAC")).

Block bootstrap stazionario (Politis & Romano, 1994): non c'e' un
equivalente diretto in statsmodels/scipy nell'ambiente di questo progetto,
quindi lo implementiamo qui: blocchi di lunghezza GEOMETRICA (media =
mean_block_months) campionati con wraparound circolare (proprieta'
"stazionaria": a differenza del block bootstrap a blocchi fissi, la serie
ricampionata resta stazionaria in senso debole se lo e' l'originale).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config


def newey_west_mean_test(returns, lags: int = config.NW_LAGS) -> dict:
    """
    Test t sulla media di una serie di rendimenti mensili, con errori
    standard Newey-West (HAC) a `lags` ritardi (default 6, congelato
    PROTOCOL.md §2.1). Implementato come OLS di y su una costante con
    cov_type="HAC": e' il modo standard di ottenere un t-test HAC sulla
    media, non uno stimatore ad hoc.

    Ritorna: mean, se, t_stat, p_value (two-sided, H0: media=0), n.
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
    IC bootstrap stazionario (Politis & Romano 1994) sulla media della
    serie. Ogni replica ricostruisce una serie sintetica di lunghezza n
    concatenando blocchi di lunghezza geometrica casuale (parametro p =
    1/mean_block_months, cosi' E[lunghezza blocco] = mean_block_months),
    campionati con wraparound circolare sull'indice originale (start casuale
    in [0,n), poi indici consecutivi modulo n). n_reps repliche indipendenti
    (rng seedato: riproducibile). CI = percentili (1-ci)/2 e 1-(1-ci)/2 della
    distribuzione delle medie bootstrap.
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
    Regressione fattoriale (PROTOCOL.md §2.4.3): rendimenti mensili in
    eccesso (gia' al netto di RF, calcolo a monte) su FF3 + Momentum +
    ST-Reversal, errori standard Newey-West a `lags` ritardi.

    excess_returns: Series indicizzata per mese (Timestamp inizio mese),
        stesso formato di data/factors.py.
    factors: DataFrame come restituito da data.factors.load_factors.

    Allineamento: inner join sull'indice (mese). Un mese presente in
    excess_returns ma assente in factors (es. la Ken French Data Library non
    ancora aggiornata all'ultimo mese) viene scartato, non genera NaN nella
    regressione.

    Ritorna: alpha, alpha_se, alpha_t, loadings (dict fattore->beta),
    loadings_se, loadings_t, n_obs, r_squared.
    """
    y_series = excess_returns if isinstance(excess_returns, pd.Series) else pd.Series(excess_returns)
    aligned = pd.concat(
        [y_series.rename("y"), factors[list(factor_cols)]], axis=1, join="inner"
    ).dropna()

    y = aligned["y"].to_numpy()
    # has_constant="add" esplicito: se per una finestra corta/degenere un
    # fattore risultasse a varianza zero, sm.add_constant di default lo
    # scambierebbe per una costante gia' presente e ne salterebbe una nuova,
    # disallineando params (5 valori) rispetto a names (6 nomi) in modo
    # silenzioso. "add" garantisce sempre una colonna in piu' per l'alpha.
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
    Decomposizione alpha long/short (PROTOCOL.md §2.4, punto 2 — test di
    falsificazione obbligatorio): la STESSA regressione fattoriale di
    factor_regression, applicata separatamente ai rendimenti mensili della
    sola gamba LONG e della sola gamba SHORT del portafoglio (nessuna
    formula nuova: e' un doppio richiamo a factor_regression, una per gamba).

    long_returns, short_returns: Series indicizzate per mese, stesso formato
        di excess_returns in factor_regression (tipicamente ottenute
        aggregando a portafoglio, con la stessa logica di src/returns.py, i
        payoff della sola gamba long/short di ciascuna coppia).

    Atteso (Tabella 7 GGR, top-20 — invariante qualitativo del Gate 1,
    PROTOCOL.md §3): alpha della gamba SHORT negativo e significativo
    (shortare un titolo con alpha negativo contribuisce positivamente alla
    strategia), alpha della gamba LONG vicino a zero e non significativo:
    la profittabilita' della coppia trading viene dalla gamba short, non
    dalla long. Se nella replica risultasse il contrario, e' un campanello
    d'allarme di un errore di segno o di costruzione (PROTOCOL.md §2.4.2),
    non un risultato da accettare a valore nominale.

    Ritorna: {"long": factor_regression(...), "short": factor_regression(...)}.
    """
    return {
        "long": factor_regression(long_returns, factors, factor_cols, lags),
        "short": factor_regression(short_returns, factors, factor_cols, lags),
    }


def decile_of_returns(returns: pd.Series, n_deciles: int = 10) -> pd.Series:
    """
    Assegna a ciascun ticker (indice di `returns`) il decile di appartenenza
    del suo rendimento (1 = decile piu' basso, n_deciles = piu' alto), via
    pd.qcut su bin di uguale frequenza. `returns` e' tipicamente il
    rendimento dell'universo nel MESE PRECEDENTE il formation (PROTOCOL.md
    §2.4, punto 1: il decile-matching si fa sul rendimento pregresso, non su
    quantita' calcolate nel trading period, per non introdurre look-ahead).

    duplicates="drop": se piu' titoli hanno rendimenti identici tali da non
    permettere n_deciles bin distinti, si ottengono meno bin invece di un
    ValueError — un edge case dei dati, non un errore della pipeline.
    """
    return pd.qcut(returns, n_deciles, labels=False, duplicates="drop") + 1


def _sample_same_decile(ticker: str, deciles: pd.Series, rng: np.random.Generator) -> str:
    """Un titolo casuale dallo stesso decile di `ticker` (compreso `ticker`
    stesso se e' l'unico membro del suo decile: nessuna alternativa possibile)."""
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
    Falsificazione bootstrap a coppie casuali decile-matched (PROTOCOL.md
    §2.4, punto 1 — test di falsificazione obbligatorio, non opzionale):
    atteso rendimento bootstrap ~0 o negativo; se risultasse positivo, la
    pipeline ha un bug o il presunto profitto e' in realta' reversal
    mascherato, non convergenza di coppie cointegrate.

    Per n_reps ripetizioni, sostituisce OGNI coppia vera (ticker_1, ticker_2)
    in `selected_pairs` con una coppia di titoli fittizi, ciascuno estratto
    CASUALMENTE dallo stesso decile di rendimento del MESE PRECEDENTE il
    formation (prior_month_returns, indicizzata per ticker) della
    rispettiva gamba originale — ticker_1 sostituito da un titolo dello
    stesso decile di ticker_1, ticker_2 da un titolo dello stesso decile di
    ticker_2, indipendentemente.

    "Stesse date di trigger/evento" (PROTOCOL.md): questa funzione genera
    SOLO l'assegnazione dei titoli fittizi per ciascuna replica; il chiamante
    e' responsabile di rieseguire la simulazione (simulate_pair_same_day o
    _wait_one_day) sui rendimenti dei titoli sostituiti MA sullo stesso
    trading period/stessa finestra della coppia vera — non si generano qui
    nuove date, si riusa la simulazione esistente con input diversi.

    selected_pairs: coppie vere del run (formato (ticker_1, ticker_2)).
    prior_month_returns: Series ticker -> rendimento del mese precedente il
        formation, sull'universo COMPLETO (non solo le coppie selezionate:
        serve per costruire i decili e pescare le sostituzioni).

    Ritorna: lista di n_reps liste di coppie fittizie (stesso ordine e
    stessa lunghezza di selected_pairs, una lista per replica).
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
