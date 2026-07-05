"""
returns.py — Aggregazione a portafoglio dei payoff di coppia (output di
simulate_pair_same_day / simulate_pair_wait_one_day) e composizione dei 6
portafogli mensili sovrapposti in un'unica serie (PROTOCOL.md §2.2).

Due livelli:
1. aggregate_portfolio_run: DENTRO un singolo run (una formation date), somma
   i payoff giornalieri di tutte le coppie selezionate e calcola
   - return on COMMITTED capital = somma payoff / N coppie selezionate
     (il denominatore e' la dimensione NOMINALE del portafoglio, es. 20:
     va passato esplicitamente come n_selected, non dedotto da quante
     coppie sono effettivamente disponibili — vedi docstring della funzione);
   - return on EMPLOYED capital = somma payoff / N coppie APERTE quel giorno
     specifico; se zero coppie sono aperte, il rendimento e' 0.0 (mai NaN,
     mai una divisione per zero silenziosa).
2. compound_to_monthly + combine_overlapping_portfolios: portano la serie
   giornaliera di UN portafoglio a rendimenti mensili (compounding entro il
   mese di calendario), poi combinano i rendimenti mensili di piu' run
   sfalsati con la media alla Jegadeesh-Titman (media semplice tra i
   portafogli attivi in ciascun mese di calendario).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pair_employed_mask(trades: list[dict], n_days: int) -> np.ndarray:
    """
    Maschera booleana (lunghezza n_days, indice 0..n_days-1 <-> giorno
    1..n_days) che marca i giorni in cui la coppia ha una posizione aperta
    "impiegata" ai fini del capitale employed.

    Convenzione (coerente con trading.py: nessun payoff il giorno di
    apertura, i pesi iniziano a mark-to-market SOLO dal giorno successivo):
    una coppia aperta al giorno open_day e chiusa al giorno close_day e'
    "employed" nei giorni (open_day, close_day] inclusivo — cioe' esclude il
    giorno di apertura stesso (payoff=0 per costruzione, nessun capitale
    ancora marcato a mercato) e include il giorno di chiusura (l'ultimo
    payoff viene comunque realizzato quel giorno prima della chiusura).

    Eventi "missed" (wait-one-day, occasione persa) sono ignorati: non
    aprono mai una posizione, non contribuiscono capitale employed.
    """
    mask = np.zeros(n_days, dtype=bool)
    open_day = None
    for ev in trades:
        if ev["event"] == "open":
            open_day = ev["day"]
        elif ev["event"] == "close" and open_day is not None:
            close_day = ev["day"]
            mask[open_day:close_day] = True
            open_day = None
    return mask


def aggregate_portfolio_run(
    pair_results: dict[str, dict], n_days: int, n_selected: int | None = None
) -> pd.DataFrame:
    """
    pair_results: {pair_id: risultato di simulate_pair_same_day/wait_one_day}
    per le coppie del portafoglio in UN run (incluse quelle mai aperte:
    contribuiscono payoff=0 tutti i giorni e nessun giorno "employed").
    n_days: lunghezza del trading period (deve combaciare per tutte le
    coppie: stesso run, stesso trading period).
    n_selected: dimensione NOMINALE del portafoglio (es. config.TOP_PAIRS=20
    per il top-20; puo' differire da len(pair_results) se l'universo non
    fornisce abbastanza coppie candidate, PROTOCOL.md §2.1 presuppone un
    universo ampio ma src/formation.py puo' restituire meno coppie su un
    golden-set ridotto). Default: len(pair_results) se non specificato.
    Se n_selected == 0 (nessuna coppia selezionabile), il committed return e'
    0.0 per ogni giorno anziche' una ZeroDivisionError: un portafoglio vuoto
    non genera rendimento, non e' un caso indefinito.

    Ritorna DataFrame indicizzato 0..n_days-1 (giorno = indice+1) con colonne:
      payoff_sum          somma dei payoff giornalieri su tutte le coppie
      n_open              numero di coppie aperte quel giorno (employed)
      committed_return    payoff_sum / n_selected (0.0 se n_selected==0)
      employed_return     payoff_sum / n_open, 0.0 se n_open==0 (mai NaN)

    Also returns long_payoff_sum/short_payoff_sum and their committed-capital
    returns (long_committed_return, short_committed_return), summing each
    pair's daily_long_payoff/daily_short_payoff (PROTOCOL.md §2.4, long/short
    alpha decomposition — feed these into inference.long_short_leg_regression
    after compound_to_monthly). Same n_selected denominator and zero-division
    handling as committed_return; requires pair_results built from a
    simulate_pair_* version that returns daily_long_payoff/daily_short_payoff.
    """
    if n_selected is None:
        n_selected = len(pair_results)

    payoff_sum = np.zeros(n_days)
    long_payoff_sum = np.zeros(n_days)
    short_payoff_sum = np.zeros(n_days)
    n_open = np.zeros(n_days, dtype=int)
    for res in pair_results.values():
        daily_payoff = np.asarray(res["daily_payoff"])
        assert len(daily_payoff) == n_days, "tutte le coppie del run devono avere lo stesso trading period"
        payoff_sum += daily_payoff
        long_payoff_sum += np.asarray(res["daily_long_payoff"])
        short_payoff_sum += np.asarray(res["daily_short_payoff"])
        n_open += pair_employed_mask(res["trades"], n_days).astype(int)

    if n_selected > 0:
        committed_return = payoff_sum / n_selected
        long_committed_return = long_payoff_sum / n_selected
        short_committed_return = short_payoff_sum / n_selected
    else:
        committed_return = np.zeros(n_days)
        long_committed_return = np.zeros(n_days)
        short_committed_return = np.zeros(n_days)

    employed_return = np.divide(
        payoff_sum, n_open, out=np.zeros_like(payoff_sum), where=n_open > 0
    )

    return pd.DataFrame({
        "payoff_sum": payoff_sum,
        "n_open": n_open,
        "committed_return": committed_return,
        "employed_return": employed_return,
        "long_payoff_sum": long_payoff_sum,
        "short_payoff_sum": short_payoff_sum,
        "long_committed_return": long_committed_return,
        "short_committed_return": short_committed_return,
    })


def compound_to_monthly(daily_returns, dates) -> pd.Series:
    """
    daily_returns: array-like lunghezza n, rendimenti giornalieri decimali di
    UN portafoglio in UN trading period (es. la colonna committed_return o
    employed_return di aggregate_portfolio_run).
    dates: date di calendario corrispondenti (stessa lunghezza n).

    Ritorna il rendimento composto per mese di calendario:
    prod_t(1+r_t) - 1 sui giorni del mese, indicizzato per Timestamp di
    inizio mese (coerente con data/factors.py).
    """
    s = pd.Series(np.asarray(daily_returns, dtype=float), index=pd.DatetimeIndex(dates))
    monthly = s.groupby(s.index.to_period("M")).apply(lambda x: float(np.prod(1 + x.to_numpy()) - 1))
    monthly.index = monthly.index.to_timestamp(how="start")
    monthly.index.name = "month"
    return monthly


def combine_overlapping_portfolios(monthly_by_run: dict[str, pd.Series]) -> pd.Series:
    """
    monthly_by_run: {run_id: serie di rendimenti mensili di UN portafoglio
    sfalsato (una per ciascuna formation date mensile), indicizzata per mese
    di calendario (Timestamp inizio mese), come da compound_to_monthly.

    Media alla Jegadeesh-Titman (PROTOCOL.md §2.2): il rendimento della
    strategia in un dato mese di calendario e' la MEDIA SEMPLICE dei
    rendimenti dei portafogli attivi quel mese (fino a N_OVERLAPPING=6 run,
    uno per ciascuna delle formation date dei mesi precedenti). Un mese in
    cui nessun portafoglio e' attivo (bordi della finestra) non compare
    affatto nella serie risultante: non e' un valore 0, e' semplicemente
    fuori dominio.
    """
    combined = pd.concat(monthly_by_run.values(), axis=1)
    return combined.mean(axis=1, skipna=True).dropna().sort_index()
