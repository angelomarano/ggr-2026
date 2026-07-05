"""
factors.py — Fattori Fama-French 3 + Momentum + Short-Term Reversal (Ken French
Data Library, via pandas-datareader) + risk-free rate (RF), frequenza mensile.

Uso:
    python -m data.factors     # scarica e cache la Parquet locale

Cache locale in Parquet (config.FACTORS_CACHE): nessun accesso di rete nei
test ne' nei run successivi al primo download.

Conversione: Ken French pubblica i fattori in percentuale (es. -0.11 = -0.11%
mensile); qui si converte a frazione decimale (-0.0011) per coerenza con le
serie di rendimento del resto della pipeline (src/trading.py, src/returns.py).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config

CACHE_PATH = Path(config.FACTORS_CACHE)

# Dataset Ken French da scaricare (nome esatto pandas_datareader -> colonne
# rilevanti). "F-F_Research_Data_Factors" porta anche RF (risk-free), non
# serve un dataset a parte.
_DATASETS = {
    "F-F_Research_Data_Factors": ["Mkt-RF", "SMB", "HML", "RF"],
    "F-F_Momentum_Factor": ["Mom"],
    "F-F_ST_Reversal_Factor": ["ST_Rev"],
}


def assemble_factors(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Unisce i DataFrame grezzi di pandas_datareader (uno per dataset Ken
    French elencato in _DATASETS, colonne come indicato, indice PeriodIndex
    mensile), converte da percentuale a frazione decimale e normalizza
    l'indice a Timestamp di inizio mese (coerente col resto della pipeline,
    che usa pd.Timestamp ovunque, mai PeriodIndex).

    Join = INNER sui mesi: Ken French a volte pubblica Momentum/ST-Reversal
    con un mese di ritardo rispetto ai 3 fattori base; un mese assente in uno
    qualunque dei tre dataset viene escluso interamente, cosi' non si
    introduce un NaN silenzioso a valle in una regressione fattoriale (§2.4.3).
    """
    merged = None
    for name, cols in _DATASETS.items():
        df = raw[name][cols].astype(float)
        merged = df if merged is None else merged.join(df, how="inner")
    merged = merged / 100.0
    merged.index = merged.index.to_timestamp(how="start")
    merged.index.name = "month"
    return merged.sort_index()


def download_and_cache_factors() -> pd.DataFrame:
    """Scarica i tre dataset dalla Ken French Data Library e salva la cache
    locale in Parquet. Da eseguire manualmente (python -m data.factors)."""
    import pandas_datareader.data as web

    # pandas_datareader defaults to roughly the last 5 years when start/end
    # are omitted; pass explicit bounds or the cache silently misses the
    # replication window.
    raw = {
        name: web.DataReader(name, "famafrench", start=config.DATA_START, end=config.DATA_END)[0]
        for name in _DATASETS
    }
    fac = assemble_factors(raw)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fac.to_parquet(CACHE_PATH)
    return fac


def load_factors(start: str = config.DATA_START, end: str = config.DATA_END) -> pd.DataFrame:
    """Legge la cache locale e restituisce i fattori nel range [start, end]
    (per mese di inizio, inclusivo). Solleva FileNotFoundError se la cache
    non esiste ancora: eseguire prima `python -m data.factors`."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"cache fattori mancante ({CACHE_PATH}); eseguire prima "
            "`python -m data.factors` per scaricarla."
        )
    fac = pd.read_parquet(CACHE_PATH)
    return fac.loc[pd.Timestamp(start) : pd.Timestamp(end)]


if __name__ == "__main__":
    f = download_and_cache_factors()
    print(f"Scaricati e cachati {len(f)} mesi di fattori -> {CACHE_PATH}")
    print(f.head())
    print(f.tail())
