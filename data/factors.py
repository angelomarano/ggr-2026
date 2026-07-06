"""
factors.py — Fama-French 3 factors + Momentum + Short-Term Reversal (Ken French
Data Library, via pandas-datareader) + risk-free rate (RF), monthly frequency.

Usage:
    python -m data.factors     # downloads and caches the local Parquet

Local Parquet cache (config.FACTORS_CACHE): no network access in tests nor
in runs after the first download.

Conversion: Ken French publishes factors as percentages (e.g. -0.11 = -0.11%
monthly); here they are converted to a decimal fraction (-0.0011) for
consistency with the return series in the rest of the pipeline
(src/trading.py, src/returns.py).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config

CACHE_PATH = Path(config.FACTORS_CACHE)

# Ken French datasets to download (exact pandas_datareader name -> relevant
# columns). "F-F_Research_Data_Factors" also carries RF (risk-free), no
# separate dataset is needed.
_DATASETS = {
    "F-F_Research_Data_Factors": ["Mkt-RF", "SMB", "HML", "RF"],
    "F-F_Momentum_Factor": ["Mom"],
    "F-F_ST_Reversal_Factor": ["ST_Rev"],
}


def assemble_factors(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merges the raw pandas_datareader DataFrames (one per Ken French dataset
    listed in _DATASETS, columns as indicated, monthly PeriodIndex), converts
    from percentage to decimal fraction, and normalizes the index to a
    start-of-month Timestamp (consistent with the rest of the pipeline,
    which uses pd.Timestamp everywhere, never PeriodIndex).

    Join = INNER on months: Ken French sometimes publishes Momentum/ST-Reversal
    a month behind the 3 base factors; a month missing from any one of the
    three datasets is excluded entirely, so no silent NaN is introduced
    downstream in a factor regression (§2.4.3).
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
    """Downloads the three datasets from the Ken French Data Library and
    saves the local Parquet cache. Run manually (python -m data.factors)."""
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
    """Reads the local cache and returns the factors in the [start, end]
    range (by start month, inclusive). Raises FileNotFoundError if the cache
    does not exist yet: run `python -m data.factors` first."""
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"factors cache missing ({CACHE_PATH}); run "
            "`python -m data.factors` first to download it."
        )
    fac = pd.read_parquet(CACHE_PATH)
    return fac.loc[pd.Timestamp(start) : pd.Timestamp(end)]


if __name__ == "__main__":
    f = download_and_cache_factors()
    print(f"Downloaded and cached {len(f)} months of factors -> {CACHE_PATH}")
    print(f.head())
    print(f.tail())
