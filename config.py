"""
config.py — Frozen protocol parameters (PROTOCOL_ggr2026.md, ratified v1.0).
Any change to this file after Gate 1 must be logged in DEVIATIONS.md.
"""

# ---------------------------------------------------------------- windows
DATA_START = "2002-01-01"
DATA_END = "2026-06-30"

# Replication window (Gate 1): trading periods that START in these months
REPLICATION_TRADING_START_FIRST = "2003-01"
REPLICATION_TRADING_START_LAST = "2008-12"   # last close: June 2009
# Ratified clause: if 2003-2009 attrition < 70%, shift to 2005-01 (log it)
ATTRITION_MIN_COMPLETE_SHARE = 0.70

# Frozen OOS window (Gate 2): executed ONCE
OOS_TRADING_START_FIRST = "2010-01"
OOS_TRADING_START_LAST = "2025-12"           # last close: June 2026

# ---------------------------------------------------------------- GGR core
FORMATION_DAYS = 252          # 12 months
TRADING_DAYS = 126            # 6 months
N_OVERLAPPING = 6             # staggered portfolios, Jegadeesh-Titman averaging
TOP_PAIRS = 20                # primary portfolio
TOP_PAIRS_SMALL = 5
CONTROL_PAIRS_RANGE = (101, 120)  # inclusive, 1-indexed on SSD ranking
OPEN_TRIGGER_SIGMAS = 2.0     # |spread| > 2 * sigma_formation
EXECUTION_VARIANTS = ("same_day", "wait_one_day")   # always both
NW_LAGS = 6                   # Newey-West

# Declared convention (GGR ambiguity): re-normalize both legs to 1 on the
# first day of the trading period; sigma estimated ONLY on the formation period.
RENORMALIZE_AT_TRADING_START = True   # task W2: cross-check vs Rubesam

# ---------------------------------------------------------------- inference
BLOCK_BOOTSTRAP_MEAN_BLOCK_MONTHS = 6
BLOCK_BOOTSTRAP_REPS = 10_000
RANDOM_PAIRS_BOOTSTRAP_REPS = 200     # GGR falsification (decile-matched)
SEED = 20260705                        # fixed: protocol ratification date

# ---------------------------------------------------------------- H2 regimes
VIX_HIGH_THRESHOLD = 25.0     # ratified: primary fixed threshold
EVENT_WINDOWS = {
    "covid_2020": ("2020-02-01", "2020-06-30"),
    "tightening_2022": ("2022-01-01", "2022-10-31"),
}

# ---------------------------------------------------------------- H5 clustering
PCA_N_COMPONENTS = 10         # ratified; declared sensitivity: {5, 15}
PCA_SENSITIVITY = (5, 15)
OPTICS_MIN_SAMPLES = 3        # ratified (fallback: k-means via silhouette)
OPTICS_XI = 0.05
KMEANS_K_RANGE = (5, 30)
EG_PVALUE_MAX = 0.05          # Engle-Granger, MacKinnon p-value
HALF_LIFE_RANGE_DAYS = (5, 60)
OOS_ADF_PVALUE = 0.10         # discovery-quality threshold on the trading period
FDR_ALPHA = 0.05              # Benjamini-Hochberg; robustness: Benjamini-Yekutieli

# ---------------------------------------------------------------- costs
COST_GRID_BP_PER_SIDE = (0, 5, 10, 20, 40)   # pair round-trip = 4 * c

# ---------------------------------------------------------------- data
CONSTITUENTS_CSV = "data/raw/sp500_membership.csv"
PRICES_DIR = "data/raw/prices"        # parquet per ticker
FAILURES_LOG = "data/raw/download_failures.csv"
FACTORS_CACHE = "data/raw/factors.parquet"   # FF3 + Mom + ST_Rev + RF, monthly

# ---------------------------------------------------------------- POST-HOC: data quality
# NOT an original protocol parameter (unlike OPEN_TRIGGER_SIGMAS,
# FORMATION_DAYS, etc., which have been frozen by PROTOCOL.md from the start).
# Added after discovering, during Gate 2, ~25 tickers with severely corrupted
# Yahoo data (ticker recycling onto stale OTC/penny-stock quotes after the
# original delisting — see DEVIATIONS.md). An absolute daily return beyond
# this threshold is almost certainly a data artifact, not a real price move,
# on a large/mid-cap S&P 500 universe.
MAX_ABS_DAILY_RETURN = 3.0    # 300%; beyond this, the ticker is excluded from the run's formation

# Second post-hoc filter, same discovery: some corrupted tickers don't jump
# but instead sit with a bit-identical Adj Close (frozen/stale quote) for
# long stretches in the formation period - zero return every day, so the
# filter above doesn't catch them, but a constant normalized price index
# artificially "matches" any other low-volatility stock, spuriously
# lowering the SSD. A liquid large/mid-cap stock practically never has
# 5 consecutive trading days with an identical Adj Close.
MAX_CONSECUTIVE_FROZEN_DAYS = 5   # beyond this, the ticker is excluded from the run's formation
