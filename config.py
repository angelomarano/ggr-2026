"""
config.py — Parametri congelati del protocollo (PROTOCOL_ggr2026.md, v1.0 ratificata).
Ogni modifica a questo file dopo il Gate 1 va loggata in DEVIATIONS.md.
"""

# ---------------------------------------------------------------- finestre
DATA_START = "2002-01-01"
DATA_END = "2026-06-30"

# Finestra di replica (Gate 1): trading periods che INIZIANO in questi mesi
REPLICATION_TRADING_START_FIRST = "2003-01"
REPLICATION_TRADING_START_LAST = "2008-12"   # ultima chiusura: giugno 2009
# Clausola ratificata: se attrition 2003-2009 < 70%, slittare a 2005-01 (loggare)
ATTRITION_MIN_COMPLETE_SHARE = 0.70

# Finestra OOS congelata (Gate 2): eseguita UNA sola volta
OOS_TRADING_START_FIRST = "2010-01"
OOS_TRADING_START_LAST = "2025-12"           # ultima chiusura: giugno 2026

# ---------------------------------------------------------------- GGR core
FORMATION_DAYS = 252          # 12 mesi
TRADING_DAYS = 126            # 6 mesi
N_OVERLAPPING = 6             # portafogli sfalsati, media alla Jegadeesh-Titman
TOP_PAIRS = 20                # portafoglio primario
TOP_PAIRS_SMALL = 5
CONTROL_PAIRS_RANGE = (101, 120)  # inclusivo, 1-indexed sul ranking SSD
OPEN_TRIGGER_SIGMAS = 2.0     # |spread| > 2 * sigma_formation
EXECUTION_VARIANTS = ("same_day", "wait_one_day")   # sempre entrambe
NW_LAGS = 6                   # Newey-West

# Convenzione dichiarata (ambiguità GGR): ri-normalizzazione a 1 delle due
# gambe al primo giorno del trading period; sigma stimata SOLO sul formation.
RENORMALIZE_AT_TRADING_START = True   # task W2: cross-check vs Rubesam

# ---------------------------------------------------------------- inferenza
BLOCK_BOOTSTRAP_MEAN_BLOCK_MONTHS = 6
BLOCK_BOOTSTRAP_REPS = 10_000
RANDOM_PAIRS_BOOTSTRAP_REPS = 200     # falsificazione GGR (decile-matched)
SEED = 20260705                        # fissato: data di ratifica del protocollo

# ---------------------------------------------------------------- H2 regimi
VIX_HIGH_THRESHOLD = 25.0     # ratificato: soglia fissa primaria
EVENT_WINDOWS = {
    "covid_2020": ("2020-02-01", "2020-06-30"),
    "tightening_2022": ("2022-01-01", "2022-10-31"),
}

# ---------------------------------------------------------------- H5 cluster
PCA_N_COMPONENTS = 10         # ratificato; sensitivity dichiarata: {5, 15}
PCA_SENSITIVITY = (5, 15)
OPTICS_MIN_SAMPLES = 3        # ratificato (fallback: k-means via silhouette)
OPTICS_XI = 0.05
KMEANS_K_RANGE = (5, 30)
EG_PVALUE_MAX = 0.05          # Engle-Granger, p-value MacKinnon
HALF_LIFE_RANGE_DAYS = (5, 60)
OOS_ADF_PVALUE = 0.10         # soglia discovery-quality sul trading period
FDR_ALPHA = 0.05              # Benjamini-Hochberg; robustness: Benjamini-Yekutieli

# ---------------------------------------------------------------- costi
COST_GRID_BP_PER_SIDE = (0, 5, 10, 20, 40)   # round-trip coppia = 4 * c

# ---------------------------------------------------------------- dati
CONSTITUENTS_CSV = "data/raw/sp500_membership.csv"
PRICES_DIR = "data/raw/prices"        # parquet per ticker
FAILURES_LOG = "data/raw/download_failures.csv"
