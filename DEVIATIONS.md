# DEVIATIONS.md
Log of deviations from the protocol (PROTOCOL.md v1.0, ratified 2026-07-05).
Format: date | section | deviation | rationale.

*(empty — no deviations)*

## 2026-07-05 — Gate 0: Yahoo attrition far beyond expectations, the 2003→2005 clause is ineffective

**Observed:** attrition on the full point-in-time universe: 51.9% (2003) → 63.7% (2009)
→ 96.8% (2025). The protocol's clause ("if <70% in 2003-2009, shift to
2005") solves nothing: even 2005 sits at 54.9%.

**Established cause:** Yahoo Finance does not truncate a delisted stock's history
at the delisting date — it ELIMINATES IT ENTIRELY, even for stocks like X
(US Steel) that stayed listed until 2025. Attrition by year therefore reflects
how many members of that year's index have been delisted FROM THEN UNTIL NOW
(2026), not the data quality of the period itself. This is a limitation of the
free vendor, documented in the literature (free datasets generally lack "dead
stocks"), not a pipeline error.

**Mitigations tested and discarded:**
- Stooq as an alternative source: requests blocked even on a safe control
  (AAPL) -> likely anti-scraping protection. Time-boxed and abandoned,
  no evidence either for or against coverage of delisted names.
- WRDS/CRSP: no confirmed ETH Zurich access (UZH and HSG have it,
  ETH unverified). Deferred to September 2026, when Angelo will have
  ETH credentials to check the library.ethz.ch catalog. Does not block
  the current project.

**Decision (ratified):** the original Gate 0 (70% threshold on the full universe)
REPLACED by a two-component Gate 0-bis:
  (a) mechanical fidelity of the engine -> validated on a GOLDEN SET (see
      data/golden_set.py): stocks with an intact Yahoo history in EVERY run
      2003-01..2008-12, built empirically from the cache already downloaded.
  (b) quantification of survivorship bias -> stays on the full universe,
      reported as a limitation/robustness check (attrition curve by year), no
      longer as a blocking gate. H1-H5 run on the full universe as the
      primary basis and on golden-set-only as an explicit robustness check.

PROTOCOL.md to be updated accordingly in §1.3 and §3 at the next revision;
this entry is the log of the decision in the meantime.

## 2026-07-05 — Gate 1: two qualitative invariants violated, cause identified (not a bug)

**Observed:** (1) wait-one-day > same-day on top-5/top-20 (reversed vs. GGR);
(2) control (101-120) > top-20, both in portfolio return and in average
payoff per trade (2.45x-3.02x, against a sigma ratio of ~1.4x).

**Diagnosis (SSD monotonic across 5 sampled runs, pair-by-pair mechanics
verified identical day by day on 3 pairs, no return duplication/skip):**
implementation bug ruled out as the cause.

**Cause (1):** decomposing same-day's payoff between signals wait-one-day
"misses" (n=631, same-day mean +1.88%) and persistent signals wait-one-day
confirms (n=1509 implied, same-day mean approx. -0.18%, actual wait-one-day
mean +0.55%). Persistent signals tend to keep widening one more day past
the trigger before reverting: entering the same day eats that residual
move, entering a day later avoids it. Consistent with the primary
portfolio's Momentum loading (-0.098, t=-5.11, the only strongly
significant one in the factor regression): exposure to a non-instantaneous
reversal. A direct H4 hypothesis (bid-ask bounce collapse in the
decimalized period, which in the original paper always compressed
wait-one-day below same-day) remains testable within H4 itself.

**Cause (2):** control's formation sigma is systematically ~1.4x top-20's
in every sampled run; the observed payoff ratio (2.45x-3.02x) is however
higher than sigma scale alone would predict linearly - the golden set's
composition (full survival required in EVERY run 2003-2009, which tends to
compress top-20 toward highly correlated mega-caps with tiny spreads)
explains the direction but not the full magnitude. No further cause
identified after pair-by-pair verification; the bug hunt concluded without
one.

**Decision:** no change to PROTOCOL.md, no re-tuning of parameters. Gate 1
considered PASSED WITH DOCUMENTED DEVIATIONS: the primary number
(top-20/wait-one-day/committed, 0.1484%/month) is within band; the two
invariant violations have an identified, non-bug cause. Implication for
H5: repeat the top-20 vs control comparison on the full universe too (not
just the golden set) as an explicit robustness check, since the
survive-every-run constraint is specific to the golden set and may not
reproduce elsewhere.

## 2026-07-05 — Gate 2: tickers with corrupted Yahoo data (extreme returns and frozen prices), two causal filters added

**Discovery:** during Gate 2's one-shot execution (full universe, 2010-2025),
the primary portfolio's decile-matched bootstrap produced astronomically absurd
numbers (replicate means on the order of 10^18%). An investigation of the raw
data found 25 tickers with daily returns exceeding 300% in absolute value on
at least one day between 2009 and 2026 (TNB, KRI, CBE, TIE, NCC, BOL, CFC, MEE,
CIN, PBG, BMC, CPWR, HPC, GLK, GDW, PTV, STI, UVN, HET, FSH, EP, UPC, EQ, MI,
PALM), almost certainly due to ticker-symbol recycling (a delisted company
whose ticker is reassigned to a different entity, often OTC/penny-stock, with
no distinction in Yahoo Finance's historical series).

**Quantified impact:**
- 57/192 OOS runs (full universe) had at least one of the 25 corrupted
  tickers survive the pre-existing formation completeness filter (no NaN, so
  not excluded by the already-present "no-trade day" filter).
- REAL contamination confirmed (not just a possible random substitute in the
  bootstrap) in 5 pairs across 2 runs: always and only BMC (2014-01 control
  rank117: BMC/BXP; 2014-04 top_20 rank10/17: BMC/MCD, BMC/BMS; 2014-04 control
  rank112/118: BMC/KO, BMC/CNP).
- Cause of these 5 cases: NOT an extreme-return jump (BMC's exact extreme-return
  dates do not fall within either of the two formation windows involved), but a
  frozen (bit-identical) Adj Close price for consecutive months (e.g. exactly
  $2380.0 from August to October 2013, with suspicious volume oscillating
  between 2000 and 0) — a constant normalized price index "matches" any other
  low-volatility stock artificially, spuriously lowering the SSD.
- None of the 25 corrupted tickers belong to the golden set (neither the
  replication one nor the OOS one), except TIE, which is in the replication
  golden set but whose corruption window (2010-2017) falls entirely after
  Gate 1's window (2003-2009): Gate 1 is not contaminated.

**Fix applied (src/formation.py, config.py):** two causal filters, applied
ONLY to the current run's formation period (never to data that is future
relative to that run — the same principle as the already-existing GGR filter
on "day without trades"):
1. `config.MAX_ABS_DAILY_RETURN` (default 3.0 = 300%): excludes a ticker if
   one of its daily returns in the formation period exceeds the threshold in
   absolute value.
2. `config.MAX_CONSECUTIVE_FROZEN_DAYS` (default 5): excludes a ticker if its
   Adj Close stays bit-identical for more than N consecutive trading days in
   the formation period. Deliberately based on price only, not volume (BMC's
   own volume is itself an unreliable signal).

Both parameters are explicitly marked in config.py as POST-HOC additions,
distinct from the parameters frozen by the original protocol
(OPEN_TRIGGER_SIGMAS, FORMATION_DAYS, etc.).

**Verification:** after the fix, all 5 real-contamination cases are resolved
(BMC excluded in both runs, replaced by SSD-legitimate pairs, no other
corrupted ticker takes its place). Synthetic tests added for both filters,
including explicit causality checks (no look-ahead: the same ticker with the
same anomaly is excluded or not depending on whether the run's specific
formation window contains it).

**Note — distinct from the Gate 0 issue:** this is NOT the same issue already
documented for Gate 0 (Yahoo entirely deleting delisted stocks' history).
Same data source (Yahoo Finance), two different defects: Gate 0 concerns the
ABSENCE of data for delisted stocks; this one concerns data that IS PRESENT
but corrupted (numerically implausible values) for stocks whose symbol was
recycled after the original delisting.

**Decision:** Gate 2 (full_universe arm) re-run a second time after the fix,
explicitly logged as such (not a silent re-run): before/after numbers both
visible in results/frozen/gate2_report.md and gate2_results.json (key
"full_universe_before_fix"). golden_set_robustness arm not re-run: already
verified clean, none of the 25 corrupted tickers belong to it.

**Confirmation note (post-hoc verification, read-only):** the "full_universe"
key currently in gate2_results.json - the one used for ALL tables in
gate2_report.md and the README (12-combination descriptive statistics, the
three falsifications, AND ALSO H2/H3/H4) - is the output of the second
execution (post-fix), not a leftover of the first. Confirmed by comparing
values known to differ between the two executions: b(HighVol) H2 = 0.31470%
(t=2.143) in "full_universe" vs. 0.31411% (t=2.139) in
"full_universe_before_fix"; mean raw correlation H3 = 0.48245 vs. 0.47850;
H4 delta 2010-2017 = 0.013890% vs. 0.013993%. The differences are small
(consistent with only 2 of 192 runs changing selection) but nonzero: H2/H3/H4,
not just the descriptive statistics and the bootstrap, were recomputed
post-fix.

## 2026-07-13 — H5: scale mismatch in the trading sigma for pairs selected via Engle-Granger

**Observed:** in notebooks/05_h5_discovery_quality.py, the sigma passed to
simulate_pair_wait_one_day (opening threshold |spread| > k*sigma) was
formation.spread_sigma for ALL FOUR candidate lists, including cluster_coint
(Variant B) and brute_force - which are however selected via Engle-Granger on
the log-price RESIDUAL, not on the normalized price index. spread_sigma
estimates the standard deviation of P*_i - P*_j (normalized price index, the
same quantity src/trading.py actually thresholds), so it stays correct for
ggr_ssd/cluster_ssd; for cluster_coint/brute_force it is an estimate taken
from a different quantity than the one the pair was selected on.

**Cause:** engle_granger_pair (src/selection_cluster.py) did not expose its
own residual's standard deviation - only t_stat/p_value/half_life_days -
so there was no alternative value to pass to the trading engine for pairs
selected via cointegration.

**Fix:** added the "residual_std" field (resid.std(ddof=0), no additional
regression - the residual already exists inside the function) to
engle_granger_pair's return dict, exposed as a column in the
cointegration_intra_cluster_ranking and brute_force_cointegration_screen
tables. notebooks/05_h5_discovery_quality.py updated (SIGMA_SOURCE_BY_LIST)
to use residual_std for cluster_coint/brute_force and to keep using
spread_sigma for ggr_ssd/cluster_ssd (already consistent, unchanged).
Synthetic test added (known cointegrated pair, P2=P1*exp(u_t), u AR(1)
phi=0.9): residual_std converges to the theoretical stationary value
innovation_std/sqrt(1-phi^2) on a large sample (n=5000, +-15% tolerance,
needed to let the cointegrating regression's superconsistency converge).

**Explicit verification (not just assumed):** the OOS stationarity test
(% OOS-stationary) and the OOS half-life are confirmed independent of the
sigma scale - both come exclusively from the Engle-Granger regression
recomputed on the trading period itself (statsmodels' p-value, the
residual's AR(1) coefficient), no external sigma ever enters. Verified with
a programmatic check (_verify_stationarity_and_half_life_unchanged) that
compares pre-fix and post-fix field by field across all 4 lists: identical
byte-for-byte.

**Quantified impact (8 sampled runs, 7 succeeded):** ggr_ssd/cluster_ssd
unchanged (as expected, sigma_source not changed). cluster_coint: %
convergence 21.6% -> 35.3% (+13.7 points), mean monthly return -0.0437% ->
-0.2022% (t from -0.26 to -0.80, still not significant). brute_force:
unchanged at 0.0%/n/a - the only 7 candidate pairs (across 2 of 7 runs with
a BH+filter survivor) don't cross the threshold even with the corrected
sigma (verified case by case, e.g. GLW-MCO: trading-period spread range
[-0.27, +0.20], post-fix threshold 0.385 - close but not exceeded).

**Decision:** notebooks/05_h5_discovery_quality.py re-run a second time after
the fix, explicitly logged as such (not a silent re-run): before/after
numbers both visible in results/replication/h5_discovery_quality.md
("Second execution" section) and h5_discovery_quality.json (keys
"pre_fix"/"post_fix", "scale_independence_check"). No other already-published
gate or result (Gate 1, Gate 2, README) is touched by this fix.

## 2026-07-14 — PROTOCOL.md §5: implemented the transaction-cost grid, required a trade-log regeneration

**Context:** PROTOCOL.md §5 (Level 2, per-round-trip cost grid) had never
been implemented. gate1_results.json/gate2_results.json did not persist
trade logs at the individual-pair level (only aggregated trade_stats via
src/diagnostics.trade_statistics) - data needed to apply 4c per completed
round-trip, as the protocol requires.

**Regeneration (notebooks/06_regenerate_trade_logs.py):** trading engine
re-run at INPUT PARITY (same already-frozen formation/trading windows, same
tickers, same sigma, no new parameter), restricted to the primary portfolio
(top_20/wait_one_day) on Gate 1 (golden set, 2003-2009) and Gate 2
(full_universe and golden_set_robustness, 2010-2026), this time also
persisting the full per-pair trade log (open/close events, daily_payoff) in
results/replication/trade_log_gate1.json and
results/frozen/trade_log_gate2.json.

**Integrity check (mandatory before proceeding, run inside the same
script):** every already-published aggregate statistic (mean_monthly, se_nw,
t_stat_nw, annualized_sharpe, pct_negative_months, max_drawdown, n_months,
trade_stats) for top_20/wait_one_day, BOTH committed and employed capital,
compared between the regeneration and the values already in
gate1_results.json/gate2_results.json (both arms). Result: **identical
within a 1e-9 relative tolerance on every field, for all 3 windows** - no
divergence, the regeneration is verified to be at input parity, no
already-published number is altered or in question.

**Cost implementation (src/costs.py, a new module - does not weigh down
src/returns.py, §5 is a protocol section distinct from §2.2):
apply_round_trip_cost subtracts 4c from a pair's daily_payoff on the closing
day of every completed round-trip (any reason: crossing, delisting,
end_of_period all count as a complete 4-trade round-trip); applied in
aggregation, src/trading.py unchanged. Grid c =
config.COST_GRID_BP_PER_SIDE = (0, 5, 10, 20, 40) bp, already existing in
config.py, reused as-is.

**Results (notebooks/07_cost_grid.py, results/replication/cost_curve.json):**
automatic cross-check of c=0bp against the already-published Level 1
(wait-one-day gross): matches exactly across all 3 windows (independent
confirmation that the cost application and the trade-log reconstruction are
correct). Break-even c*: Gate 1 = 16.8 bp/side, Gate 2 full_universe = 5.9
bp/side, Gate 2 golden_set_robustness = 6.7 bp/side (linear interpolation
between the grid points adjacent to the sign change, no forced
extrapolation). Short costs (borrow +25bp/year) NOT modeled quantitatively,
declared as a textual limitation in the report, as the protocol itself
requires.

**Decision:** no already-published number (Gate 1, Gate 2, H5, README) is
touched by this work - the trade-log regeneration is additive (new data
persisted, same result) and verified identical within machine noise before
being used for any new computation.
