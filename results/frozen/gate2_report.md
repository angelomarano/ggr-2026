# Gate 2 frozen run report

OOS window: trading start 2010-01 to 2025-12 (last trading period closes ~mid-2026).
Golden set (OOS) size: 568 tickers (results/frozen/golden_set_oos.csv).

No comparison against any acceptance band is made here; these are the computed numbers only.

## Descriptive statistics (both arms, adjacent rows per combination)

| portfolio | variant | capital | universe | mean/month | SE (NW) | t (NW) | ann. Sharpe | % neg months | max drawdown |
|---|---|---|---|---|---|---|---|---|---|
| top_5 | same_day | committed | full_universe | 0.0551% | 0.0394% | 1.40 | 0.30 | 41.4% | -4.3% |
| top_5 | same_day | committed | golden_set_robustness | 0.0969% | 0.0512% | 1.89 | 0.48 | 39.9% | -5.1% |
| top_5 | same_day | employed | full_universe | 0.1095% | 0.0722% | 1.52 | 0.34 | 41.9% | -10.0% |
| top_5 | same_day | employed | golden_set_robustness | 0.1561% | 0.0891% | 1.75 | 0.45 | 43.4% | -11.4% |
| top_5 | wait_one_day | committed | full_universe | 0.0454% | 0.0369% | 1.23 | 0.26 | 44.9% | -3.8% |
| top_5 | wait_one_day | committed | golden_set_robustness | 0.0840% | 0.0476% | 1.76 | 0.44 | 42.4% | -5.0% |
| top_5 | wait_one_day | employed | full_universe | 0.0841% | 0.0709% | 1.19 | 0.26 | 46.0% | -11.5% |
| top_5 | wait_one_day | employed | golden_set_robustness | 0.1640% | 0.0905% | 1.81 | 0.46 | 42.4% | -10.4% |
| top_20 | same_day | committed | full_universe | 0.0568% | 0.0427% | 1.33 | 0.33 | 43.9% | -5.9% |
| top_20 | same_day | committed | golden_set_robustness | 0.0569% | 0.0457% | 1.24 | 0.29 | 45.5% | -6.0% |
| top_20 | same_day | employed | full_universe | 0.0983% | 0.0872% | 1.13 | 0.28 | 44.9% | -12.5% |
| top_20 | same_day | employed | golden_set_robustness | 0.1016% | 0.0908% | 1.12 | 0.28 | 44.4% | -12.0% |
| top_20 | wait_one_day | committed | full_universe | 0.0425% | 0.0402% | 1.06 | 0.26 | 47.0% | -5.2% |
| top_20 | wait_one_day | committed | golden_set_robustness | 0.0494% | 0.0442% | 1.12 | 0.26 | 46.5% | -5.5% |
| top_20 | wait_one_day | employed | full_universe | 0.0706% | 0.0906% | 0.78 | 0.19 | 48.5% | -11.4% |
| top_20 | wait_one_day | employed | golden_set_robustness | 0.0797% | 0.0960% | 0.83 | 0.20 | 46.0% | -11.5% |
| control | same_day | committed | full_universe | 0.0804% | 0.0472% | 1.71 | 0.37 | 45.5% | -6.4% |
| control | same_day | committed | golden_set_robustness | 0.0591% | 0.0443% | 1.33 | 0.29 | 47.5% | -6.0% |
| control | same_day | employed | full_universe | 0.0941% | 0.0777% | 1.21 | 0.24 | 48.0% | -9.7% |
| control | same_day | employed | golden_set_robustness | 0.0699% | 0.0840% | 0.83 | 0.18 | 48.0% | -12.8% |
| control | wait_one_day | committed | full_universe | 0.0832% | 0.0454% | 1.83 | 0.40 | 47.0% | -5.6% |
| control | wait_one_day | committed | golden_set_robustness | 0.0520% | 0.0435% | 1.20 | 0.27 | 48.0% | -6.4% |
| control | wait_one_day | employed | full_universe | 0.1507% | 0.0817% | 1.85 | 0.37 | 47.0% | -9.1% |
| control | wait_one_day | employed | golden_set_robustness | 0.0352% | 0.0903% | 0.39 | 0.09 | 48.0% | -15.2% |

## Trade statistics (both arms)

| portfolio | variant | universe | avg round-trips/pair | avg holding days | % pairs never opened |
|---|---|---|---|---|---|
| top_5 | same_day | full_universe | 1.29 | 49.5 | 10.2% |
| top_5 | same_day | golden_set_robustness | 1.33 | 48.6 | 9.7% |
| top_5 | wait_one_day | full_universe | 1.13 | 51.3 | 13.2% |
| top_5 | wait_one_day | golden_set_robustness | 1.16 | 50.7 | 13.0% |
| top_20 | same_day | full_universe | 1.30 | 51.9 | 8.5% |
| top_20 | same_day | golden_set_robustness | 1.31 | 51.9 | 7.7% |
| top_20 | wait_one_day | full_universe | 1.15 | 53.7 | 11.4% |
| top_20 | wait_one_day | golden_set_robustness | 1.17 | 53.6 | 10.6% |
| control | same_day | full_universe | 1.33 | 52.6 | 6.7% |
| control | same_day | golden_set_robustness | 1.33 | 52.7 | 6.4% |
| control | wait_one_day | full_universe | 1.20 | 54.0 | 8.8% |
| control | wait_one_day | golden_set_robustness | 1.19 | 54.2 | 8.9% |

## Falsifications (primary portfolio only, both arms)

Note on the long/short leg decomposition below: long alpha minus short
alpha is not expected to equal the factor regression's net alpha above it,
and that is not a bug (verified empirically, see DEVIATIONS.md /
src/inference.py). Both legs are compounded monthly as independent
stand-alone sub-portfolios, while the net portfolio compounds the
already-netted daily series; compounding is non-linear, so the two
aggregation paths diverge by construction. The divergence is typically a
few bp/month.

### full_universe: top_20 / wait_one_day / committed

Factor regression: alpha = -0.0626%/month (t = -1.21), R² = 0.138, n = 197 months.

| factor | loading | t (NW) |
|---|---|---|
| Mkt-RF | -0.001 | -0.10 |
| SMB | 0.034 | 2.10 |
| HML | 0.017 | 0.79 |
| Mom | -0.007 | -0.58 |
| ST_Rev | 0.051 | 2.76 |

Long leg alpha = 0.1474%/month (t = 1.43); short leg alpha = 0.0959%/month (t = 0.82).

Decile-matched bootstrap (200 reps): real mean monthly return = 0.0425%; bootstrap replicate means: mean = 0.0238%, std = 0.0305%, [5th, 95th] pct = [-0.0229%, 0.0760%].

### golden_set_robustness: top_20 / wait_one_day / committed

Factor regression: alpha = -0.0508%/month (t = -0.87), R² = 0.152, n = 197 months.

| factor | loading | t (NW) |
|---|---|---|
| Mkt-RF | -0.003 | -0.29 |
| SMB | 0.032 | 1.80 |
| HML | 0.007 | 0.26 |
| Mom | -0.017 | -1.24 |
| ST_Rev | 0.066 | 2.81 |

Long leg alpha = 0.1253%/month (t = 1.25); short leg alpha = 0.0625%/month (t = 0.56).

Decile-matched bootstrap (200 reps): real mean monthly return = 0.0494%; bootstrap replicate means: mean = 0.0267%, std = 0.0331%, [5th, 95th] pct = [-0.0310%, 0.0794%].

## H2 - VIX regime and event windows (both arms)

### full_universe

High-vol threshold: VIX >= 25.0. 23/198 months classified high-vol.
Regression: primary_return = a + b*HighVol -> a = 0.0059%, b = 0.3147%, t(b) = 2.14 (n = 198).

- covid_2020 (5 months): cumulative return = 3.5819%, mean monthly bootstrap CI = [0.0566%, 1.3692%], approx. compounded CI = [0.2833%, 7.0363%].
- tightening_2022 (10 months): cumulative return = 3.3950%, mean monthly bootstrap CI = [0.1575%, 0.5263%], approx. compounded CI = [1.5859%, 5.3896%].

### golden_set_robustness

High-vol threshold: VIX >= 25.0. 23/198 months classified high-vol.
Regression: primary_return = a + b*HighVol -> a = 0.0088%, b = 0.3492%, t(b) = 1.83 (n = 198).

- covid_2020 (5 months): cumulative return = 4.8568%, mean monthly bootstrap CI = [0.0488%, 1.8816%], approx. compounded CI = [0.2441%, 9.7690%].
- tightening_2022 (10 months): cumulative return = 3.4366%, mean monthly bootstrap CI = [0.1756%, 0.5144%], approx. compounded CI = [1.7701%, 5.2645%].

## H3 - rolling 24-month correlation, top-20 vs control (both arms)

- full_universe: mean raw correlation = 0.482, mean residual (5-factor) correlation = 0.389 (full series saved in gate2_results.json).
- golden_set_robustness: mean raw correlation = 0.522, mean residual (5-factor) correlation = 0.454 (full series saved in gate2_results.json).

## H4 - same-day minus wait-one-day delta, top-20 committed (both arms)

### full_universe

- 2010_2017 (96 months): mean delta = 0.0139%, bootstrap SE = 0.0109%, CI = [-0.0064%, 0.0360%]
- 2018_2026 (102 months): mean delta = 0.0148%, bootstrap SE = 0.0100%, CI = [-0.0055%, 0.0337%]
- 2003_2009_gate1_reused (77 months): mean delta = -0.0208%, bootstrap SE = n/a -- reused from Gate 1's saved aggregate stats (difference of means); Gate 1 did not persist the paired monthly series, so no paired-bootstrap SE is available for this subperiod, unlike the two OOS subperiods below.

### golden_set_robustness

- 2010_2017 (96 months): mean delta = 0.0046%, bootstrap SE = 0.0110%, CI = [-0.0156%, 0.0271%]
- 2018_2026 (102 months): mean delta = 0.0103%, bootstrap SE = 0.0096%, CI = [-0.0085%, 0.0292%]
- 2003_2009_gate1_reused (77 months): mean delta = -0.0208%, bootstrap SE = n/a -- reused from Gate 1's saved aggregate stats (difference of means); Gate 1 did not persist the paired monthly series, so no paired-bootstrap SE is available for this subperiod, unlike the two OOS subperiods below.

## Anomalies

### full_universe
Universe size per run: min 496, max 506, mean 501.7. 192/192 runs completed.
Failed runs: 0.
Runs with warnings: 192, total warning count: 18763.

### golden_set_robustness
Universe size per run: min 311, max 479, mean 394.5. 192/192 runs completed.
Failed runs: 0.
Runs with warnings: 0, total warning count: 0.


---

## Second execution: full_universe re-run after data-quality fix

full_universe re-run after fixing two post-hoc data-quality filters in src/formation.py (config.MAX_ABS_DAILY_RETURN, config.MAX_CONSECUTIVE_FROZEN_DAYS); see DEVIATIONS.md. golden_set_robustness is unchanged from the first execution (none of the corrupted tickers belong to the golden set).

This is a SECOND, explicitly logged execution of the full_universe arm only - not a silent overwrite. The pre-fix numbers are preserved below for direct comparison.

### Descriptive statistics, full_universe: before vs after fix

| portfolio | variant | capital | before/after | mean/month | t (NW) | ann. Sharpe | max drawdown |
|---|---|---|---|---|---|---|---|
| top_5 | same_day | committed | before_fix | 0.0551% | 1.40 | 0.30 | -4.3% |
| top_5 | same_day | committed | after_fix | 0.0551% | 1.40 | 0.30 | -4.3% |
| top_5 | same_day | employed | before_fix | 0.1095% | 1.52 | 0.34 | -10.0% |
| top_5 | same_day | employed | after_fix | 0.1095% | 1.52 | 0.34 | -10.0% |
| top_5 | wait_one_day | committed | before_fix | 0.0454% | 1.23 | 0.26 | -3.8% |
| top_5 | wait_one_day | committed | after_fix | 0.0454% | 1.23 | 0.26 | -3.8% |
| top_5 | wait_one_day | employed | before_fix | 0.0841% | 1.19 | 0.26 | -11.5% |
| top_5 | wait_one_day | employed | after_fix | 0.0841% | 1.19 | 0.26 | -11.5% |
| top_20 | same_day | committed | before_fix | 0.0574% | 1.34 | 0.34 | -5.9% |
| top_20 | same_day | committed | after_fix | 0.0568% | 1.33 | 0.33 | -5.9% |
| top_20 | same_day | employed | before_fix | 0.0975% | 1.12 | 0.28 | -12.5% |
| top_20 | same_day | employed | after_fix | 0.0983% | 1.13 | 0.28 | -12.5% |
| top_20 | wait_one_day | committed | before_fix | 0.0430% | 1.07 | 0.26 | -5.2% |
| top_20 | wait_one_day | committed | after_fix | 0.0425% | 1.06 | 0.26 | -5.2% |
| top_20 | wait_one_day | employed | before_fix | 0.0692% | 0.76 | 0.19 | -11.4% |
| top_20 | wait_one_day | employed | after_fix | 0.0706% | 0.78 | 0.19 | -11.4% |
| control | same_day | committed | before_fix | 0.0769% | 1.61 | 0.36 | -7.1% |
| control | same_day | committed | after_fix | 0.0804% | 1.71 | 0.37 | -6.4% |
| control | same_day | employed | before_fix | 0.0883% | 1.13 | 0.23 | -10.7% |
| control | same_day | employed | after_fix | 0.0941% | 1.21 | 0.24 | -9.7% |
| control | wait_one_day | committed | before_fix | 0.0798% | 1.74 | 0.38 | -6.3% |
| control | wait_one_day | committed | after_fix | 0.0832% | 1.83 | 0.40 | -5.6% |
| control | wait_one_day | employed | before_fix | 0.1437% | 1.74 | 0.35 | -10.4% |
| control | wait_one_day | employed | after_fix | 0.1507% | 1.85 | 0.37 | -9.1% |

### Decile-matched bootstrap (primary portfolio), before vs after fix

Before the fix, a severely corrupted ticker (CBE, wild price swings between $0.005 and $170 with zero-volume stale quotes) was drawn as a random decile-matched substitute in the bootstrap, producing astronomical (numerically meaningless) replicate statistics.

- before_fix: real mean = 0.0430%; bootstrap replicate means: mean = -184211282020495982592.0000%, std = 1562009252283990147072.0000%, [5th, 95th] pct = [-329914010441485568.0000%, 43901590186494.2188%], range = [-15714708785791611437056.0000%, 11146843130098253824.0000%].
- after_fix: real mean = 0.0425%; bootstrap replicate means: mean = 0.0238%, std = 0.0305%, [5th, 95th] pct = [-0.0229%, 0.0760%], range = [-0.0731%, 0.1140%].

### Confirmed real-selection contamination cases and their resolution

BMC (long frozen/stale-price runs, not extreme jumps) was selected as a real pair member - not just a possible bootstrap substitute - in 2 runs before the fix:
- 2014-01, control rank 117: BMC/BXP.
- 2014-04, top_20 ranks 10 and 17: BMC/MCD, BMC/BMS.
- 2014-04, control ranks 112 and 118: BMC/KO, BMC/CNP.

After the fix, BMC is excluded from formation in both runs (frozen-price filter, not the extreme-return filter), and every rank shifts up to the next legitimately-ranked pair - no other corrupted ticker enters in its place.

