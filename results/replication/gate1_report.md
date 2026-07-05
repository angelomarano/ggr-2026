# Gate 1 replication report

Replication window: trading start 2003-01 to 2008-12 (last trading period closes ~June 2009).
Universe: golden set only (315 tickers), 71/72 runs completed.
Universe size per run: min 250, max 307, mean 272.4.

No comparison against PROTOCOL.md §3's acceptance band is made here; these are the computed numbers only.

## Descriptive statistics

| portfolio | variant | capital | mean/month | SE (NW) | t (NW) | ann. Sharpe | % neg months | max drawdown |
|---|---|---|---|---|---|---|---|---|
| top_5 | same_day | committed | 0.0302% | 0.0914% | 0.33 | 0.09 | 45.5% | -9.9% |
| top_5 | same_day | employed | 0.0578% | 0.1609% | 0.36 | 0.13 | 49.4% | -16.5% |
| top_5 | wait_one_day | committed | 0.0809% | 0.0936% | 0.86 | 0.23 | 44.2% | -9.2% |
| top_5 | wait_one_day | employed | 0.1298% | 0.1648% | 0.79 | 0.25 | 48.1% | -14.5% |
| top_20 | same_day | committed | 0.1276% | 0.0966% | 1.32 | 0.45 | 45.5% | -8.1% |
| top_20 | same_day | employed | 0.1306% | 0.1400% | 0.93 | 0.31 | 49.4% | -12.5% |
| top_20 | wait_one_day | committed | 0.1484% | 0.0987% | 1.50 | 0.52 | 42.9% | -7.7% |
| top_20 | wait_one_day | employed | 0.1814% | 0.1821% | 1.00 | 0.39 | 45.5% | -15.3% |
| control | same_day | committed | 0.3088% | 0.1170% | 2.64 | 1.03 | 36.4% | -4.7% |
| control | same_day | employed | 0.3052% | 0.1920% | 1.59 | 0.60 | 40.3% | -10.3% |
| control | wait_one_day | committed | 0.2942% | 0.1131% | 2.60 | 0.98 | 33.8% | -4.9% |
| control | wait_one_day | employed | 0.3453% | 0.1790% | 1.93 | 0.68 | 37.7% | -8.2% |

## Trade statistics

| portfolio | variant | avg round-trips/pair | avg holding days | % pairs never opened |
|---|---|---|---|---|
| top_5 | same_day | 1.44 | 54.6 | 3.1% |
| top_5 | wait_one_day | 1.34 | 55.1 | 4.5% |
| top_20 | same_day | 1.51 | 51.7 | 2.5% |
| top_20 | wait_one_day | 1.38 | 52.8 | 4.4% |
| control | same_day | 1.43 | 50.6 | 4.6% |
| control | wait_one_day | 1.28 | 52.4 | 6.7% |

## Falsifications (primary portfolio only: top_20 / wait_one_day / committed)

### Factor regression (5-factor excess return alpha)

alpha = -0.0882%/month (t = -0.89), R² = 0.396, n = 77 months.

| factor | loading | t (NW) |
|---|---|---|
| Mkt-RF | -0.050 | -1.73 |
| SMB | 0.058 | 1.49 |
| HML | 0.067 | 1.73 |
| Mom | -0.098 | -5.11 |
| ST_Rev | 0.069 | 1.48 |

### Long/short leg decomposition

Long leg alpha = 0.0397%/month (t = 0.32)
Short leg alpha = -0.0851%/month (t = -0.72)

### Decile-matched random-pairs bootstrap

200 replications. Real primary portfolio mean monthly return: 0.1484%.
Bootstrap replicate means: mean = 0.0968%, std = 0.0743%, [5th, 95th] percentile = [-0.0246%, 0.2215%], range = [-0.0835%, 0.3173%].

## Anomalies

Failed runs: 1.
- 2003-01: ValueError: no reference trading day before 2002-01-02 00:00:00 to anchor the first return

Runs that raised warnings during processing: 0.
Total warning count across all runs: 0.
