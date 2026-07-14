# Transaction cost curve (PROTOCOL.md §5)

Primary portfolio only: top-20, wait-one-day, committed capital. Level 2 (explicit cost grid c in [0, 5, 10, 20, 40] bp per side, round-trip cost = 4c, PROTOCOL.md §5) computed here by applying src/costs.apply_round_trip_cost to the per-pair trade logs persisted by notebooks/06_regenerate_trade_logs.py -- no re-run of the trading engine. Level 0 (same-day gross) and Level 1 (wait-one-day gross) are reused verbatim from gate1_results.json/gate2_results.json, not recomputed; Level 1 is exactly this script's own c=0bp point (cross-checked below, must match to reused-data precision).

## Return and Sharpe vs cost, by window

### Gate 1 (2003-2009, golden set)

Level 0 (same-day gross, reused): 0.1276%/month, Sharpe 0.45.
Level 1 (wait-one-day gross, reused): 0.1484%/month, Sharpe 0.52.

| c (bp/side) | mean/month | t (NW) | ann. Sharpe | n months |
|---|---|---|---|---|
| 0 | 0.1484% | 1.50 | 0.52 | 77 |
| 5 | 0.1043% | 1.07 | 0.36 | 77 |
| 10 | 0.0602% | 0.63 | 0.21 | 77 |
| 20 | -0.0280% | -0.30 | -0.10 | 77 |
| 40 | -0.2043% | -2.20 | -0.73 | 77 |

**Break-even cost c\* = 16.8 bp/side.** linearly interpolated between grid points c=10bp (mean=0.000602) and c=20bp (mean=-0.000280), where the sign change occurs.

### Gate 2 (2010-2026, full universe)

Level 0 (same-day gross, reused): 0.0568%/month, Sharpe 0.33.
Level 1 (wait-one-day gross, reused): 0.0425%/month, Sharpe 0.26.

| c (bp/side) | mean/month | t (NW) | ann. Sharpe | n months |
|---|---|---|---|---|
| 0 | 0.0425% | 1.06 | 0.26 | 198 |
| 5 | 0.0065% | 0.16 | 0.04 | 198 |
| 10 | -0.0295% | -0.75 | -0.18 | 198 |
| 20 | -0.1015% | -2.61 | -0.62 | 198 |
| 40 | -0.2454% | -6.41 | -1.49 | 198 |

**Break-even cost c\* = 5.9 bp/side.** linearly interpolated between grid points c=5bp (mean=0.000065) and c=10bp (mean=-0.000295), where the sign change occurs.

### Gate 2 (2010-2026, golden set robustness)

Level 0 (same-day gross, reused): 0.0569%/month, Sharpe 0.29.
Level 1 (wait-one-day gross, reused): 0.0494%/month, Sharpe 0.26.

| c (bp/side) | mean/month | t (NW) | ann. Sharpe | n months |
|---|---|---|---|---|
| 0 | 0.0494% | 1.12 | 0.26 | 198 |
| 5 | 0.0127% | 0.29 | 0.07 | 198 |
| 10 | -0.0240% | -0.56 | -0.13 | 198 |
| 20 | -0.0973% | -2.30 | -0.52 | 198 |
| 40 | -0.2438% | -5.91 | -1.30 | 198 |

**Break-even cost c\* = 6.7 bp/side.** linearly interpolated between grid points c=5bp (mean=0.000127) and c=10bp (mean=-0.000240), where the sign change occurs.

## Short costs (PROTOCOL.md §5) -- not modeled quantitatively

PROTOCOL.md §5 declares short costs as a qualitative scenario only ("+25 bp/anno di borrow su general collateral"; hard-to-borrow names mentioned as a limit, not a number to compute) -- this is not simulated anywhere in this repository. The cost grid above covers only the round-trip trading costs on both legs (PROTOCOL.md §5, Level 2); it does not include any securities-lending/borrow cost for the short leg. A strategy shown profitable above a given c on this grid could still be unprofitable once realistic borrow costs are added, particularly for small/illiquid names that are more likely to be hard-to-borrow than the large/mid-cap S&P 500 constituents this project trades. This is stated here as an explicit limitation, not quantified.

