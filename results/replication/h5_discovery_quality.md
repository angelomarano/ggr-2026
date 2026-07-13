# H5 discovery-quality comparison

Sample: 7/8 runs from the replication window (2003-2009, golden set), evenly spaced: 2003-01, 2003-11, 2004-09, 2005-07, 2006-06, 2007-04, 2008-02, 2008-12.

Discovery quality is H5's PRIMARY comparison metric (PROTOCOL.md §4/H5); net performance is reported SECONDARY, per the protocol's own framing ("il messaggio non e' clustering=piu' soldi").

## Search-space reduction

| run | n universe | clustering method | n clusters | largest cluster | noise share | total possible pairs | intra-cluster pairs tested | reduction ratio |
|---|---|---|---|---|---|---|---|---|
| 2003-11 | 258 | optics | 18 | 8 (10.0%) | 69.0% | 33,153 | 159 | 0.48% |
| 2004-09 | 263 | optics | 15 | 11 (13.3%) | 68.4% | 34,453 | 239 | 0.69% |
| 2005-07 | 267 | optics | 17 | 12 (14.0%) | 67.8% | 35,511 | 225 | 0.63% |
| 2006-06 | 268 | optics | 20 | 7 (8.9%) | 70.5% | 35,778 | 130 | 0.36% |
| 2007-04 | 281 | optics | 21 | 8 (8.0%) | 64.4% | 39,340 | 209 | 0.53% |
| 2008-02 | 289 | optics | 14 | 13 (18.3%) | 75.4% | 41,616 | 207 | 0.50% |
| 2008-12 | 307 | optics | 20 | 9 (9.1%) | 67.8% | 46,971 | 227 | 0.48% |
| **aggregate** | | | | | | 266,822 | 1,396 | 0.52% |

## Discovery quality (primary), aggregated across the 8 runs

| list | n stationarity-evaluable | % OOS-stationary | n simulated | % converged >=1x | mean half-life OOS (days, mean of run means) |
|---|---|---|---|---|---|
| GGR-SSD (baseline) | 140 | 12.1% | 140 | 52.1% | 74.6 |
| Cluster+SSD (Variant A) | 140 | 12.9% | 140 | 33.6% | 13.9 |
| Cluster+Cointegration (Variant B) | 102 | 16.7% | 102 | 21.6% | 14.0 |
| Brute-force+BH (comparator) | 7 | 14.3% | 7 | 0.0% | 6.3 |

## Discovery quality, per run

| run | list | n candidates | % OOS-stationary | % converged >=1x | mean half-life OOS |
|---|---|---|---|---|---|
| 2003-11 | GGR-SSD (baseline) | 20 | 0.0% | 55.0% | 291.1 |
| 2003-11 | Cluster+SSD (Variant A) | 20 | 25.0% | 20.0% | 18.0 |
| 2003-11 | Cluster+Cointegration (Variant B) | 17 | 17.6% | 11.8% | 11.6 |
| 2003-11 | Brute-force+BH (comparator) | 6 | 0.0% | 0.0% | 9.7 |
| 2004-09 | GGR-SSD (baseline) | 20 | 10.0% | 50.0% | 155.3 |
| 2004-09 | Cluster+SSD (Variant A) | 20 | 20.0% | 20.0% | 13.0 |
| 2004-09 | Cluster+Cointegration (Variant B) | 14 | 7.1% | 7.1% | 13.3 |
| 2004-09 | Brute-force+BH (comparator) | 0 | n/a | n/a | n/a |
| 2005-07 | GGR-SSD (baseline) | 20 | 30.0% | 60.0% | 13.5 |
| 2005-07 | Cluster+SSD (Variant A) | 20 | 0.0% | 20.0% | 13.2 |
| 2005-07 | Cluster+Cointegration (Variant B) | 20 | 20.0% | 15.0% | 13.9 |
| 2005-07 | Brute-force+BH (comparator) | 0 | n/a | n/a | n/a |
| 2006-06 | GGR-SSD (baseline) | 20 | 10.0% | 30.0% | 13.4 |
| 2006-06 | Cluster+SSD (Variant A) | 20 | 5.0% | 20.0% | 11.4 |
| 2006-06 | Cluster+Cointegration (Variant B) | 6 | 0.0% | 33.3% | 14.8 |
| 2006-06 | Brute-force+BH (comparator) | 0 | n/a | n/a | n/a |
| 2007-04 | GGR-SSD (baseline) | 20 | 10.0% | 55.0% | 21.1 |
| 2007-04 | Cluster+SSD (Variant A) | 20 | 5.0% | 50.0% | 16.0 |
| 2007-04 | Cluster+Cointegration (Variant B) | 10 | 40.0% | 0.0% | 18.2 |
| 2007-04 | Brute-force+BH (comparator) | 1 | 100.0% | 0.0% | 3.0 |
| 2008-02 | GGR-SSD (baseline) | 20 | 15.0% | 50.0% | 13.2 |
| 2008-02 | Cluster+SSD (Variant A) | 20 | 25.0% | 55.0% | 9.3 |
| 2008-02 | Cluster+Cointegration (Variant B) | 18 | 5.6% | 27.8% | 14.4 |
| 2008-02 | Brute-force+BH (comparator) | 0 | n/a | n/a | n/a |
| 2008-12 | GGR-SSD (baseline) | 20 | 10.0% | 65.0% | 14.2 |
| 2008-12 | Cluster+SSD (Variant A) | 20 | 10.0% | 50.0% | 16.4 |
| 2008-12 | Cluster+Cointegration (Variant B) | 17 | 23.5% | 52.9% | 11.9 |
| 2008-12 | Brute-force+BH (comparator) | 0 | n/a | n/a | n/a |

## Multiple-testing accounting

List 1 (GGR-SSD) runs no hypothesis test at all -- SSD ranking, not reported here. Lists 2/3's n_tests come from intra_cluster_pairs (no BH/BY correction applied to them, per PROTOCOL.md §4/H5 step 5 vs step 6). List 4's n_tests/survivors come directly from brute_force_cointegration_screen.

| run | cluster_ssd+cluster_coint n_tests (intra-cluster pairs) | brute_force n_tests | expected false positives | n BH survivors | n BY survivors |
|---|---|---|---|---|---|
| 2003-11 | 159 | 33,153 | 1657.7 | 15 | 0 |
| 2004-09 | 239 | 34,453 | 1722.7 | 0 | 0 |
| 2005-07 | 225 | 35,511 | 1775.6 | 0 | 0 |
| 2006-06 | 130 | 35,778 | 1788.9 | 0 | 0 |
| 2007-04 | 209 | 39,340 | 1967.0 | 1 | 0 |
| 2008-02 | 207 | 41,616 | 2080.8 | 0 | 0 |
| 2008-12 | 227 | 46,971 | 2348.6 | 3 | 1 |

## Net performance (secondary), aggregated across the 8 runs

Committed capital, wait-one-day, nominal n_selected=20 for every list (see module docstring for why the nominal size is used even when a list finds fewer than 20 candidates).

| list | mean/month | t (NW) | ann. Sharpe | n months |
|---|---|---|---|---|
| GGR-SSD (baseline) | 0.1835% | 1.10 | 0.40 | 44 |
| Cluster+SSD (Variant A) | 0.3083% | 1.98 | 1.11 | 44 |
| Cluster+Cointegration (Variant B) | -0.0437% | -0.26 | -0.11 | 44 |
| Brute-force+BH (comparator) | 0.0000% | n/a | n/a | 44 |

## Anomalies

Failed runs: 1.
- 2003-01: ValueError: no reference trading day before 2002-01-02 00:00:00 to anchor the first return

Runs that raised warnings during processing: 0.
Total warning count across all runs: 0.
