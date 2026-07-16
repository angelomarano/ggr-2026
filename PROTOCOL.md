# PROTOCOL — "Pairs Trading, Twenty Years Later"
### Frozen replication of Gatev–Goetzmann–Rouwenhorst (RFS 2006), out-of-sample extension to 2010–2026, and a clustering-based selection arm

**Version:** 1.0 — to be frozen before Gate 2.
**Golden rule:** every deviation from this document after Gate 1 must be logged in `DEVIATIONS.md` with date and rationale. No exceptions. The project's value in an interview depends on being able to say "the protocol was written before seeing the results."

---

## 0. Objective and principles

**Umbrella question:** is GGR-style pairs trading dead, dormant, or alive only in turbulence? And does clustering-based pair selection improve *discovery quality* relative to brute-force search?

**Deliverable:** a public GitHub repository (code + README with question/method/results/figures/limits) + one CV line defensible in every detail.

**Non-negotiable principles:**
1. **Frozen protocol.** Every parameter is fixed in this document (§2, §4). No parameter optimization, ever. The GGR convention (12m/6m, 2σ) is inherited, not tuned: the absence of data snooping IS the result.
2. **Validation gates.** No moving to the next phase without passing the gate (§3). The 2010–2026 out-of-sample run is executed **once only**.
3. **Formation/trading separation.** No quantity computed on trading-period data may influence selection or thresholds. Automatic anti-leakage test mandatory (§6, tests).
4. **Honesty about limitations.** Survivorship, costs, and statistical power are quantified and written into the README, not hidden.

---

## 1. Data

### 1.1 Sources
| Data | Primary source | Backup/validation | Notes |
|---|---|---|---|
| Point-in-time S&P 500 constituents | GitHub `fja05680/sp500` (from 1996) | GitHub `hanshof/sp500_constituents` | monthly snapshot: universe as of the formation start date |
| Daily prices (adjusted + raw close, volume) | Yahoo Finance via `yfinance` | Stooq (`pandas-datareader`) | adjusted close = total-return proxy |
| Risk-free (3M T-bill, DTB3) | FRED | — | for excess return and Sharpe |
| FF3 + Momentum + ST Reversal factors | Ken French Data Library | — | for the risk regressions |
| Index and volatility (^GSPC, ^VIX) | Yahoo | — | for H2 (regimes) |

**Total data window:** 2002-01-01 → 2026-06-30.
**Replication window (Gate 1):** trading periods starting 2003-01 through 2008-12 (last close June 2009) — overlapping Do & Faff's third sub-period (2003–June 2009, mean 0.24%/month on CRSP).
**Frozen OOS window:** trading periods starting 2010-01 through 2025-12 (last close June 2026).

### 1.2 Universe construction (for every formation date, monthly)
1. Take the list of S&P 500 constituents as of the formation period's start date (point-in-time: never the current list).
2. Download prices for the entire formation+trading period.
3. **GGR filter:** drop stocks with ≥1 no-trade day (proxy: zero volume or missing price) in the formation period. No minimum-price filter (pure GGR).
4. **Attrition table (mandatory, Gate 0):** for every year, report (a) number of constituents, (b) % with a complete Yahoo/Stooq price history, (c) % lost to delisting/ticker changes. This table goes in the README.

### 1.3 Survivorship: declared position
Yahoo does not retain the history of delisted stocks → a fraction of the historical universe is unrecoverable with free data. **Direction of the bias:** in pairs trading, delistings are often the outcome of divergences that never converged (losing positions forcibly closed); excluding them **inflates** returns. So our OOS numbers are an **upper bound**: if even the upper bound is ≈0, the "decline completed" conclusion holds *a fortiori*. If instead we find profits, the caveat must be declared as the main limitation. (GGR handle delistings with the delisting return; we cannot: say so explicitly.)

### 1.4 Cleaning
- Total return index: adjusted close; sanity check of raw vs. adjusted on 10 sample tickers (known splits: AAPL 2020, TSLA 2020, NVDA 2021/2024).
- Isolated missing prices in the trading period: no forward-fill for signals; if a stock stops trading for good, the position closes at the last available price (adapted GGR rule).
- Local Parquet cache; every download logged with a timestamp (reproducibility).

---

## 2. Core methodology — GGR replication (frozen)

### 2.1 Parameters
| Parameter | Value | Source |
|---|---|---|
| Formation period | 12 months (252 trading days) | GGR |
| Trading period | 6 months (126 days) | GGR |
| Staggering | new portfolio every month → 6 overlapping portfolios, Jegadeesh–Titman averaging | GGR |
| Matching | minimum SSD between normalized total-return price indices | GGR |
| Portfolios | top-5, **top-20 (primary)**, pairs 101–120 (control) | GGR |
| Opening trigger | \|spread\| > 2σ, σ = std of the spread in formation | GGR |
| Closing | crossing of normalized prices; end of period; delisting | GGR |
| Reopening | allowed after every convergence | GGR |
| Ticker uniqueness | a stock may appear in more than one pair (GGR do not impose uniqueness) | GGR |
| Execution | **always both variants:** same-day and wait-one-day | GGR |
| Inference | Newey–West 6 lags on the monthly series | GGR |

### 2.2 Formulas (to be implemented exactly as follows)
- Normalized index in formation: `P*_it = ∏_{τ=1..t} (1 + r_iτ)`, with `P*_i0 = 1`, r = daily total-return.
- Distance: `SSD_ij = Σ_t (P*_it − P*_jt)²` over 252 days.
- **Trading-period convention (a known ambiguity in the paper — GGR do not specify whether the series are re-normalized at the start of the trading period):** choice declared a priori: both legs **re-normalize to 1 on the first day of the trading period**; spread `s_t = P*_1t − P*_2t`; σ remains the one estimated on formation. **Task W2:** compare this convention against Rubesam's notebook; if it diverges, document it in `DEVIATIONS.md` and report Gate 1's sensitivity to both conventions.
- Opening at `|s_t| > 2σ`: short the high leg, long the low leg, $1 per leg.
- Daily payoff of the pair: `w_L,t·r_L,t − w_S,t·r_S,t`, with weights evolving `w_{t} = w_{t−1}(1+r_{t−1})` from 1 at the opening (mark-to-market, buy-and-hold within the trade). Aggregation across pairs and daily→monthly compounding follow **exactly eqs. (2)–(3) of the paper** (value-weighted average with evolving weights, then compounding to monthly): implement them this way, not with daily equal-weight simplifications. GGR's note: cash from intermediate payoffs earns zero interest (a conservative assumption) — replicate it.
- **Return on committed capital** = sum of payoffs / n. pairs in the portfolio (20). **Return on employed capital** = / n. pairs actually open. Both are reported; committed is primary for the tests.
- Monthly strategy return = average of the 6 staggered portfolios active that month.

### 2.3 Statistics reported (for every portfolio, for every variant)
Monthly mean, Newey–West(6) t, SE, annualized Sharpe, min/max, skewness, kurtosis, % negative months, max drawdown, mean round-trips per pair, mean holding duration, % pairs never opened.

### 2.4 Falsification tests (to be replicated, not optional)
1. **Random-pairs bootstrap** (200 replications): replace every real pair with two random stocks from the same prior-month return decile, same event dates → expected ≈0/negative. If our bootstrap yields positive returns, either the pipeline has a bug or the effect is masked reversal.
2. **Long vs. short decomposition:** 5-factor alpha per leg. Expected (GGR Table 7, top-20): **short**-leg alpha **negative and significant** (−0.52%/month, t=−3.35 → a positive contribution to the strategy that shorts it), **long**-leg alpha **≈0 and not significant** (+0.24%, t=1.27). Profitability comes from the short leg: if the long leg dominates in our replication, there is a sign or construction error.
3. **Factor regression:** monthly excess returns on FF3 + MOM + ST-Reversal; report alpha, t(NW), loadings.

---

## 3. Validation gates

### Gate 0 — Data integrity (end of W1)
- [ ] Point-in-time universe reconstructed for every formation date 2002–2025.
- [ ] Attrition table compiled; if the % of complete history in 2003–2009 is < 70%, shift the start of the replication window to 2005 and log it in `DEVIATIONS.md`.
- [ ] Price spot-check: 10 tickers × 20 dates vs. Stooq, discrepancies < 0.5%.
- [ ] Unit tests passed: normalization, SSD on synthetic series, synthetic pair with hand-computed trades (§6).
- [ ] Anti-leakage test passed: truncating the data at the end of formation, pair selection is bit-for-bit identical.

### Gate 1 — Replication fidelity (2003–2009, end of W3)
**Quantitative band (declared now, with one condition to be closed in W1):** top-20, wait-one-day, committed capital → monthly mean in **[0.05%, 0.45%]**. The reference is Do & Faff, 0.24%/month on CRSP full-universe 2003–June 2009 — **but the exact measure that number refers to (committed vs. employed) must be verified against their paper before Gate 1** (they compute both, like GGR). **Task W1:** retrieve D&F's table, fix the measure and the execution rule for the target, and only then freeze the band on that same measure. Our large-cap-only, survivorship-biased universe may sit above or below: the band reflects this uncertainty.

**Numerical anchors from the paper (CRSP Jul1963–Dec2002, documentation only — not directly replicable with our data):** top-20 same-day: 1.44%/month fully-invested (t=11.56) / 0.81% committed; top-20 wait-one-day: 0.90% fully-invested (t=9.29) / 0.52% committed; average top-20 trigger ≈ 5.3%; ~19.3 of 20 pairs open; ~1.96 round-trips per pair; average duration ~3.8 months.

**Qualitative invariants (all mandatory):**
- wait-one-day < same-day (positive gap on both capital measures);
- dominant short leg: short alpha negative and significant, long alpha ≈ 0 (see §2.4.2);
- positive returns in 2008 (turbulence) — consistent with Do & Faff;
- random-pairs bootstrap ≈ 0 or negative;
- pairs 101–120 < top-20;
- trading statistics in the GGR range: ~2 round-trips per pair, duration ~3–4 months, almost every pair opens;
- composition diagnostic: expected concentration of the top pairs in low-volatility sectors (in GGR, 71% of the top-20 stocks were utilities; on an S&P 500 universe the composition will differ, but top pairs dominated by high-volatility stocks = red flag).
**If out of band:** hunt for a bug (returns computation, overlap averaging, normalization convention), **never** retouch the parameters. Qualitative comparison against the Rubesam (CRSP) replication as an external reference.

### Gate 2 — Frozen OOS run (W4)
Executed **once** on 2010–2026 with the code validated at Gate 1, unchanged. Output written to `results/frozen/` and never regenerated (commit hash in the README).

---

## 4. Hypotheses, metrics, predictions, interpretations

### H1 — Has the decline completed? (primary)
- **Primary metric:** mean monthly excess return, top-20, wait-one-day, committed capital, 2010–2026.
- **Test:** H0: μ ≤ 0, Newey–West(6) t; 95% CI with stationary block bootstrap (mean block 6 months, 10,000 replications).
- **Prediction (from Do & Faff: 0.86 → 0.37 → 0.24%/month; exact measure to be fixed in W1, see Gate 1):** ≤ 20 bp/month, not significant; ≈ 0 after costs (§5).
- **Interpretations:** μ ≈ 0 → "the decline completed," confirmed with modern inference over 16 years never tested in the GGR literature (expected outcome, fully publishable on GitHub). μ > 0 significant → a surprising result: before believing it, run the §6 anti-leakage checklist and verify it isn't driven by 2020/2022 (→ H2).

### H2 — Did the turbulence exception hold? (the most interesting)
- **Regime definition (a priori):** a month is "high-vol" if the monthly average VIX ≥ 25 (fixed, interpretable threshold). Robustness: VIX terciles over the trailing 10-year distribution.
- **Test:** regression `r_t = a + b·HighVol_t + ε`, NW(6) errors; expected b > 0, significant. Plus declared event studies: **Feb–Jun 2020** and **Jan–Oct 2022** (cumulative returns with bootstrap CI).
- **Rationale from the paper:** GGR find strong performance in the weak markets of the 1970s and flat performance in the 1990s bull run; Do & Faff confirm it in 2000–02 and 2007–09. COVID is the sharpest turbulence test since 2008, and the published GGR literature stops around 2014.
- **Interpretations:** b>0 with profits concentrated in 2020/2022 → "dormant strategy that wakes up under stress" (the most interview-worthy finding). b≈0 → the turbulence exception is dead too: an equally clean finding.

### H3 — Is the latent factor dormant or reawakened?
- **Metric:** rolling 24-month correlation between top-20 and 101–120 returns (disjoint portfolios). GGR: 0.48 full-sample; 0.51 pre-1989 → 0.18 post-1988; on 5-factor residuals: 0.42 → 0.20 (so the common factor is not explained by the standard factors — compute both versions, raw and residual).
- **Prediction:** mean ~0.2 with spikes in high-vol months.
- **Output:** figure "GGR Fig. 4, extended to 2026" — central to the README.

### H4 — Has the bid-ask bounce evaporated (a guaranteed positive result)
- **Metric:** Δ = mean(same-day) − mean(wait-one-day), by sub-period (2003–09 vs. 2010–17 vs. 2018–26), computed on **both** capital measures. GGR baseline for the top-20: **~54 bp/month on fully-invested** (1.44% → 0.90%) and **~28 bp/month on committed** (0.81% → 0.52%); the paper reports declines of 30–55 bp (fully-invested) and 20–35 bp (committed) across portfolios. From the fully-invested version, GGR imply an effective spread of ~81 bp per stock (162 bp round-trip per pair) in a pre-decimalization world.
- **Test:** difference with paired bootstrap SE (same event dates).
- **Prediction:** Δ < 10 bp/month in 2010–2026 (large-cap spreads today = a few bp).
- **Why it matters:** Δ is a *direct* estimate of the microstructure cost relevant to the strategy. Even if H1 gives zero, H4 gives a clean, quantitative finding. Gold-standard interview material (microstructure, effective spread).

### H5 — Clustering-based selection: discovery quality (the extension)
**Alternative pipeline (everything else identical to the core):**
1. Formation daily-return matrix (252 × N), per-stock standardization.
2. PCA: **10 components** (a priori; Sarmento & Horta recommend < 15). Declared sensitivity: {5, 15}.
3. Clustering, **primary: OPTICS** (min_samples = 3, ξ = 0.05) on the loadings; "noise" points are excluded. Sensitivity: k-means with k chosen via silhouette in [5, 30].
4. Candidate pairs = intra-cluster only.
5. **Variant A (clean comparison with GGR):** intra-cluster SSD ranking, top-20.
   **Variant B (cointegration):** Engle–Granger on log-prices (ADF on the residuals, MacKinnon p-value), threshold p < 0.05, AR(1) half-life filter on the spread in [5, 60] days, top-20 by p-value.
6. **Brute-force comparator:** Engle–Granger on all ~125,000 pairs with Benjamini–Hochberg correction at 5%; survivor count; top-20 by p-value. **Declared robustness:** the tests on the pairs are strongly dependent (shared stocks, common factors) and BH assumes positive dependence (PRDS) — also report Benjamini–Yekutieli, valid under arbitrary dependence, and comment on the difference. A gold-standard point for a statistician in an interview.

**The three comparison metrics (declared now):**
1. **Discovery quality (H5's primary metric):** % of selected pairs whose spread is stationary in the *trading period* (ADF p < 0.10, a priori threshold) + % that converge at least once + distribution of OOS half-lives. Comparison: cluster-A vs. cluster-B vs. brute-force-BH vs. GGR-SSD.
2. **Multiple-testing accounting:** n. tests run per arm (≈125k vs. thousands); expected false positives under the global null at α=0.05 (≈6,200 in the brute-force arm); pairs surviving BH.
3. **Net performance:** the same H1 statistics on the four lists.

**Honest prediction:** performance differences within the noise; measurable discovery-quality differences in favor of the restricted selection. The message is not "clustering = more money" but "clustering = fewer false discoveries at the same money" — exactly the right message for a QR interview.

---

## 5. Transaction costs (a curve, not a single number)
- Level 0: gross same-day results (theoretical upper bound).
- Level 1: wait-one-day (embeds the implicit microstructure cost — this is the primary variant).
- Level 2: **explicit grid** c ∈ {0, 5, 10, 20, 40} bp per side per trade; a pair round-trip = 4 trades → cost 4c per round-trip. Output: return/Sharpe curve vs. c and **break-even cost c\***.
- Short costs: qualitative scenario of +25 bp/year borrow on general collateral; mention hard-to-borrow names as a limitation.
- Reference: Do & Faff use time-varying costs; we prefer the curve because it makes the result robust to any opinion about the "right" level.

---

## 6. Repository structure and implementation

```
ggr-2026/
├── README.md              # question → method → results → figures → limits
├── PROTOCOL.md            # this document (frozen pre-Gate 2)
├── DEVIATIONS.md          # dated deviation log
├── config.py              # ALL parameters from §2 and §4, one single place
├── data/
│   ├── universe.py        # point-in-time constituents + attrition table
│   ├── prices.py          # yfinance/Stooq, parquet cache, validations
│   └── factors.py         # French, FRED, VIX
├── src/
│   ├── formation.py       # normalization, SSD, matching
│   ├── selection_cluster.py  # PCA, OPTICS/k-means, intra-cluster EG, BH
│   ├── trading.py         # trigger, crossing, delisting, wait-one-day
│   ├── returns.py         # compounding weights, committed/employed, overlap averaging
│   ├── inference.py       # NW, block bootstrap, factor regressions
│   └── regimes.py         # VIX/vol regimes, event windows
├── tests/
│   ├── test_synthetic_pair.py   # hand-built pair, known trades and P&L
│   ├── test_no_lookahead.py     # truncation at formation ⇒ identical selection
│   └── test_returns_math.py     # compounding weights, overlap averaging
├── notebooks/             # 01_data_audit, 02_replication, 03_frozen_oos,
│                          # 04_clustering_arm, 05_figures
└── results/
    ├── replication/       # Gate 1
    └── frozen/            # Gate 2 — written once, commit hash in the README
```

**Libraries:** pandas, numpy, statsmodels (`adfuller`, `coint`, OLS with HAC covariance), scikit-learn (PCA, OPTICS, KMeans), scipy, yfinance, pandas-datareader, matplotlib. **No external backtesting framework:** the GGR loop (crossing close, compounding weights, overlap) is specific, and writing it — and testing it — is the whole point of the project.

**Synthetic test (specification):** two series with imposed cointegration (P2 = P1·e^{u_t}, u ~ stationary AR(1)) + 2 divergences injected at known dates → assert opening/closing dates and P&L computed by hand on paper. If this test doesn't pass, no real data.

---

## 7. Timeline (part-time, 5 weeks + buffer)
| Week | Goal | Definition of done |
|---|---|---|
| 1 | Data | Gate 0 passed; attrition table written; Do & Faff table retrieved and Gate 1's target measure frozen |
| 2 | Engine | formation+trading+returns; synthetic test green; first rough replication runs |
| 3 | Replication | Gate 1 passed; falsifications (bootstrap, long/short, factors) replicated |
| 4 | OOS | Gate 2 executed; H1–H4 with tables and figures |
| 5 | Clustering | H5 complete; README v1; final figures |
| 6 | Buffer | polish, re-reads, CV line |

**Mandatory README figures:** (1) equity curve 2003–2026 with shaded regimes; (2) H3 rolling correlation; (3) same-day/wait-one-day Δ bars by sub-period (H4); (4) discovery quality by arm (H5); (5) cost break-even curve; (6) 2020 event study.

**Cuts allowed if time runs short (in this order):** H5 sensitivity (PCA/k-means) → H5 Variant B → top-5 portfolio. **Never cuttable:** Gates 0–2, H1, H4, the synthetic test, anti-leakage.

---

## 8. Risks and fallback plans
| Risk | Mitigation |
|---|---|
| Rate limiting / dirty Yahoo data | incremental parquet cache; Stooq as a second provider; overnight batch downloads |
| 2003–09 attrition too high | shift replication to 2005–09 (logged); in extremis, a qualitative-only Gate 1 + Rubesam comparison |
| OOS results "too good" | mandatory checklist before believing it: anti-leakage, normalization convention, date alignment, adjusted vs. raw |
| Degenerate OPTICS (all noise or one mega-cluster) | declared fallback: k-means via silhouette as primary, logged in DEVIATIONS |
| Time | cuts allowed per §7, in that order |

---

## 9. Open decisions (to be ratified before starting)
1. Gate 1 band [0.05%, 0.45%]: OK, or do you want to tighten/widen it?
2. VIX regime threshold ≥ 25 as primary (terciles as robustness): OK?
3. OPTICS(min_samples=3, ξ=0.05) primary with k-means fallback: OK?
4. Replication start 2003 with a slip-to-2005 clause if attrition < 70%: OK?

*Once these four are ratified, the document freezes and Week 1 begins.*
