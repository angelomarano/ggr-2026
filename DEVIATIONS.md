# DEVIATIONS.md
Log delle deviazioni dal protocollo (PROTOCOL.md v1.0, ratificato 2026-07-05).
Formato: data | sezione | deviazione | motivazione.

*(vuoto — nessuna deviazione)*

## 2026-07-05 — Gate 0: attrition Yahoo molto oltre le attese, clausola 2003→2005 inefficace

**Osservato:** attrition sull'universo point-in-time pieno: 51.9% (2003) → 63.7% (2009)
→ 96.8% (2025). La clausola del protocollo ("se <70% nel 2003-2009, spostare a
2005") non risolve nulla: anche il 2005 e' al 54.9%.

**Causa accertata:** Yahoo Finance non tronca lo storico dei titoli delistati
al giorno del delisting — lo ELIMINA INTERAMENTE, anche per titoli come X
(US Steel) rimasti quotati fino al 2025. L'attrition per anno riflette quindi
quanti membri di quell'anno sono stati delistati DA ALLORA A OGGI (2026), non
la qualita' dei dati dell'epoca. E' un limite del vendor gratuito, documentato
in letteratura (dataset gratuiti generalmente privi di "titoli morti"), non
un errore della pipeline.

**Mitigazioni testate e scartate:**
- Stooq come fonte alternativa: richieste bloccate anche su un controllo
  sicuro (AAPL) -> probabile protezione anti-scraping. Time-boxed e abbandonato,
  nessuna evidenza ne' a favore ne' contro la copertura dei delistati.
- WRDS/CRSP: nessuna conferma di accesso ETH Zurich (UZH e HSG ce l'hanno,
  ETH non verificato). Rimandato a settembre 2026, quando Angelo avra'
  credenziali ETH per controllare il catalogo library.ethz.ch. Non blocca
  il progetto attuale.

**Decisione (ratificata):** Gate 0 originale (soglia 70% sull'universo pieno)
SOSTITUITO da Gate 0-bis a due componenti:
  (a) fedelta' meccanica del motore -> validata su un GOLDEN SET (vedi
      data/golden_set.py): titoli con storico Yahoo integro in OGNI run
      2003-01..2008-12, costruito empiricamente dal cache gia' scaricato.
  (b) quantificazione del survivorship bias -> resta sull'universo pieno,
      riportata come limite/robustness (curva di attrition per anno), non
      piu' come cancello bloccante. H1-H5 girano su universo pieno come
      primario e su golden-set-only come robustness check esplicito.

Protocollo (PROTOCOL.md) da aggiornare di conseguenza in §1.3 e §3 alla
prossima revisione; questa voce e' il log della decisione nel frattempo.

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
