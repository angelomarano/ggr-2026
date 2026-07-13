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

## 2026-07-05 — Gate 2: ticker con dati Yahoo corrotti (rendimenti estremi e prezzi congelati), due filtri causali aggiunti

**Scoperta:** durante l'esecuzione one-shot di Gate 2 (universo pieno, 2010-2025),
il bootstrap decile-matched del portafoglio primario ha prodotto numeri
astronomicamente assurdi (media delle repliche dell'ordine di 10^18%). Un'indagine
sui dati grezzi ha trovato 25 ticker con rendimenti giornalieri superiori al 300%
in valore assoluto in almeno un giorno tra il 2009 e il 2026 (TNB, KRI, CBE, TIE,
NCC, BOL, CFC, MEE, CIN, PBG, BMC, CPWR, HPC, GLK, GDW, PTV, STI, UVN, HET, FSH,
EP, UPC, EQ, MI, PALM), quasi certamente per riciclo del simbolo ticker (una
societa' delistata il cui ticker viene riassegnato a un'altra entita', spesso
OTC/penny-stock, senza distinzione nella serie storica di Yahoo Finance).

**Impatto quantificato:**
- 57/192 run OOS (universo pieno) hanno avuto almeno uno dei 25 ticker corrotti
  sopravvivere al filtro di completezza formation preesistente (nessun NaN, quindi
  non escluso dal filtro "no-trade day" gia' presente).
- Contaminazione REALE confermata (non solo possibile sostituto casuale nel
  bootstrap) in 5 coppie su 2 run: sempre e solo BMC (2014-01 control rank117:
  BMC/BXP; 2014-04 top_20 rank10/17: BMC/MCD, BMC/BMS; 2014-04 control
  rank112/118: BMC/KO, BMC/CNP).
- Causa di questi 5 casi: NON un salto di rendimento estremo (le date esatte dei
  rendimenti estremi di BMC non cadono in nessuna delle due finestre di formation
  coinvolte), ma un prezzo Adj Close congelato (bit-identico) per mesi consecutivi
  (es. $2380.0 esatto da agosto a ottobre 2013, con volume sospetto che oscilla
  tra 2000 e 0) — un indice di prezzo normalizzato costante "combacia"
  artificialmente con qualunque altro titolo a bassa volatilita', abbassando la
  SSD in modo spurio.
- Nessuno dei 25 ticker corrotti appartiene al golden set (ne' quello di replica
  ne' quello OOS), tranne TIE che e' nel golden set di replica ma la cui finestra
  di corruzione (2010-2017) e' interamente dopo la finestra di Gate 1 (2003-2009):
  Gate 1 non e' contaminato.

**Fix applicato (src/formation.py, config.py):** due filtri causali, applicati
SOLO al formation period del run corrente (mai a dati futuri rispetto a quel run
— stesso principio del filtro GGR gia' esistente su "giorno senza scambi"):
1. `config.MAX_ABS_DAILY_RETURN` (default 3.0 = 300%): esclude un ticker se un
   suo rendimento giornaliero nel formation period supera la soglia in valore
   assoluto.
2. `config.MAX_CONSECUTIVE_FROZEN_DAYS` (default 5): esclude un ticker se il suo
   Adj Close resta bit-identico per piu' di N giorni di borsa consecutivi nel
   formation period. Deliberatamente basato solo sul prezzo, non sul volume (il
   volume di BMC e' esso stesso un segnale inaffidabile).

Entrambi i parametri sono marcati esplicitamente in config.py come aggiunte
POST-HOC, distinte dai parametri congelati dal protocollo originale
(OPEN_TRIGGER_SIGMAS, FORMATION_DAYS, ecc.).

**Verifica:** dopo il fix, tutti e 5 i casi di contaminazione reale sono risolti
(BMC escluso in entrambi i run, sostituito da coppie SSD-legittime, nessun altro
ticker corrotto entra al loro posto). Test sintetici aggiunti per entrambi i
filtri, incluse verifiche esplicite di causalita' (nessun look-ahead: lo stesso
ticker con la stessa anomalia e' escluso o meno a seconda che la specifica
finestra di formation del run la contenga).

**Nota — distinto dal problema Gate 0:** questo NON e' lo stesso problema gia'
documentato per Gate 0 (Yahoo che elimina interamente lo storico dei titoli
delistati). Stessa fonte dati (Yahoo Finance), due difetti diversi: Gate 0
riguarda l'ASSENZA di dati per titoli delistati; questo riguarda dati PRESENTI
ma corrotti (valori numericamente implausibili) per titoli il cui simbolo e'
stato riciclato dopo il delisting originale.

**Decisione:** Gate 2 (braccio full_universe) ri-eseguito una seconda volta dopo
il fix, esplicitamente loggato come tale (non un rilancio silenzioso): numeri
prima/dopo entrambi visibili in results/frozen/gate2_report.md e
gate2_results.json (chiave "full_universe_before_fix"). Braccio
golden_set_robustness non ri-eseguito: gia' verificato pulito, nessuno dei 25
ticker corrotti vi appartiene.

**Nota di conferma (verifica a posteriori, sola lettura):** la chiave
"full_universe" attualmente in gate2_results.json - quella usata per TUTTE le
tabelle di gate2_report.md e del README (statistiche descrittive a 12
combinazioni, le tre falsificazioni, E ANCHE H2/H3/H4) - e' l'output della
seconda esecuzione (post-fix), non un residuo della prima. Confermato
confrontando valori noti per differire tra le due esecuzioni:
b(HighVol) H2 = 0.31470% (t=2.143) in "full_universe" contro 0.31411%
(t=2.139) in "full_universe_before_fix"; correlazione grezza media H3 =
0.48245 contro 0.47850; delta H4 2010-2017 = 0.013890% contro 0.013993%.
Le differenze sono piccole (coerenti con sole 2 run su 192 che cambiano
selezione) ma non nulle: H2/H3/H4, non solo le descrittive e il bootstrap,
sono state ricalcolate post-fix.

## 2026-07-13 — H5: mismatch di scala nel sigma di trading per le coppie selezionate via Engle-Granger

**Osservato:** in notebooks/05_h5_discovery_quality.py, il sigma passato a
simulate_pair_wait_one_day (soglia di apertura |spread| > k*sigma) era
formation.spread_sigma per TUTTE e quattro le liste candidate, incluse
cluster_coint (Variante B) e brute_force - selezionate pero' via
Engle-Granger sul RESIDUO del log-prezzo, non sull'indice di prezzo
normalizzato. spread_sigma stima la deviazione standard di
P*_i - P*_j (indice di prezzo normalizzato, la stessa quantita' che
src/trading.py effettivamente soglia), quindi resta corretto per
ggr_ssd/cluster_ssd; per cluster_coint/brute_force e' una stima presa da
una quantita' diversa da quella su cui la coppia e' stata selezionata.

**Causa:** engle_granger_pair (src/selection_cluster.py) non esponeva la
deviazione standard del proprio residuo - solo t_stat/p_value/half_life_days
- quindi non c'era alcun valore alternativo da passare al motore di trading
per le coppie selezionate via cointegrazione.

**Fix:** aggiunto il campo "residual_std" (resid.std(ddof=0), nessuna
regressione aggiuntiva - il residuo esiste gia' nella funzione) al dict di
ritorno di engle_granger_pair, esposto come colonna nelle tabelle di
cointegration_intra_cluster_ranking e brute_force_cointegration_screen.
notebooks/05_h5_discovery_quality.py aggiornato (SIGMA_SOURCE_BY_LIST) per
usare residual_std per cluster_coint/brute_force e continuare a usare
spread_sigma per ggr_ssd/cluster_ssd (gia' coerente, invariato). Test
sintetico aggiunto (coppia cointegrata nota, P2=P1*exp(u_t), u AR(1)
phi=0.9): residual_std converge al valore teorico stazionario
innovation_std/sqrt(1-phi^2) su un campione grande (n=5000, tolleranza
+-15%, necessaria per lasciare convergere la superconsistency della
regressione di cointegrazione).

**Verifica esplicita (non solo assunta):** il test di stazionarieta' OOS
(% OOS-stationary) e la half-life OOS sono confermati indipendenti dalla
scala del sigma - entrambi derivano esclusivamente dalla regressione
Engle-Granger ricalcolata sul trading period stesso (p-value di
statsmodels, coefficiente AR(1) del residuo), nessun sigma esterno vi
entra mai. Verificato con un controllo programmatico
(_verify_stationarity_and_half_life_unchanged) che confronta pre-fix e
post-fix campo per campo su tutte e 4 le liste: identici byte-per-byte.

**Impatto quantificato (8 run campionati, 7 riusciti):** ggr_ssd/cluster_ssd
invariati (come atteso, sigma_source non cambiato). cluster_coint: %
convergenza 21.6% -> 35.3% (+13.7 punti), rendimento medio mensile
-0.0437% -> -0.2022% (t da -0.26 a -0.80, resta non significativo).
brute_force: invariato a 0.0%/n/a - le uniche 7 coppie candidate (su 2
run su 7 con survivor BH+filtro) non attraversano la soglia nemmeno con
il sigma corretto (verificato caso per caso, es. GLW-MCO: range di
spread nel trading period [-0.27, +0.20], soglia post-fix 0.385 - vicina
ma non superata).

**Decisione:** notebooks/05_h5_discovery_quality.py ri-eseguito una
seconda volta dopo il fix, esplicitamente loggato come tale (non un
rilancio silenzioso): numeri prima/dopo entrambi visibili in
results/replication/h5_discovery_quality.md ("Second execution" section)
e h5_discovery_quality.json (chiavi "pre_fix"/"post_fix",
"scale_independence_check"). Nessun altro gate o risultato gia'
pubblicato (Gate 1, Gate 2, README) e' toccato da questo fix.
