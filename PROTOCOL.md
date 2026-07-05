# PROTOCOL — "Pairs Trading, Twenty Years Later"
### Replica congelata di Gatev–Goetzmann–Rouwenhorst (RFS 2006), estensione out-of-sample 2010–2026, e braccio di selezione via clustering

**Versione:** 1.0 — da congelare prima del Gate 2.
**Regola d'oro:** ogni deviazione da questo documento dopo il Gate 1 va registrata in `DEVIATIONS.md` con data e motivazione. Nessuna eccezione. Il valore del progetto in colloquio dipende dal poter dire "il protocollo era scritto prima di vedere i risultati".

---

## 0. Obiettivo e principi

**Domanda ombrello:** il pairs trading alla GGR è morto, dormiente, o vivo solo nella turbolenza? E una selezione delle coppie via clustering migliora la *qualità delle scoperte* rispetto alla ricerca brute-force?

**Deliverable:** repository GitHub pubblico (codice + README con domanda/metodo/risultati/figure/limiti) + una riga di CV difendibile in ogni dettaglio.

**Principi non negoziabili:**
1. **Frozen protocol.** Tutti i parametri sono fissati in questo documento (§2, §4). Nessuna ottimizzazione di parametri, mai. La convenzione GGR (12m/6m, 2σ) si eredita, non si tara: l'assenza di data snooping È il risultato.
2. **Validation gates.** Non si passa alla fase successiva senza superare il gate (§3). Il run out-of-sample 2010–2026 si esegue **una sola volta**.
3. **Separazione formation/trading.** Nessuna quantità calcolata su dati del trading period può influenzare selezione o soglie. Test automatico anti-leakage obbligatorio (§6, tests).
4. **Onestà sui limiti.** Survivorship, costi e potenza statistica si quantificano e si scrivono nel README, non si nascondono.

---

## 1. Dati

### 1.1 Fonti
| Dato | Fonte primaria | Backup/validazione | Note |
|---|---|---|---|
| Costituenti S&P 500 point-in-time | GitHub `fja05680/sp500` (dal 1996) | GitHub `hanshof/sp500_constituents` | snapshot mensile: universo alla data di inizio formation |
| Prezzi giornalieri (adjusted + raw close, volume) | Yahoo Finance via `yfinance` | Stooq (`pandas-datareader`) | adjusted close = proxy total return |
| Risk-free (3M T-bill, DTB3) | FRED | — | per excess return e Sharpe |
| Fattori FF3 + Momentum + ST Reversal | Ken French Data Library | — | per le regressioni di rischio |
| Indice e volatilità (^GSPC, ^VIX) | Yahoo | — | per H2 (regimi) |

**Finestra dati totale:** 2002-01-01 → 2026-06-30.
**Finestra di replica (Gate 1):** trading periods con inizio da 2003-01 a 2008-12 (ultima chiusura giugno 2009) — sovrapposta al terzo sottoperiodo di Do & Faff (2003–giugno 2009, media 0.24%/mese su CRSP).
**Finestra OOS congelata:** trading periods con inizio da 2010-01 a 2025-12 (ultima chiusura giugno 2026).

### 1.2 Costruzione dell'universo (per ogni formation date, mensile)
1. Prendere la lista dei costituenti S&P 500 alla data di inizio del formation period (point-in-time: mai la lista attuale).
2. Scaricare i prezzi per l'intero formation+trading period.
3. **Filtro GGR:** eliminare i titoli con ≥1 giorno senza scambi (proxy: volume 0 o prezzo mancante) nel formation period. Nessun filtro di prezzo minimo (GGR puro).
4. **Tabella di attrition (obbligatoria, Gate 0):** per ogni anno, riportare (a) numero costituenti, (b) % con storia prezzi completa su Yahoo/Stooq, (c) % persa per delisting/ticker change. Questa tabella va nel README.

### 1.3 Survivorship: posizione dichiarata
Yahoo non conserva lo storico dei titoli delisted → una frazione dell'universo storico è irrecuperabile con dati gratuiti. **Direzione del bias:** nel pairs trading i delisting sono spesso l'esito di divergenze mai convergite (posizioni in perdita chiuse forzatamente); escluderli **gonfia** i rendimenti. Quindi i nostri numeri OOS sono un **upper bound**: se anche l'upper bound è ≈0, la conclusione "declino completato" è *a fortiori*. Se invece troviamo profitti, il caveat va dichiarato come limite principale. (GGR gestiscono i delisting col delisting return; noi non possiamo: dirlo esplicitamente.)

### 1.4 Pulizia
- Total return index: adjusted close; controllo di sanità raw vs adjusted su 10 ticker campione (split noti: AAPL 2020, TSLA 2020, NVDA 2021/2024).
- Prezzi mancanti isolati nel trading period: nessun forward-fill per i segnali; se un titolo smette definitivamente di quotare, la posizione si chiude all'ultimo prezzo disponibile (regola GGR adattata).
- Cache locale in Parquet; ogni download loggato con timestamp (riproducibilità).

---

## 2. Metodologia core — replica GGR (congelata)

### 2.1 Parametri
| Parametro | Valore | Fonte |
|---|---|---|
| Formation period | 12 mesi (252 gg trading) | GGR |
| Trading period | 6 mesi (126 gg) | GGR |
| Staggering | nuovo portafoglio ogni mese → 6 portafogli sovrapposti, media alla Jegadeesh–Titman | GGR |
| Matching | minima SSD tra indici di prezzo total-return normalizzati | GGR |
| Portafogli | top-5, **top-20 (primario)**, coppie 101–120 (controllo) | GGR |
| Trigger apertura | \|spread\| > 2σ, σ = std dello spread nel formation | GGR |
| Chiusura | crossing dei prezzi normalizzati; fine periodo; delisting | GGR |
| Riapertura | consentita dopo ogni convergenza | GGR |
| Unicità titoli | un titolo può comparire in più coppie (GGR non impongono unicità) | GGR |
| Esecuzione | **due varianti sempre:** same-day e wait-one-day | GGR |
| Inferenza | Newey–West 6 lag su serie mensile | GGR |

### 2.2 Formule (da implementare esattamente così)
- Indice normalizzato nel formation: `P*_it = ∏_{τ=1..t} (1 + r_iτ)`, con `P*_i0 = 1`, r = rendimento total-return giornaliero.
- Distanza: `SSD_ij = Σ_t (P*_it − P*_jt)²` su 252 gg.
- **Convenzione trading period (ambiguità nota del paper — GGR non specificano se le serie vengono ri-normalizzate all'inizio del trading period):** scelta dichiarata a priori: entrambe le gambe si **ri-normalizzano a 1 al primo giorno del trading period**; spread `s_t = P*_1t − P*_2t`; σ resta quella stimata sul formation. **Task W2:** confrontare questa convenzione con il notebook di Rubesam; se diverge, documentare in `DEVIATIONS.md` e riportare la sensitività del Gate 1 a entrambe le convenzioni.
- Apertura a `|s_t| > 2σ`: short sulla gamba alta, long sulla bassa, $1 per gamba.
- Payoff giornaliero della coppia: `w_L,t·r_L,t − w_S,t·r_S,t`, con pesi che evolvono `w_{t} = w_{t−1}(1+r_{t−1})` da 1 all'apertura (mark-to-market, buy-and-hold entro il trade). L'aggregazione tra coppie e la composizione giornaliera→mensile seguono **esattamente le eq. (2)–(3) del paper** (media value-weighted con pesi evolventi, poi compounding a mensile): implementarle così, non con semplificazioni equal-weight giornaliere. Nota di GGR: la cassa dei payoff intermedi rende zero interesse (assunzione conservativa) — replicarla.
- **Return on committed capital** = somma payoff / n. coppie del portafoglio (20). **Return on employed capital** = / n. coppie effettivamente aperte. Si riportano entrambi; il primario per i test è il committed.
- Rendimento mensile della strategia = media dei 6 portafogli sfalsati attivi nel mese.

### 2.3 Statistiche riportate (per ogni portafoglio, per ogni variante)
Media mensile, t Newey–West(6), SE, Sharpe annualizzato, min/max, skewness, kurtosis, % mesi negativi, max drawdown, n. medio round-trip per coppia, durata media posizione, % coppie mai aperte.

### 2.4 Test di falsificazione (da replicare, non opzionali)
1. **Bootstrap coppie casuali** (200 repliche): sostituire ogni coppia vera con due titoli casuali dello stesso decile di rendimento del mese precedente, stessi event dates → atteso ≈0/negativo. Se il nostro bootstrap dà rendimenti positivi, la pipeline ha un bug o l'effetto è reversal mascherato.
2. **Decomposizione long vs short:** alpha a 5 fattori per gamba. Atteso (Tabella 7 GGR, top-20): alpha del portafoglio **short negativo e significativo** (−0.52%/mese, t=−3.35 → contributo positivo alla strategia che lo shorta), alpha del **long ≈ 0 e non significativo** (+0.24%, t=1.27). La profittabilità viene dalla gamba short: se nella nostra replica domina il long, c'è un errore di segno o di costruzione.
3. **Regressione fattoriale:** excess return mensili su FF3 + MOM + ST-Reversal; riportare alpha, t(NW), loadings.

---

## 3. Gates di validazione

### Gate 0 — Integrità dati (fine W1)
- [ ] Universo point-in-time ricostruito per tutte le formation dates 2002–2025.
- [ ] Tabella attrition compilata; se la % di storia completa nel 2003–2009 è < 70%, spostare l'inizio della finestra di replica al 2005 e loggarlo in `DEVIATIONS.md`.
- [ ] Spot-check prezzi: 10 ticker × 20 date vs Stooq, discrepanze < 0.5%.
- [ ] Unit test superati: normalizzazione, SSD su serie sintetiche, coppia sintetica con trade calcolati a mano (§6).
- [ ] Test anti-leakage superato: troncando i dati alla fine del formation, la selezione delle coppie è identica bit-per-bit.

### Gate 1 — Fedeltà di replica (2003–2009, fine W3)
**Banda quantitativa (dichiarata ora, con una condizione da chiudere in W1):** top-20, wait-one-day, committed capital → media mensile in **[0.05%, 0.45%]**. Il riferimento è Do & Faff, 0.24%/mese su CRSP full-universe 2003–giu 2009 — **ma la misura esatta a cui quel numero si riferisce (committed vs employed) va verificata sul loro paper prima del Gate 1** (loro calcolano entrambe, come GGR). **Task W1:** recuperare la tabella di D&F, fissare misura e regola di esecuzione del target, e solo allora congelare la banda sulla stessa misura. Il nostro universo large-cap-only e survivorship-biased può stare sopra o sotto: la banda riflette questa incertezza.

**Ancore numeriche dal paper (CRSP lug1963–dic2002, solo documentazione — non direttamente replicabili coi nostri dati):** top-20 same-day: 1.44%/mese fully-invested (t=11.56) / 0.81% committed; top-20 wait-one-day: 0.90% fully-invested (t=9.29) / 0.52% committed; trigger medio top-20 ≈ 5.3%; ~19.3 coppie su 20 aprono; ~1.96 round-trip per coppia; durata media ~3.8 mesi.

**Invarianti qualitative (tutte obbligatorie):**
- wait-one-day < same-day (gap positivo su entrambe le misure di capitale);
- gamba short dominante: alpha short negativo e significativo, alpha long ≈ 0 (vedi §2.4.2);
- rendimenti positivi nel 2008 (turbolenza) — coerente con Do & Faff;
- bootstrap coppie casuali ≈ 0 o negativo;
- coppie 101–120 < top-20;
- statistiche di trading nell'ordine GGR: ~2 round-trip per coppia, durata ~3–4 mesi, quasi tutte le coppie aprono;
- diagnostica di composizione: attesa concentrazione dei top pair in settori a bassa volatilità (in GGR il 71% dei titoli nei top-20 erano utilities; su un universo S&P 500 la composizione differirà, ma top pair dominati da titoli ad alta volatilità = campanello d'allarme).
**Se fuori banda:** caccia al bug (returns computation, overlap averaging, convenzione di normalizzazione), **mai** ritocco dei parametri. Confronto qualitativo con la replica Rubesam (CRSP) come riferimento esterno.

### Gate 2 — Frozen run OOS (W4)
Si esegue **una volta** su 2010–2026 con il codice validato al Gate 1, senza modifiche. Output scritti in `results/frozen/` e mai rigenerati (hash del commit nel README).

---

## 4. Ipotesi, metriche, predizioni, interpretazioni

### H1 — Il declino si è completato? (primaria)
- **Metrica primaria:** media mensile excess return, top-20, wait-one-day, committed capital, 2010–2026.
- **Test:** H0: μ ≤ 0, t Newey–West(6); CI 95% con block bootstrap stazionario (blocco medio 6 mesi, 10.000 repliche).
- **Predizione (da Do & Faff: 0.86 → 0.37 → 0.24%/mese; misura esatta da fissare in W1, vedi Gate 1):** ≤ 20 bp/mese, non significativo; ≈ 0 dopo costi (§5).
- **Interpretazioni:** μ ≈ 0 → "the decline completed", conferma con inferenza moderna su 16 anni mai testati nel filone GGR (esito atteso, pienamente pubblicabile su GitHub). μ > 0 significativo → risultato sorprendente: prima di crederci, checklist anti-leakage §6 e verifica che non sia guidato da 2020/2022 (→ H2).

### H2 — L'eccezione-turbolenza ha retto? (la più interessante)
- **Definizione regime (a priori):** mese "high-vol" se VIX medio mensile ≥ 25 (soglia fissa, interpretabile). Robustness: terzili del VIX su distribuzione trailing 10 anni.
- **Test:** regressione `r_t = a + b·HighVol_t + ε`, errori NW(6); atteso b > 0 significativo. Più event study dichiarati: **feb–giu 2020** e **gen–ott 2022** (rendimenti cumulati con CI bootstrap).
- **Razionale dal paper:** GGR trovano performance forte negli anni '70 (mercato debole) e piatta nel bull run anni '90; Do & Faff la confermano in 2000–02 e 2007–09. Il COVID è il test di turbolenza più netto dal 2008 e il filone GGR pubblicato si ferma ~2014.
- **Interpretazioni:** b>0 con profitti concentrati in 2020/2022 → "strategia dormiente, si sveglia nello stress" (il finding più spendibile in colloquio). b≈0 → anche l'eccezione-turbolenza è morta: finding altrettanto netto.

### H3 — Il fattore latente è dormiente o risvegliato?
- **Metrica:** correlazione rolling 24 mesi tra rendimenti top-20 e 101–120 (portafogli disgiunti). GGR: 0.48 full-sample; 0.51 pre-1989 → 0.18 post-1988; sui residui a 5 fattori: 0.42 → 0.20 (quindi il fattore comune non è spiegato dai fattori standard — calcolare entrambe le versioni, raw e residui).
- **Predizione:** media ~0.2 con picchi nei mesi high-vol.
- **Output:** figura "GGR Fig. 4, esteso al 2026" — centrale nel README.

### H4 — Il bid-ask bounce è evaporato (risultato positivo garantito)
- **Metrica:** Δ = media(same-day) − media(wait-one-day), per sottoperiodo (2003–09 vs 2010–17 vs 2018–26), calcolata su **entrambe** le misure di capitale. Baseline GGR per il top-20: **~54 bp/mese sul fully-invested** (1.44% → 0.90%) e **~28 bp/mese sul committed** (0.81% → 0.52%); il paper riporta cali di 30–55 bp (fully-invested) e 20–35 bp (committed) tra i portafogli. Dalla versione fully-invested GGR derivano uno spread effettivo implicito di ~81 bp per titolo (162 bp round-trip per coppia) in un mondo pre-decimalizzazione.
- **Test:** differenza con SE bootstrap accoppiato (stesse date-evento).
- **Predizione:** Δ < 10 bp/mese nel 2010–2026 (spread su large cap oggi = pochi bp).
- **Perché conta:** Δ è una stima *diretta* del costo di microstruttura rilevante per la strategia. Anche se H1 dà zero, H4 dà un finding pulito e quantitativo. Materiale d'oro per colloqui (microstruttura, effective spread).

### H5 — Selezione via clustering: qualità delle scoperte (l'estensione)
**Pipeline alternativa (tutto il resto identico al core):**
1. Matrice rendimenti giornalieri del formation (252 × N), standardizzazione per titolo.
2. PCA: **10 componenti** (a priori; Sarmento & Horta raccomandano < 15). Sensitivity dichiarata: {5, 15}.
3. Clustering **primario: OPTICS** (min_samples = 3, ξ = 0.05) sui loadings; i punti "noise" sono esclusi. Sensitivity: k-means con k scelto via silhouette in [5, 30].
4. Coppie candidate = solo intra-cluster.
5. **Variante A (confronto pulito con GGR):** ranking SSD intra-cluster, top-20.
   **Variante B (cointegrazione):** Engle–Granger sui log-prezzi (ADF sui residui, p-value MacKinnon), soglia p < 0.05, filtro half-life AR(1) dello spread in [5, 60] giorni, top-20 per p-value.
6. **Comparatore brute-force:** Engle–Granger su tutte le ~125.000 coppie con correzione Benjamini–Hochberg al 5%; conteggio sopravvissuti; top-20 per p-value. **Robustness dichiarata:** i test sulle coppie sono fortemente dipendenti (titoli condivisi, fattori comuni) e BH assume dipendenza positiva (PRDS) — riportare anche Benjamini–Yekutieli, valido sotto dipendenza arbitraria, e commentare la differenza. Punto d'oro da statistico in colloquio.

**Le tre metriche di confronto (dichiarate ora):**
1. **Discovery quality (primaria di H5):** % delle coppie selezionate il cui spread nel *trading period* è stazionario (ADF p < 0.10, soglia a priori) + % che convergono almeno una volta + distribuzione delle half-life OOS. Confronto: cluster-A vs cluster-B vs brute-force-BH vs GGR-SSD.
2. **Contabilità del multiple testing:** n. test eseguiti per braccio (≈125k vs migliaia); falsi positivi attesi sotto ipotesi nulla globale a α=0.05 (≈6.200 nel brute-force); coppie sopravvissute a BH.
3. **Performance netta:** stesse statistiche di H1 sulle quattro liste.

**Predizione onesta:** differenze di performance entro il rumore; differenze di discovery quality misurabili a favore della selezione ristretta. Il messaggio non è "clustering = più soldi" ma "clustering = meno false scoperte a parità di soldi" — che è esattamente il messaggio giusto per un colloquio da QR.

---

## 5. Costi di transazione (curva, non un numero)
- Livello 0: risultati lordi same-day (limite superiore teorico).
- Livello 1: wait-one-day (ingloba il costo di microstruttura implicito — è la variante primaria).
- Livello 2: **griglia esplicita** c ∈ {0, 5, 10, 20, 40} bp per lato per trade; un round-trip di coppia = 4 trade → costo 4c per round-trip. Output: curva rendimento/Sharpe vs c e **costo di break-even c\***.
- Short costs: scenario qualitativo +25 bp/anno di borrow su general collateral; menzione dei nomi hard-to-borrow come limite.
- Riferimento: Do & Faff usano costi time-varying; noi preferiamo la curva perché rende il risultato robusto a qualsiasi opinione sul livello "giusto".

---

## 6. Struttura del repository e implementazione

```
ggr-2026/
├── README.md              # domanda → metodo → risultati → figure → limiti
├── PROTOCOL.md            # questo documento (congelato pre-Gate 2)
├── DEVIATIONS.md          # log deviazioni datate
├── config.py              # TUTTI i parametri del §2 e §4, un solo posto
├── data/
│   ├── universe.py        # costituenti point-in-time + tabella attrition
│   ├── prices.py          # yfinance/Stooq, cache parquet, validazioni
│   └── factors.py         # French, FRED, VIX
├── src/
│   ├── formation.py       # normalizzazione, SSD, matching
│   ├── selection_cluster.py  # PCA, OPTICS/k-means, EG intra-cluster, BH
│   ├── trading.py         # trigger, crossing, delisting, wait-one-day
│   ├── returns.py         # pesi compounding, committed/employed, overlap averaging
│   ├── inference.py       # NW, block bootstrap, regressioni fattoriali
│   └── regimes.py         # VIX/vol regimes, event windows
├── tests/
│   ├── test_synthetic_pair.py   # coppia costruita a mano, trade e P&L noti
│   ├── test_no_lookahead.py     # troncamento al formation ⇒ selezione identica
│   └── test_returns_math.py     # pesi compounding, overlap averaging
├── notebooks/             # 01_data_audit, 02_replication, 03_frozen_oos,
│                          # 04_clustering_arm, 05_figures
└── results/
    ├── replication/       # Gate 1
    └── frozen/            # Gate 2 — scritti una volta, hash del commit nel README
```

**Librerie:** pandas, numpy, statsmodels (`adfuller`, `coint`, OLS con cov HAC), scikit-learn (PCA, OPTICS, KMeans), scipy, yfinance, pandas-datareader, matplotlib. **Nessun framework di backtesting esterno:** il loop GGR (crossing close, pesi compounding, overlap) è specifico e scriverlo — e testarlo — è il punto del progetto.

**Test sintetico (specifica):** due serie con cointegrazione imposta (P2 = P1·e^{u_t}, u ~ AR(1) stazionario) + 2 divergenze iniettate a date note → asserire date di apertura/chiusura e P&L calcolato a mano su foglio. Se questo test non passa, niente dati reali.

---

## 7. Timeline (part-time, 5 settimane + buffer)
| Sett. | Obiettivo | Definition of done |
|---|---|---|
| 1 | Dati | Gate 0 superato; tabella attrition scritta; tabella Do & Faff recuperata e misura-target del Gate 1 congelata |
| 2 | Motore | formation+trading+returns; test sintetico verde; prima replica grezza gira |
| 3 | Replica | Gate 1 superato; falsificazioni (bootstrap, long/short, fattori) replicate |
| 4 | OOS | Gate 2 eseguito; H1–H4 con tabelle e figure |
| 5 | Clustering | H5 completo; README v1; figure finali |
| 6 | Buffer | rifiniture, riletture, riga CV |

**Figure obbligatorie del README:** (1) equity curve 2003–2026 con regimi ombreggiati; (2) rolling correlation H3; (3) barre Δ same-day/wait-one-day per sottoperiodo (H4); (4) discovery quality per braccio (H5); (5) curva break-even costi; (6) event study 2020.

**Tagli ammessi se il tempo stringe (in quest'ordine):** sensitivity di H5 (PCA/k-means) → variante B di H5 → portafoglio top-5. **Mai tagliabili:** Gate 0–2, H1, H4, test sintetico, anti-leakage.

---

## 8. Mappa colloquio (ogni scelta → la domanda che ti faranno)
| Domanda attesa | Risposta ancorata al progetto |
|---|---|
| "Come hai evitato il look-ahead bias?" | Universo point-in-time; selezione solo su formation; test automatico di troncamento; σ stimata solo sul formation |
| "Perché 12/6 mesi e 2σ?" | Convenzione GGR ereditata e congelata: zero gradi di libertà = zero snooping. Il protocollo era scritto prima dei risultati |
| "Come verifichi la cointegrazione?" | EG two-step, ADF sui residui con p-value MacKinnon; differenza correlazione vs cointegrazione; half-life via AR(1)/OU |
| "E il multiple testing su 125k coppie?" | ≈6.200 falsi positivi attesi ad α=5%; BH nel braccio brute-force; clustering come riduzione a priori dello spazio di ricerca |
| "Survivorship bias?" | Quantificato in tabella; direzione argomentata (gonfia i rendimenti → i risultati sono upper bound) |
| "Costi di transazione più alti?" | Curva di break-even: la risposta è un numero, non un'opinione |
| "Il risultato è negativo, quindi?" | Il deliverable è l'inferenza, non lo Sharpe: declino confermato OOS, eccezione-turbolenza testata, spread effettivo collassato |

---

## 9. Rischi e piani B
| Rischio | Mitigazione |
|---|---|
| Rate limit / dati sporchi Yahoo | cache parquet incrementale; Stooq come secondo provider; download in batch notturni |
| Attrition 2003–09 troppo alta | spostare replica a 2005–09 (loggato); in extremis Gate 1 solo qualitativo + confronto Rubesam |
| Risultati OOS "troppo belli" | checklist obbligatoria prima di crederci: anti-leakage, convenzione normalizzazione, allineamento date, adjusted vs raw |
| OPTICS degenere (tutto noise o un mega-cluster) | fallback dichiarato: k-means via silhouette come primario, loggato in DEVIATIONS |
| Tempo | tagli ammessi §7, in quell'ordine |

---

## 10. Decisioni aperte (da ratificare prima di iniziare)
1. Banda del Gate 1 [0.05%, 0.45%]: ok o vuoi stringerla/allargarla?
2. Soglia regime VIX ≥ 25 come primaria (terzili come robustness): ok?
3. OPTICS(min_samples=3, ξ=0.05) primario con k-means fallback: ok?
4. Inizio replica 2003 con clausola di slittamento a 2005 se attrition < 70%: ok?

*Ratificate queste quattro, il documento si congela e si parte dalla Settimana 1.*
