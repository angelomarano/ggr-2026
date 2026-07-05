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
