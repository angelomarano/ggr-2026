# Setup (Settimana 1 — cosa fare stasera)

1. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. Verifica universo (già testato, deve passare anche da te):
   `python -c "import sys; sys.path.insert(0,'.'); from data.universe import *; m=load_membership('data/raw/sp500_membership.csv'); print(len(constituents_at(m,'2010-01-04')))"`  → atteso 499
3. Download prezzi (ore; riprendibile, usa la cache):
   `python -m data.prices`
4. Report attrition (Gate 0):
   `python -m data.prices attrition`

## Cosa rimandare indietro nella chat
- `results/replication/attrition.csv`
- `data/raw/download_failures.csv` (conteggio + 10 righe di esempio)
- La media annua di `share_complete` stampata a fine run

Con questi tre output chiudiamo il Gate 0, decidiamo la clausola 2003→2005,
e nel prossimo incremento consegno il motore formation/trading/returns
con il test della coppia sintetica calcolata a mano.

## Note
- I fallimenti di download NON sono errori da sistemare: sono la materia
  prima della tabella di attrition (delisted/renamed → protocollo §1.3).
- Stooq serve solo per spot-check: non entra nel matching (niente dividendi).
