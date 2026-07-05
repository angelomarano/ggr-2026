"""
check_stooq_recovery_v2.py — Come il precedente, ma con header User-Agent
(Stooq spesso rifiutano richieste senza uno user-agent da browser, dando 404
anche per ticker validi) e un CONTROLLO su un titolo sicuramente vivo.

Uso: python check_stooq_recovery_v2.py
"""
import time
import urllib.request

import pandas as pd

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# AAPL come controllo (deve funzionare); poi il campione di delistati.
CONTROL = ["AAPL"]
SAMPLE = ["X", "BSC", "CELG", "ANTM", "BLL", "WLP", "ATVI", "XLNX", "BNI", "BRCM"]


def try_stooq(ticker: str) -> dict:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        raw = urllib.request.urlopen(req, timeout=10).read()
        text = raw.decode("utf-8", errors="replace")
        if text.strip().lower().startswith("no data") or "<html" in text.lower():
            return {"ticker": ticker, "stooq_rows": 0, "note": "no data / html error page"}
        from io import StringIO
        df = pd.read_csv(StringIO(text))
        if df.empty or "Date" not in df.columns:
            return {"ticker": ticker, "stooq_rows": 0, "note": "csv vuoto o senza colonna Date"}
        df["Date"] = pd.to_datetime(df["Date"])
        return {
            "ticker": ticker,
            "stooq_rows": len(df),
            "first": df["Date"].min().date(),
            "last": df["Date"].max().date(),
        }
    except Exception as e:  # noqa: BLE001
        return {"ticker": ticker, "stooq_rows": 0, "error": str(e)}


print("=== CONTROLLO (deve avere migliaia di righe fino a oggi) ===")
for t in CONTROL:
    print(try_stooq(t))

print("\n=== CAMPIONE TITOLI DELISTATI ===")
results = []
for t in SAMPLE:
    r = try_stooq(t)
    print(r)
    results.append(r)
    time.sleep(1)

pd.DataFrame(results).to_csv("stooq_recovery_check_v2.csv", index=False)
print("\nSalvato in stooq_recovery_check_v2.csv")
print("Se il CONTROLLO fallisce anche lui -> problema di rete/formato, non di copertura.")
print("Se il CONTROLLO funziona ma il CAMPIONE no -> Stooq davvero non ha questi delistati.")