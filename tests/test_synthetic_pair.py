"""
test_synthetic_pair.py — Il test del PROTOCOL.md §6: coppia sintetica con
trade e P&L calcolati A MANO. Se questo non passa, niente dati reali.

Due episodi indipendenti (una chiamata a simulate_pair_same_day ciascuno,
cosi' lo stato si resetta):

EPISODIO 1 — apertura long2/short1, chiusura per CROSSING dopo 1 giorno.
  sigma=0.05 (soglia=0.10). returns_1=[+0.20, -1/12], returns_2=[0.00, 0.10].
  Giorno1: P1=1.20, P2=1.00, spread=+0.20 > 0.10 -> apre long leg2/short leg1.
  Giorno2: P1=1.20*(11/12)=1.10, P2=1.00*1.10=1.10, spread=0 -> crossing, chiude.
  Payoff giorno2 = 1*r2 - 1*r1 = 0.10 - (-1/12) = 11/60 (a mano, verificato).

EPISODIO 2 — apertura long1/short2, tenuta 2 giorni con compounding dei
pesi, chiusura forzata a FINE PERIODO (nessun crossing naturale).
  sigma=0.05. returns_1=[-0.20, 0.00, 0.30], returns_2=[0.00, 0.05, 0.02].
  Giorno1: P1=0.80, P2=1.00, spread=-0.20 < -0.10 -> apre long leg1/short leg2.
  Giorno2: payoff = w_long(1.0)*0.00 - w_short(1.0)*0.05 = -0.05
           pesi dopo: w_long=1.0, w_short=1.05 (spread ancora -0.25, no cross)
  Giorno3: payoff = w_long(1.0)*0.30 - w_short(1.05)*0.02 = 0.30-0.021 = 0.279
           spread finale = 1.04-1.071 = -0.031 (stesso segno -> NESSUN crossing
           naturale) -> chiusura forzata per fine periodo (t==n).
  Totale = -0.05 + 0.279 = 0.229 (a mano, verificato).
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.trading import simulate_pair_same_day, simulate_pair_wait_one_day

TOL = 1e-9


def test_episode1_crossing_close():
    returns_1 = np.array([0.20, -1 / 12])
    returns_2 = np.array([0.00, 0.10])
    r = simulate_pair_same_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert abs(r["spread"][1] - 0.20) < TOL, "spread giorno1 atteso 0.20"
    assert len(r["trades"]) == 2, f"attesi 2 eventi (open+close), trovati {len(r['trades'])}"

    open_ev, close_ev = r["trades"]
    assert open_ev["event"] == "open" and open_ev["day"] == 1
    assert open_ev["direction"] == "long2_short1", "spread>soglia -> long la gamba bassa (2), short l'alta (1)"
    assert close_ev["event"] == "close" and close_ev["day"] == 2
    assert close_ev["reason"] == "crossing"
    assert abs(close_ev["spread"]) < TOL, "al crossing lo spread deve essere ~0"

    expected_payoff_day2 = 11 / 60
    assert abs(r["daily_payoff"][1] - expected_payoff_day2) < TOL
    assert abs(r["daily_payoff"][0] - 0.0) < TOL, "nessun payoff il giorno di apertura"
    assert abs(r["total_payoff"] - 11 / 60) < TOL

    # long/short leg decomposition: long leg is 2, short leg is 1.
    # long_contribution = w_long * r2[1] = 1.0 * 0.10 = 0.10
    # short_contribution = w_short * r1[1] = 1.0 * (-1/12)
    assert abs(r["daily_long_payoff"][1] - 0.10) < TOL
    assert abs(r["daily_short_payoff"][1] - (-1 / 12)) < TOL
    assert abs(r["daily_long_payoff"][1] - r["daily_short_payoff"][1] - expected_payoff_day2) < TOL


def test_episode2_weight_compounding_and_end_of_period_close():
    returns_1 = np.array([-0.20, 0.00, 0.30])
    returns_2 = np.array([0.00, 0.05, 0.02])
    r = simulate_pair_same_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert abs(r["spread"][1] - (-0.20)) < TOL
    assert len(r["trades"]) == 2

    open_ev, close_ev = r["trades"]
    assert open_ev["day"] == 1 and open_ev["direction"] == "long1_short2", \
        "spread<-soglia -> long la gamba bassa (1), short l'alta (2)"
    assert close_ev["day"] == 3 and close_ev["reason"] == "end_of_period", \
        "nessun crossing naturale in questo episodio: la chiusura DEVE essere forzata a fine periodo"

    expected_day2 = -0.05           # 1.0*0.00 - 1.0*0.05
    expected_day3 = 0.30 - 1.05 * 0.02  # 0.279, pesi compoundati sulla gamba short
    assert abs(r["daily_payoff"][0] - 0.0) < TOL
    assert abs(r["daily_payoff"][1] - expected_day2) < TOL
    assert abs(r["daily_payoff"][2] - expected_day3) < TOL

    # long leg is 1, short leg is 2. Long contribution day3 = 1.0*0.30,
    # short contribution day3 = 1.05*0.02 (weight compounded from day2).
    assert abs(r["daily_long_payoff"][1] - 0.0) < TOL
    assert abs(r["daily_short_payoff"][1] - 0.05) < TOL
    assert abs(r["daily_long_payoff"][2] - 0.30) < TOL
    assert abs(r["daily_short_payoff"][2] - 1.05 * 0.02) < TOL
    assert abs(r["total_payoff"] - 0.229) < TOL


def test_no_signal_no_trade():
    """Spread sotto soglia per tutto il periodo: zero trade, zero payoff."""
    returns_1 = np.array([0.01, -0.01, 0.005])
    returns_2 = np.array([0.00, 0.00, 0.00])
    r = simulate_pair_same_day(returns_1, returns_2, sigma=0.05, k=2.0)
    assert len(r["trades"]) == 0
    assert abs(r["total_payoff"]) < TOL


def test_wait_one_day_confirmed_signal_opens_next_day():
    """
    Segnale al giorno1 (spread=0.20>0.10), CONFERMATO al giorno2 (spread
    ancora 0.20, invariato) -> apertura al giorno2 (non al giorno1), nessun
    payoff il giorno di apertura. Giorno3: swing ampio -> crossing e chiusura,
    payoff calcolato a mano con pesi ancora 1.0 (un solo giorno trascorso
    dall'apertura).
      returns_1 = [+0.20, 0.00, -0.20], returns_2 = [0.00, 0.00, +0.30]
      Giorno1: P1=1.20, P2=1.00, spread=0.20 -> SEGNALE (non apre subito).
      Giorno2: P1=1.20, P2=1.00, spread=0.20 ancora > 0.10 -> CONFERMA, apre
               long2/short1 al giorno2, pesi=1.
      Giorno3: P1=1.20*0.80=0.96, P2=1.00*1.30=1.30, spread=-0.34 (segno
               cambiato -> crossing). payoff = 1*r2 - 1*r1 = 0.30-(-0.20)=0.50.
    """
    returns_1 = np.array([0.20, 0.00, -0.20])
    returns_2 = np.array([0.00, 0.00, 0.30])
    r = simulate_pair_wait_one_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert len(r["trades"]) == 2, f"attesi 2 eventi (open+close), trovati {r['trades']}"
    open_ev, close_ev = r["trades"]
    assert open_ev["event"] == "open" and open_ev["day"] == 2, \
        "apertura al giorno2 (giorno dopo il segnale), non al giorno1"
    assert open_ev["signal_day"] == 1, "il segnale va tracciato al giorno in cui e' stato osservato"
    assert open_ev["direction"] == "long2_short1"
    assert close_ev["event"] == "close" and close_ev["day"] == 3 and close_ev["reason"] == "crossing"

    assert abs(r["daily_payoff"][0] - 0.0) < TOL, "nessun payoff nei giorni pre-apertura"
    assert abs(r["daily_payoff"][1] - 0.0) < TOL, "nessun payoff il giorno di apertura"
    assert abs(r["daily_payoff"][2] - 0.50) < TOL
    assert abs(r["total_payoff"] - 0.50) < TOL


def test_same_day_delisting_forces_close_at_last_valid_price():
    """
    Delisting a meta' periodo (PROTOCOL.md §1.4/§2.2), posizione APERTA:
      returns_1 = [+0.20, +0.10, NaN, 0.00], returns_2 = [0.00, -0.05, 0.00, 0.00]
      sigma=0.05 (soglia=0.10).
      Giorno1: P1=1.20, P2=1.00, spread=0.20>0.10 -> apre long2/short1.
      Giorno2: r_long=r2=-0.05, r_short=r1=0.10.
               payoff = 1*(-0.05) - 1*(0.10) = -0.15.
               P1=1.20*1.10=1.32, P2=1.00*0.95=0.95, spread=0.37 (nessun
               crossing naturale: stesso segno di prima).
      Giorno3: returns_1[2]=NaN -> il titolo 1 delista qui. L'ultimo giorno
               con prezzi validi per ENTRAMBE le gambe e' il giorno2:
               la posizione (ancora aperta) si chiude FORZATAMENTE li',
               reason="delisting", NON al giorno3 ne' a fine periodo (n=4).
               Il giorno4 (dopo il delisting) non viene proprio processato
               (il loop si ferma al giorno2): daily_payoff[2]=daily_payoff[3]=0.
      Totale atteso = 0 (giorno1, apertura) + (-0.15) (giorno2) = -0.15.
    """
    returns_1 = np.array([0.20, 0.10, np.nan, 0.00])
    returns_2 = np.array([0.00, -0.05, 0.00, 0.00])
    r = simulate_pair_same_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert len(r["trades"]) == 2
    open_ev, close_ev = r["trades"]
    assert open_ev["day"] == 1 and open_ev["direction"] == "long2_short1"
    assert close_ev["event"] == "close" and close_ev["day"] == 2, \
        "chiusura forzata al giorno2 = ultimo prezzo valido, non al giorno3 (NaN) ne' a fine periodo"
    assert close_ev["reason"] == "delisting"

    assert abs(r["daily_payoff"][0] - 0.0) < TOL
    assert abs(r["daily_payoff"][1] - (-0.15)) < TOL
    assert abs(r["daily_payoff"][2] - 0.0) < TOL, "giorno3 (post-delisting) non processato: payoff 0"
    assert abs(r["daily_payoff"][3] - 0.0) < TOL, "giorno4 (post-delisting) non processato: payoff 0"
    assert abs(r["total_payoff"] - (-0.15)) < TOL


def test_same_day_delisting_without_open_position_just_stops():
    """
    Delisting SENZA posizione aperta (spread sempre sotto soglia prima del
    NaN): la coppia smette semplicemente di generare segnali da quel punto,
    nessun trade, nessun payoff, nessun crash.
      returns_1 = [0.01, NaN, 0.01], returns_2 = [0.00, 0.00, 0.00], sigma=0.05.
    """
    returns_1 = np.array([0.01, np.nan, 0.01])
    returns_2 = np.array([0.00, 0.00, 0.00])
    r = simulate_pair_same_day(returns_1, returns_2, sigma=0.05, k=2.0)
    assert r["trades"] == []
    assert abs(r["total_payoff"]) < TOL


def test_wait_one_day_delisting_forces_close_at_last_valid_price():
    """
    Stesso scenario di test_same_day_delisting_forces_close_at_last_valid_price
    ma con esecuzione wait-one-day (conferma un giorno dopo il segnale):
      returns_1 = [+0.20, 0.00, +0.10, NaN], returns_2 = [0.00, 0.00, -0.05, 0.00]
      sigma=0.05 (soglia=0.10).
      Giorno1: spread=0.20>0.10 -> SEGNALE (non apre).
      Giorno2: spread invariato (0.20, ancora >0.10) -> CONFERMA, apre
               long2/short1 al giorno2 (nessun payoff, giorno di apertura).
      Giorno3: r_long=r2=-0.05, r_short=r1=0.10 -> payoff = -0.05-0.10=-0.15.
               spread=0.37, nessun crossing naturale.
      Giorno4: returns_1[3]=NaN -> delisting. Ultimo giorno valido = giorno3:
               chiusura forzata li' (reason="delisting"), non a fine periodo.
      Totale atteso = -0.15 (solo giorno3).
    """
    returns_1 = np.array([0.20, 0.00, 0.10, np.nan])
    returns_2 = np.array([0.00, 0.00, -0.05, 0.00])
    r = simulate_pair_wait_one_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert len(r["trades"]) == 2
    open_ev, close_ev = r["trades"]
    assert open_ev["day"] == 2 and open_ev["signal_day"] == 1
    assert close_ev["day"] == 3 and close_ev["reason"] == "delisting"

    assert abs(r["daily_payoff"][1] - 0.0) < TOL, "giorno2 = apertura, nessun payoff"
    assert abs(r["daily_payoff"][2] - (-0.15)) < TOL
    assert abs(r["daily_payoff"][3] - 0.0) < TOL, "giorno4 (post-delisting) non processato"
    assert abs(r["total_payoff"] - (-0.15)) < TOL


def test_wait_one_day_delisting_without_open_position_just_stops():
    """Delisting SENZA posizione aperta ne' segnale pending irrisolto:
    nessun trade, nessun payoff, nessun crash."""
    returns_1 = np.array([0.01, np.nan, 0.01])
    returns_2 = np.array([0.00, 0.00, 0.00])
    r = simulate_pair_wait_one_day(returns_1, returns_2, sigma=0.05, k=2.0)
    assert r["trades"] == []
    assert abs(r["total_payoff"]) < TOL


def test_same_day_open_on_last_valid_day_closes_same_iteration():
    """
    Caso limite scoperto in audit (non specifico al delisting: si presenta
    anche a fine periodo ordinaria): il segnale scatta SOLO all'ultimo
    giorno valido del periodo (nessun giorno successivo per marcare a
    mercato). Senza chiusura forzata nella stessa iterazione, il trade log
    conterrebbe un evento "open" MAI abbinato a un "close" (is_open
    resterebbe True oltre il return della funzione) - un trade fantasma che
    corromperebbe qualunque diagnostica a valle su round-trip/durata
    (PROTOCOL.md §2.3). Qui n=2, segnale solo al giorno2:
      returns_1 = [0.00, 0.20], returns_2 = [0.00, 0.00], sigma=0.05.
      Giorno1: spread=0, nessun segnale.
      Giorno2: P1=1.20, P2=1.00, spread=0.20>0.10 -> apre long2/short1 E
               chiude nella STESSA iterazione (fine periodo, durata zero).
    Atteso: esattamente 2 eventi (open+close, entrambi al giorno2), payoff
    totale zero (nessun giorno successivo per marcare a mercato).
    """
    returns_1 = np.array([0.00, 0.20])
    returns_2 = np.array([0.00, 0.00])
    r = simulate_pair_same_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert len(r["trades"]) == 2, f"open E close attesi, non un trade fantasma: {r['trades']}"
    open_ev, close_ev = r["trades"]
    assert open_ev["event"] == "open" and open_ev["day"] == 2
    assert close_ev["event"] == "close" and close_ev["day"] == 2
    assert close_ev["reason"] == "end_of_period"
    assert abs(r["total_payoff"]) < TOL


def test_wait_one_day_open_on_last_valid_day_closes_same_iteration():
    """
    Stesso caso limite di test_same_day_open_on_last_valid_day_closes_same_iteration
    ma con esecuzione wait-one-day: la CONFERMA (non solo il segnale) cade
    sull'ultimo giorno valido.
      returns_1 = [0.00, 0.20, 0.20], returns_2 = [0.00, 0.00, 0.00], sigma=0.05.
      Giorno1: nessun segnale. Giorno2: spread=0.20>0.10 -> SEGNALE (pending).
      Giorno3 (ultimo giorno, n=3): spread ancora 0.20+0.20=0.44>0.10 ->
      CONFERMA, apre E chiude nella stessa iterazione (durata zero).
    """
    returns_1 = np.array([0.00, 0.20, 0.20])
    returns_2 = np.array([0.00, 0.00, 0.00])
    r = simulate_pair_wait_one_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert len(r["trades"]) == 2, f"open E close attesi, non un trade fantasma: {r['trades']}"
    open_ev, close_ev = r["trades"]
    assert open_ev["event"] == "open" and open_ev["day"] == 3 and open_ev["signal_day"] == 2
    assert close_ev["event"] == "close" and close_ev["day"] == 3
    assert close_ev["reason"] == "end_of_period"
    assert abs(r["total_payoff"]) < TOL


def test_wait_one_day_missed_opportunity_reverts_before_execution():
    """
    Segnale al giorno1 (spread=0.20>0.10), ma al giorno2 lo spread e' gia'
    rientrato sotto soglia (0.02 < 0.10, stesso segno) prima dell'esecuzione
    -> il trade NON si apre: occasione persa, evento "missed", zero payoff,
    nessuna posizione mai aperta.
      returns_1 = [+0.20, -0.15], returns_2 = [0.00, 0.00]
      Giorno1: P1=1.20, P2=1.00, spread=0.20 -> SEGNALE.
      Giorno2: P1=1.20*0.85=1.02, P2=1.00, spread=0.02 < soglia -> MISSED.
    """
    returns_1 = np.array([0.20, -0.15])
    returns_2 = np.array([0.00, 0.00])
    r = simulate_pair_wait_one_day(returns_1, returns_2, sigma=0.05, k=2.0)

    assert len(r["trades"]) == 1, f"atteso solo l'evento missed, trovati {r['trades']}"
    missed_ev = r["trades"][0]
    assert missed_ev["event"] == "missed" and missed_ev["day"] == 2
    assert missed_ev["signal_day"] == 1
    assert missed_ev["direction"] == "long2_short1"
    assert abs(r["total_payoff"]) < TOL, "occasione persa: nessuna posizione, nessun payoff"


if __name__ == "__main__":
    test_episode1_crossing_close()
    test_episode2_weight_compounding_and_end_of_period_close()
    test_no_signal_no_trade()
    test_wait_one_day_confirmed_signal_opens_next_day()
    test_wait_one_day_missed_opportunity_reverts_before_execution()
    test_same_day_delisting_forces_close_at_last_valid_price()
    test_same_day_delisting_without_open_position_just_stops()
    test_wait_one_day_delisting_forces_close_at_last_valid_price()
    test_wait_one_day_delisting_without_open_position_just_stops()
    test_same_day_open_on_last_valid_day_closes_same_iteration()
    test_wait_one_day_open_on_last_valid_day_closes_same_iteration()
    print("Tutti i test sintetici PASSATI.")
    print("  Episodio 1 (crossing close):        total_payoff =", 11/60)
    print("  Episodio 2 (end-of-period, compound): total_payoff =", 0.229)
    print("  Wait-one-day confermato (open+close): total_payoff =", 0.50)
    print("  Wait-one-day occasione persa:          total_payoff =", 0.0)
