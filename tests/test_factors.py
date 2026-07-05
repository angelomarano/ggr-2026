"""
test_factors.py — assemble_factors (join dei dataset Ken French grezzi,
conversione percentuale -> decimale) e load_factors (lettura cache + filtro
range), SENZA accesso di rete: i "raw" dataset sono costruiti a mano con
valori noti, la cache e' un parquet fittizio scritto in tmp_path.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import pytest

import data.factors as factors_mod
from data.factors import assemble_factors, load_factors

TOL = 1e-9


def _period_df(values: dict, months: list[str]) -> pd.DataFrame:
    idx = pd.PeriodIndex(months, freq="M")
    return pd.DataFrame(values, index=idx)


def test_assemble_factors_percent_to_decimal_and_inner_join():
    """
    3 dataset con copertura mensile leggermente diversa (Ken French a volte
    pubblica Momentum/ST-Reversal con un mese di ritardo rispetto ai 3
    fattori base):
      FF3:    2020-01, 2020-02, 2020-03   (Mkt-RF, SMB, HML, RF in %)
      Mom:    2020-01, 2020-02                       (manca 2020-03)
      ST_Rev: 2020-01, 2020-02, 2020-03
    Atteso: 2020-03 ESCLUSO (inner join: manca in Mom), 2020-01/02 presenti,
    tutti i valori divisi per 100, indice = Timestamp di inizio mese.
    """
    ff3 = _period_df(
        {"Mkt-RF": [1.0, -2.0, 3.0], "SMB": [0.5, 0.0, -0.5],
         "HML": [-1.0, 1.0, 0.0], "RF": [0.1, 0.1, 0.12]},
        ["2020-01", "2020-02", "2020-03"],
    )
    mom = _period_df({"Mom": [2.0, -1.0]}, ["2020-01", "2020-02"])
    strev = _period_df({"ST_Rev": [0.3, -0.3, 1.2]}, ["2020-01", "2020-02", "2020-03"])

    raw = {
        "F-F_Research_Data_Factors": ff3,
        "F-F_Momentum_Factor": mom,
        "F-F_ST_Reversal_Factor": strev,
    }
    out = assemble_factors(raw)

    assert list(out.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")], \
        "2020-03 va escluso: manca nel dataset Momentum (inner join)"
    assert list(out.columns) == ["Mkt-RF", "SMB", "HML", "RF", "Mom", "ST_Rev"]

    row0 = out.loc["2020-01-01"]
    assert abs(row0["Mkt-RF"] - 0.01) < TOL
    assert abs(row0["SMB"] - 0.005) < TOL
    assert abs(row0["HML"] - (-0.01)) < TOL
    assert abs(row0["RF"] - 0.001) < TOL
    assert abs(row0["Mom"] - 0.02) < TOL
    assert abs(row0["ST_Rev"] - 0.003) < TOL


def test_load_factors_range_filter(tmp_path, monkeypatch):
    """load_factors legge la cache e filtra [start, end] per mese; qui si usa
    una cache fittizia (4 mesi) per evitare accesso di rete nel test."""
    cache_file = tmp_path / "factors_test.parquet"
    idx = pd.DatetimeIndex(
        ["2019-11-01", "2019-12-01", "2020-01-01", "2020-02-01"], name="month"
    )
    df = pd.DataFrame(
        {"Mkt-RF": [0.01, 0.02, 0.03, 0.04], "SMB": [0.0] * 4, "HML": [0.0] * 4,
         "RF": [0.001] * 4, "Mom": [0.0] * 4, "ST_Rev": [0.0] * 4},
        index=idx,
    )
    df.to_parquet(cache_file)
    monkeypatch.setattr(factors_mod, "CACHE_PATH", cache_file)

    out = load_factors(start="2019-12-01", end="2020-01-31")
    assert list(out.index) == [pd.Timestamp("2019-12-01"), pd.Timestamp("2020-01-01")]


def test_load_factors_missing_cache_raises(tmp_path, monkeypatch):
    """Cache non ancora scaricata -> errore esplicito, non un crash generico
    ne' un DataFrame vuoto silenzioso."""
    monkeypatch.setattr(factors_mod, "CACHE_PATH", tmp_path / "does_not_exist.parquet")
    with pytest.raises(FileNotFoundError):
        load_factors()


if __name__ == "__main__":
    test_assemble_factors_percent_to_decimal_and_inner_join()
    print("test_factors: test puri OK (i test con tmp_path/monkeypatch richiedono pytest)")
