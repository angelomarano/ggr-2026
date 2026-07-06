"""
test_factors.py — assemble_factors (join of the raw Ken French datasets,
percent -> decimal conversion) and load_factors (cache read + range
filter), WITHOUT network access: the "raw" datasets are built by hand with
known values, the cache is a fictitious parquet written to tmp_path.
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
    3 datasets with slightly different monthly coverage (Ken French
    sometimes publishes Momentum/ST-Reversal a month behind the 3 base
    factors):
      FF3:    2020-01, 2020-02, 2020-03   (Mkt-RF, SMB, HML, RF in %)
      Mom:    2020-01, 2020-02                       (2020-03 missing)
      ST_Rev: 2020-01, 2020-02, 2020-03
    Expected: 2020-03 EXCLUDED (inner join: missing in Mom), 2020-01/02
    present, all values divided by 100, index = start-of-month Timestamp.
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
        "2020-03 must be excluded: missing from the Momentum dataset (inner join)"
    assert list(out.columns) == ["Mkt-RF", "SMB", "HML", "RF", "Mom", "ST_Rev"]

    row0 = out.loc["2020-01-01"]
    assert abs(row0["Mkt-RF"] - 0.01) < TOL
    assert abs(row0["SMB"] - 0.005) < TOL
    assert abs(row0["HML"] - (-0.01)) < TOL
    assert abs(row0["RF"] - 0.001) < TOL
    assert abs(row0["Mom"] - 0.02) < TOL
    assert abs(row0["ST_Rev"] - 0.003) < TOL


def test_load_factors_range_filter(tmp_path, monkeypatch):
    """load_factors reads the cache and filters [start, end] by month; a
    fictitious cache (4 months) is used here to avoid network access in the test."""
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
    """Cache not yet downloaded -> explicit error, not a generic crash nor a
    silent empty DataFrame."""
    monkeypatch.setattr(factors_mod, "CACHE_PATH", tmp_path / "does_not_exist.parquet")
    with pytest.raises(FileNotFoundError):
        load_factors()


if __name__ == "__main__":
    test_assemble_factors_percent_to_decimal_and_inner_join()
    print("test_factors: pure tests OK (tests with tmp_path/monkeypatch require pytest)")
