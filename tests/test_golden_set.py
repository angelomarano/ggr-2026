"""
test_golden_set.py -- guards build_golden_set()'s parametrization refactor:
calling it with no arguments must still default to exactly the replication
window and output location it used to be hardcoded to.

Note: this only checks the defaults are wired correctly (fast, hermetic,
no real price data). The refactor's actual behavior-preservation claim (same
315-ticker output for the replication window) was verified separately by
re-running build_golden_set() against the real price cache and diffing
byte-for-byte against the pre-refactor results/replication/golden_set.csv
(identical) -- that check needs the project's .venv (real cached parquet
files aren't readable under the system pyarrow this suite runs under) so it
isn't part of the automated suite, consistent with the rest of data/*.py's
IO scripts not being unit-tested here.
"""
import sys
sys.path.insert(0, ".")

import inspect
from pathlib import Path

import config
from data.golden_set import build_golden_set


def test_build_golden_set_defaults_match_replication_window():
    sig = inspect.signature(build_golden_set)
    assert sig.parameters["trading_start_first"].default == config.REPLICATION_TRADING_START_FIRST
    assert sig.parameters["trading_start_last"].default == config.REPLICATION_TRADING_START_LAST
    assert sig.parameters["out_dir"].default == Path("results/replication")
    assert sig.parameters["out_name"].default == "golden_set"


if __name__ == "__main__":
    test_build_golden_set_defaults_match_replication_window()
    print("test_golden_set: all tests PASSED.")
