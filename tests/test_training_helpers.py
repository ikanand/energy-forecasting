from types import SimpleNamespace as V

import pytest

pytest.importorskip("pyspark")  # training.py imports pyspark; skipped in CI where Spark isn't installed
from energy_forecasting.training import version_for_run  # noqa: E402


def test_picks_the_version_from_this_run():
    versions = [V(version="1", run_id="a"), V(version="3", run_id="b"), V(version="2", run_id="c")]
    assert version_for_run(versions, "c") == "2"


def test_falls_back_to_newest_when_run_id_missing():
    versions = [V(version="1", run_id=None), V(version="10", run_id=None), V(version="2", run_id=None)]
    assert version_for_run(versions, "zzz") == "10"  # numeric, not string, ordering
