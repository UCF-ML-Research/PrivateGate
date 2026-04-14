import json
import subprocess
import sys

from privategate.eval.run import run_main_table


def test_run_main_table_returns_all_baselines():
    table = run_main_table()
    names = [r["name"] for r in table["results"]]
    assert {"plaintext", "full_mask", "full_abstract", "full_secure", "privategate"} <= set(names)
    assert table["n_examples"] >= 5


def test_plaintext_has_higher_exposure_than_privategate():
    table = run_main_table()
    by_name = {r["name"]: r for r in table["results"]}
    plain = by_name["plaintext"]["privacy"]["span_exposure_rate"]
    pg = by_name["privategate"]["privacy"]["span_exposure_rate"]
    assert plain >= pg
    # the seed dataset includes spans, so plaintext leakage must be > 0
    assert plain > 0.0


def test_full_secure_has_zero_exposure():
    table = run_main_table()
    by_name = {r["name"]: r for r in table["results"]}
    assert by_name["full_secure"]["privacy"]["span_exposure_rate"] == 0.0
    assert by_name["full_secure"]["efficiency"]["secure_path_fraction"] == 1.0


def test_run_module_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "privategate.eval.run"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert "results" in payload
