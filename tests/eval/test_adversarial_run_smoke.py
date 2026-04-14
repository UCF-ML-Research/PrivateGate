import json
import subprocess
import sys

from privategate.eval.adversarial_run import run_adversarial_table


def _row(table, baseline, attacker):
    for r in table["results"]:
        if r["baseline"] == baseline and r["attacker"] == attacker:
            return r
    raise AssertionError(f"missing row {baseline}/{attacker}")


def test_runs_full_grid():
    table = run_adversarial_table()
    pairs = {(r["baseline"], r["attacker"]) for r in table["results"]}
    expected = {
        (b, a)
        for b in ("plaintext", "full_mask", "full_abstract", "full_secure", "privategate")
        for a in ("embedding_inversion", "llm_guesser")
    }
    assert expected <= pairs


def test_plaintext_is_fully_recovered_by_inversion():
    table = run_adversarial_table()
    row = _row(table, "plaintext", "embedding_inversion")
    assert row["overall_rate"] == 1.0


def test_full_secure_leaks_nothing():
    table = run_adversarial_table()
    for attacker in ("embedding_inversion", "llm_guesser"):
        row = _row(table, "full_secure", attacker)
        assert row["overall_rate"] == 0.0


def test_privategate_credentials_not_recovered():
    table = run_adversarial_table()
    for attacker in ("embedding_inversion", "llm_guesser"):
        row = _row(table, "privategate", attacker)
        cred = row["by_category"].get("CREDENTIAL")
        if cred is not None:
            assert cred["rate"] == 0.0, f"{attacker} recovered a credential from PrivateGate"


def test_privategate_recovery_strictly_below_plaintext():
    table = run_adversarial_table()
    for attacker in ("embedding_inversion", "llm_guesser"):
        plain = _row(table, "plaintext", attacker)["overall_rate"]
        pg = _row(table, "privategate", attacker)["overall_rate"]
        assert pg < plain, f"{attacker}: PrivateGate rate {pg} not below plaintext {plain}"


def test_module_main_runs():
    result = subprocess.run(
        [sys.executable, "-m", "privategate.eval.adversarial_run"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert "results" in payload
