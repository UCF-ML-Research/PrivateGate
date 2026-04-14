import json
import subprocess
import sys

from privategate.eval.ablations.run import run_all


def test_run_all_returns_four_ablations():
    out = run_all()
    names = [a["ablation"] for a in out["ablations"]]
    assert names == ["detector", "semantic_probe", "policy_table", "abstraction_level"]


def test_main_module_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "privategate.eval.ablations.run"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert "ablations" in payload
