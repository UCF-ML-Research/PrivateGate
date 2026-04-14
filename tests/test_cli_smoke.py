import json
import subprocess
import sys


def test_cli_rewrite_smoke():
    result = subprocess.run(
        [sys.executable, "-m", "privategate.cli", "rewrite", "hello world"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert "transformed" in payload
    assert "placeholder_map" in payload
    assert payload["transformed"] == "hello world"
