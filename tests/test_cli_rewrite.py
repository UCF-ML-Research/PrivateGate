import json
import subprocess
import sys


def test_cli_rewrite_strips_ssn():
    result = subprocess.run(
        [sys.executable, "-m", "privategate.cli", "rewrite", "ssn 123-45-6789"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert "123-45-6789" not in payload["transformed"]
    assert any("123-45-6789" in v for v in payload["placeholder_map"].values())
