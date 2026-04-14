import json
import subprocess
import sys


def test_cli_ask_routes_secure_for_credential():
    result = subprocess.run(
        [sys.executable, "-m", "privategate.cli", "ask", "rotate sk-abcdefghij1234567890 please"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["routing"] == "secure"


def test_cli_ask_routes_standard_for_clean_query():
    result = subprocess.run(
        [sys.executable, "-m", "privategate.cli", "ask", "what is the capital of france"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["routing"] == "standard"
