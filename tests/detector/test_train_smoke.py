import subprocess
import sys


def test_train_script_smoke():
    result = subprocess.run(
        [sys.executable, "scripts/train_ml_detector.py", "--smoke"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "smoke ok" in result.stdout
