"""Run all four M7 ablations and dump the results as JSON."""
from __future__ import annotations

import json

from privategate.eval.ablations import abstraction, detector, policy, probe
from privategate.eval.datasets.synthetic_mixed import load_synthetic_mixed


def run_all() -> dict:
    examples = load_synthetic_mixed()
    return {
        "n_examples": len(examples),
        "ablations": [
            detector.run(examples),
            probe.run(examples),
            policy.run(examples),
            abstraction.run(examples),
        ],
    }


def main() -> int:
    print(json.dumps(run_all(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
