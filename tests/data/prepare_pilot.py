"""Deterministically sample a 50-item diverse pilot set from raw sources.

Stratification (default recipe):
  - 15 from oasst1          (realistic prompts, mostly benign -> PLAINTEXT)
  - 20 from ai4privacy      (synthetic PII -> LDP, some HE)
  - 15 from jailbreakbench  (mix of harmful/benign -> ABSTAIN / PLAINTEXT)

Seeded for reproducibility. Output JSONL matches the `{id, prompt, source,
source_meta}` schema produced by `download.py`, so the same Labeler code
can consume it.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

RECIPE = {
    "oasst1": 15,
    "ai4privacy": 20,
    "jailbreakbench": 15,
}


def _read(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing source: {path} (run tests.data.download first)")
    items: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                items.append(json.loads(line))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/sampled/pilot_50.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = Path(args.raw)
    picked: list[dict] = []
    for src, n in RECIPE.items():
        pool = _read(raw / src / "items.jsonl")
        got = rng.sample(pool, min(n, len(pool)))
        picked.extend(got)
        print(f"  {src}: sampled {len(got)} (pool {len(pool)})")
    rng.shuffle(picked)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(picked)} items -> {out}")


if __name__ == "__main__":
    main()
