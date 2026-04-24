"""Stratified sampling from raw + synthetic JSONLs into the labeling pools.

Produces two output files:
  data/sampled/weak_pool.jsonl  — 50k unlabeled prompts (weak-supervision stage 1)
  data/sampled/gold_pool.jsonl  —  5k unlabeled prompts (gold stage 2)

Composition (default, tuned for the 58.9k-row raw corpus + ~4k synthetic):
  gold_pool.jsonl (~4.6k — sampled FIRST so the weak pool can't leak into it)
    - 2k   from oasst1
    - 1k   from ai4privacy
    - 100  from jailbreakbench (half of the 200 available)
    - 1.5k from synthetic/ (~200–250 per rare class × 7 classes)
  weak_pool.jsonl (~50k — sampled SECOND from whatever's left per source)
    - all remaining oasst1     (≈13.2k after gold)
    - 35k of remaining ai4privacy (absorbs the oasst1/jbb shortfall)
    - remaining jailbreakbench (≈100)
    - remaining synthetic      (≈2.7k at --per-class 600)

Key invariant: no item appears in both pools (sampled by id).

Sampling is deterministic given --seed. Run after download.py and
synthesize_rare.py.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(json.loads(line))
    return out


def _write(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _sample(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n >= len(pool):
        return list(pool)
    return rng.sample(pool, n)


# Order matters: gold_pool is sampled first so weak_pool can subtract the
# already-picked IDs and avoid overlap. Each value is a soft target — if the
# source pool is smaller (oasst1, jbb), we take what's available.
RECIPE: dict[str, dict[str, int]] = {
    "gold_pool": {
        "oasst1": 2_000,
        "ai4privacy": 1_000,
        "jailbreakbench": 100,
        "synthetic": 1_500,
    },
    "weak_pool": {
        # None ⇒ take all remaining from that source.
        "oasst1": None,          # type: ignore[dict-item]
        "ai4privacy": 35_000,    # bumped from 5k to absorb oasst1/jbb shortfall
        "jailbreakbench": None,  # type: ignore[dict-item]
        "synthetic": None,       # type: ignore[dict-item]
    },
}


def _read_source(raw_dir: Path, source: str) -> list[dict]:
    if source == "synthetic":
        rows: list[dict] = []
        syn_dir = raw_dir.parent / "synthetic"
        if syn_dir.exists():
            for p in sorted(syn_dir.glob("*.jsonl")):
                rows.extend(_read(p))
        return rows
    return _read(raw_dir / source / "items.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="data/raw")
    ap.add_argument("--out", dest="out_dir", default="data/sampled")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    pools: dict[str, list[dict]] = {src: _read_source(raw_dir, src) for src in
                                    {s for rec in RECIPE.values() for s in rec}}
    taken_ids: set[str] = set()   # items already assigned to a pool

    # RECIPE dict ordering (gold_pool then weak_pool) matters: gold draws
    # first so weak can subtract used ids and avoid test-train overlap.
    for pool_name, recipe in RECIPE.items():
        picked: list[dict] = []
        for source, n in recipe.items():
            available = [it for it in pools[source] if it["id"] not in taken_ids]
            target = n if n is not None else len(available)
            got = _sample(available, target, rng)
            for it in got:
                taken_ids.add(it["id"])
            picked.extend(got)
            req_str = "all" if n is None else str(n)
            print(f"  {pool_name} <- {source}: requested {req_str}, "
                  f"got {len(got)} (available {len(available)} / pool {len(pools[source])})")
        rng.shuffle(picked)
        path = out_dir / f"{pool_name}.jsonl"
        n_out = _write(picked, path)
        print(f"[ok] {pool_name}: {n_out} rows → {path}")


if __name__ == "__main__":
    main()
