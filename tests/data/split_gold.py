"""Stratified train/val/test split of the Qwen-labeled gold drafts.

Produces three deterministic JSONL files at `data/gold/{train,val,test}.jsonl`.
Stratification is over the composite key (gold_mode, source) so each split
preserves both the routing-label distribution and the source mix.

Usage:
  PYTHONPATH=src python3 -m tests.data.split_gold \\
      --in data/gold/drafts_thinking_on.jsonl \\
      --out-dir data/gold --seed 1337 \\
      --train 0.80 --val 0.10 --test 0.10
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val",   type=float, default=0.10)
    ap.add_argument("--test",  type=float, default=0.10)
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "ratios must sum to 1"

    rng = random.Random(args.seed)
    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    print(f"loaded {len(rows)} items from {in_path}")

    # Stratum = (gold_mode, source). Small strata are thrown into a catch-all.
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["gold_mode"], r["source"])].append(r)

    train: list[dict] = []
    val:   list[dict] = []
    test:  list[dict] = []
    for key, bucket in buckets.items():
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(round(n * args.train))
        n_val   = int(round(n * args.val))
        # test gets the remainder so ratios never round to zero items missing
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train:n_train + n_val])
        test.extend(bucket[n_train + n_val:])

    for split_name, split_rows in (("train", train), ("val", val), ("test", test)):
        rng.shuffle(split_rows)
        path = out_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in split_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        mode_dist = Counter(r["gold_mode"] for r in split_rows)
        src_dist  = Counter(r["source"]    for r in split_rows)
        print(f"[ok] {split_name}: {len(split_rows)} rows → {path}")
        print(f"      by mode  : {dict(mode_dist.most_common())}")
        print(f"      by source: {dict(src_dist.most_common())}")


if __name__ == "__main__":
    main()
