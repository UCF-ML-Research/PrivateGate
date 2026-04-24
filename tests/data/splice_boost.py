"""Splice class-boost items into the gold train split (Method B).

Usage:
  PYTHONPATH=src python3 -m tests.data.splice_boost \\
      --train data/gold/train.jsonl \\
      --val   data/gold/val.jsonl \\
      --test  data/gold/test.jsonl \\
      --boost data/weak_labels/regulated_boost.jsonl \\
      --out   data/gold/train_boosted.jsonl \\
      --only-classes regulated_eu regulated_us

The boost file must be schema-valid (same shape as `gold/train.jsonl`), i.e.
Qwen-labeled rows normalized by `tests.data.auto_label`. Items whose id already
appears in any of `train/val/test` are dropped (id-disjoint invariant). When
`--only-classes` is provided, only items positive in at least one of those
classes are kept — prevents inflating unrelated classes from a mis-targeted
synthesis run.

Writes:
  --out                         (gold train + filtered boost rows, shuffled)
  --out with `.manifest.json`   (counts, per-class deltas, dropped-id summary)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .schema import REQUIRED_CATEGORIES, validate_item


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def _class_counts(rows: list[dict]) -> dict[str, int]:
    counts = {c: 0 for c in REQUIRED_CATEGORIES}
    for r in rows:
        for c in REQUIRED_CATEGORIES:
            if r.get("categories", {}).get(c, False):
                counts[c] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="gold train JSONL (source)")
    ap.add_argument("--val",   required=True, help="gold val JSONL (for id-disjoint check)")
    ap.add_argument("--test",  required=True, help="gold test JSONL (for id-disjoint check)")
    ap.add_argument("--boost", nargs="+", required=True,
                    help="one or more boost JSONLs (schema-valid, Qwen-labeled)")
    ap.add_argument("--out",   required=True, help="output boosted train JSONL")
    ap.add_argument("--only-classes", nargs="*", default=None,
                    help="keep boost rows with at least one of these classes true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--validate-boost", action="store_true",
                    help="run schema validate_item on each boost row (slower, safer)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train = _read_jsonl(Path(args.train))
    val   = _read_jsonl(Path(args.val))
    test  = _read_jsonl(Path(args.test))

    taken = {r["id"] for r in train} | {r["id"] for r in val} | {r["id"] for r in test}
    before_counts = _class_counts(train)
    only_classes = set(args.only_classes) if args.only_classes else None

    boost_rows: list[dict] = []
    dropped_dup = 0
    dropped_off_class = 0
    dropped_bad = 0
    seen_boost_ids: set[str] = set()
    for p in args.boost:
        for row in _read_jsonl(Path(p)):
            rid = row.get("id")
            if not rid:
                dropped_bad += 1
                continue
            if rid in taken or rid in seen_boost_ids:
                dropped_dup += 1
                continue
            cats = row.get("categories", {})
            if only_classes is not None and not any(cats.get(c, False) for c in only_classes):
                dropped_off_class += 1
                continue
            if args.validate_boost:
                try:
                    validate_item(row)
                except Exception as e:
                    dropped_bad += 1
                    print(f"[drop] invalid {rid}: {e}")
                    continue
            seen_boost_ids.add(rid)
            boost_rows.append(row)

    merged = list(train) + boost_rows
    rng.shuffle(merged)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    after_counts = _class_counts(merged)
    deltas = {c: after_counts[c] - before_counts[c] for c in REQUIRED_CATEGORIES}
    manifest = {
        "train_in": str(args.train),
        "boost_in": args.boost,
        "out": str(out),
        "n_train_in": len(train),
        "n_boost_in": sum(len(_read_jsonl(Path(p))) for p in args.boost),
        "n_boost_kept": len(boost_rows),
        "n_dropped_duplicate_id": dropped_dup,
        "n_dropped_off_class": dropped_off_class,
        "n_dropped_invalid": dropped_bad,
        "n_out": len(merged),
        "only_classes": sorted(only_classes) if only_classes else None,
        "class_counts_before": before_counts,
        "class_counts_after": after_counts,
        "class_deltas": deltas,
    }
    with out.with_suffix(out.suffix + ".manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[ok] {out}: {len(merged)} rows "
          f"(= {len(train)} gold-train + {len(boost_rows)} boost)")
    print(f"     dropped: dup={dropped_dup} off_class={dropped_off_class} invalid={dropped_bad}")
    print(f"     class deltas: "
          + ", ".join(f"{c}={deltas[c]:+d}" for c in REQUIRED_CATEGORIES if deltas[c]))


if __name__ == "__main__":
    main()
