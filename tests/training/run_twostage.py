"""CLI wrapper for two-stage weak→gold training (Method A).

Typical use:
  PYTHONPATH=src python3 -m tests.training.run_twostage \\
      --weak  data/weak_labels/pool.jsonl \\
      --train data/gold/train.jsonl \\
      --val   data/gold/val.jsonl \\
      --out   artifacts/classifier/d5_twostage
"""
from __future__ import annotations

import argparse
import json
import sys

from .ablation import ABLATION_KEYS
from .twostage import run_twostage


def _parse_kv(items):
    out: dict[str, int] = {}
    for kv in items or []:
        if "=" not in kv:
            raise SystemExit(f"expected key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        out[k.strip()] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weak",  dest="weak_jsonl",       required=True,
                    help="weak-label pool JSONL (Qwen-labeled)")
    ap.add_argument("--train", dest="gold_train_jsonl", required=True,
                    help="gold train split JSONL")
    ap.add_argument("--val",   dest="val_jsonl",        required=True,
                    help="gold val split JSONL")
    ap.add_argument("--out",   dest="output_root",      required=True)
    ap.add_argument("--keys", nargs="+", default=None,
                    help=f"subset of {ABLATION_KEYS}; default = all")
    ap.add_argument("--epochs-weak", nargs="*", default=None,
                    help="per-variant override, e.g. modernbert=3 minilm-l6-frozen-mlp=5")
    ap.add_argument("--epochs-gold", nargs="*", default=None,
                    help="per-variant override, e.g. modernbert=6")
    args = ap.parse_args()

    results = run_twostage(
        weak_jsonl=args.weak_jsonl,
        gold_train_jsonl=args.gold_train_jsonl,
        val_jsonl=args.val_jsonl,
        output_root=args.output_root,
        keys=args.keys,
        epochs_weak_overrides=_parse_kv(args.epochs_weak),
        epochs_gold_overrides=_parse_kv(args.epochs_gold),
    )

    board = sorted(
        (
            (k,
             r.get("best_macro_f1_gold", -1),
             r.get("best_macro_f1_weak", -1))
            for k, r in results.items()
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\n============ TWO-STAGE LEADERBOARD (val macroF1) =============")
    print(f"{'base_model':26s} {'gold':>8s} {'weak':>8s}")
    for k, gf1, wf1 in board:
        print(f"{k:26s} {gf1:8.4f} {wf1:8.4f}")

    json.dump({"leaderboard": board}, sys.stdout, indent=2, default=str)
    print()


if __name__ == "__main__":
    main()
