"""CLI wrapper for the D5 ablation driver.

Typical use:
  PYTHONPATH=src python3 -m tests.training.run_ablation \\
      --train data/gold/train.jsonl \\
      --val   data/gold/val.jsonl \\
      --out   artifacts/classifier/d5 \\
      --epochs 4
"""
from __future__ import annotations

import argparse
import json
import sys

from .ablation import ABLATION_KEYS, run_d5_ablation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", dest="train_jsonl", required=True)
    ap.add_argument("--val",   dest="val_jsonl",   required=True)
    ap.add_argument("--out",   dest="output_root", required=True)
    ap.add_argument("--epochs", type=int, default=4,
                    help="epochs for fine-tuned variants; frozen variant defaults to 15")
    ap.add_argument("--stage", default="gold")
    ap.add_argument("--keys", nargs="+", default=None,
                    help=f"subset of {ABLATION_KEYS}; default = all")
    args = ap.parse_args()

    results = run_d5_ablation(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_root=args.output_root,
        epochs=args.epochs,
        stage=args.stage,
        keys=args.keys,
    )

    # Pretty leaderboard at the tail of stdout — easy to skim in SLURM logs.
    board = sorted(
        (
            (k, r.get("best_macro_f1", -1),
             r.get("best_epoch", -1), r.get("total_time_sec", -1))
            for k, r in results.items()
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\n================ D5 LEADERBOARD =================")
    print(f"{'base_model':26s} {'macroF1':>8s} {'epoch':>6s} {'time_s':>8s}")
    for k, f1, ep, t in board:
        print(f"{k:26s} {f1:8.4f} {ep:6d} {t:8.1f}")

    json.dump({"leaderboard": board}, sys.stdout, indent=2, default=str)
    print()


if __name__ == "__main__":
    main()
