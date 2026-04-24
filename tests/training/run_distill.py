"""CLI wrapper for knowledge distillation (Method C).

Teacher: artifacts/classifier/d5/modernbert/best.pt
Student: MiniLM-L6 frozen-encoder + MLP heads (same shape as the deployed
         4.9 MB checkpoint; only the heads are trained).

Typical use:
  PYTHONPATH=src python3 -m tests.training.run_distill \\
      --teacher artifacts/classifier/d5/modernbert \\
      --weak    data/weak_labels/pool.jsonl \\
      --train   data/gold/train.jsonl \\
      --val     data/gold/val.jsonl \\
      --out     artifacts/classifier/distilled/minilm-l6-frozen-mlp
"""
from __future__ import annotations

import argparse
import json
import sys

from .distill import DistillConfig, run_distillation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", dest="teacher_ckpt_dir", required=True,
                    help="dir containing teacher best.pt (e.g. d5/modernbert)")
    ap.add_argument("--student", dest="student_key", default="minilm-l6-frozen-mlp",
                    help="must be a freeze_encoder=True variant")
    ap.add_argument("--weak",   dest="weak_jsonl",       default="data/weak_labels/pool.jsonl")
    ap.add_argument("--train",  dest="gold_train_jsonl", default="data/gold/train.jsonl")
    ap.add_argument("--val",    dest="val_jsonl",        default="data/gold/val.jsonl")
    ap.add_argument("--out",    dest="output_dir",       required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--alpha",  type=float, default=0.5,
                    help="weight on hard BCE; 1-alpha on (T**2)*soft BCE")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cfg = DistillConfig(
        teacher_ckpt_dir=args.teacher_ckpt_dir,
        student_key=args.student_key,
        weak_jsonl=args.weak_jsonl,
        gold_train_jsonl=args.gold_train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        temperature=args.temperature,
        seed=args.seed,
    )
    summary = run_distillation(cfg)
    print("\n================ DISTILL SUMMARY =================")
    print(f"teacher  : {summary['teacher_base_model']}")
    print(f"student  : {summary['student_base_model']}")
    print(f"best F1  : {summary['best_macro_f1']:.4f}  (epoch {summary['best_epoch']})")
    print(f"alpha    : {summary['alpha']}")
    print(f"T        : {summary['temperature']}")
    print(f"n_train  : {summary['n_train']}  "
          f"(weak={summary['n_train_weak']} gold={summary['n_train_gold']})")
    print(f"elapsed  : {summary['total_time_sec']:.1f}s")

    json.dump(summary, sys.stdout, indent=2, default=str)
    print()


if __name__ == "__main__":
    main()
