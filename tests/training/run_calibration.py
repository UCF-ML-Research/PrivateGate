"""CLI wrapper: calibrate every trained checkpoint under an output root.

Writes `<ckpt_dir>/calibration.json` for each variant and prints a summary
table contrasting uncalibrated vs calibrated val metrics (and test metrics
when the test split is supplied).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibrate import calibrate_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="directory containing one sub-dir per variant, each "
                         "with best.pt + tokenizer/")
    ap.add_argument("--val",  dest="val_jsonl",  required=True)
    ap.add_argument("--test", dest="test_jsonl", default=None)
    ap.add_argument("--device", default=None,
                    help="'cuda' or 'cpu' (auto-detect if omitted)")
    args = ap.parse_args()

    root = Path(args.root)
    variants = sorted(p for p in root.iterdir() if (p / "best.pt").is_file())
    if not variants:
        raise SystemExit(f"no checkpoints found under {root}")

    print(f"[calibrate] {len(variants)} variants under {root}")
    print(f"[calibrate] val = {args.val_jsonl}")
    print(f"[calibrate] test = {args.test_jsonl or '<skipped>'}")

    rows: list[dict] = []
    for vdir in variants:
        name = vdir.name
        print(f"\n===== calibrating: {name} =====", flush=True)
        res = calibrate_checkpoint(
            ckpt_dir=vdir,
            val_jsonl=args.val_jsonl,
            test_jsonl=args.test_jsonl,
            device=args.device,
        )
        rows.append({
            "variant": name,
            "val_uncal_f1":  res["metrics_uncalibrated_val"]["macro_f1"],
            "val_cal_f1":    res["metrics_calibrated_val"]["macro_f1"],
            "val_uncal_ece": res["metrics_uncalibrated_val"]["macro_ece"],
            "val_cal_ece":   res["metrics_calibrated_val"]["macro_ece"],
            "test_uncal_f1": res.get("metrics_uncalibrated_test", {}).get("macro_f1"),
            "test_cal_f1":   res.get("metrics_calibrated_test", {}).get("macro_f1"),
            "test_uncal_ece": res.get("metrics_uncalibrated_test", {}).get("macro_ece"),
            "test_cal_ece":   res.get("metrics_calibrated_test", {}).get("macro_ece"),
        })

    rows.sort(key=lambda r: (r["test_cal_f1"] or r["val_cal_f1"]), reverse=True)

    print("\n================ CALIBRATION SUMMARY =================")
    print(f"{'variant':26s} {'val F1 raw':>11s} {'val F1 cal':>11s} "
          f"{'val ECE raw':>12s} {'val ECE cal':>12s} "
          f"{'test F1 cal':>12s} {'test ECE cal':>13s}")
    for r in rows:
        print(f"{r['variant']:26s} "
              f"{r['val_uncal_f1']:>11.4f} {r['val_cal_f1']:>11.4f} "
              f"{r['val_uncal_ece']:>12.4f} {r['val_cal_ece']:>12.4f} "
              f"{(r['test_cal_f1'] or 0):>12.4f} {(r['test_cal_ece'] or 0):>13.4f}")

    with (root / "calibration_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
