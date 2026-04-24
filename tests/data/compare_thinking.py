"""Compare two labeled JSONLs on the same items (e.g. thinking-off vs thinking-on).

Reports:
  - Overall and per-category agreement
  - gold_mode confusion matrix
  - Up to K examples of disagreement (for qualitative inspection)
  - Mean severity per category on each side
  - Abstain-rate diff

Usage:
  python -m tests.data.compare_thinking \\
      --a data/weak_labels/pilot_50_thinking_off.jsonl \\
      --b data/weak_labels/pilot_50_thinking_on.jsonl \\
      --label-a off --label-b on \\
      --show 8
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CATS = [
    "pii", "phi", "pci", "secret", "ip_confidential",
    "regulated_eu", "regulated_us", "injection",
]


def _load(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                row = json.loads(line)
                out[row["id"]] = row
    return out


def _fmt_pct(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{num}/{den} = {num / den:.1%}"


def _mode_confusion(shared, a, b):
    modes = ["plaintext", "ldp", "he", "abstain"]
    conf = Counter((a[i]["gold_mode"], b[i]["gold_mode"]) for i in shared)
    return modes, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--show", type=int, default=5, help="max disagreement examples to show")
    args = ap.parse_args()

    a = _load(Path(args.a))
    b = _load(Path(args.b))
    shared = sorted(set(a) & set(b))
    missing_a = sorted(set(b) - set(a))
    missing_b = sorted(set(a) - set(b))
    print(f"n({args.label_a}) = {len(a)}")
    print(f"n({args.label_b}) = {len(b)}")
    print(f"n(shared)         = {len(shared)}")
    if missing_a:
        print(f"only in {args.label_b}: {len(missing_a)} (first ids: {missing_a[:3]})")
    if missing_b:
        print(f"only in {args.label_a}: {len(missing_b)} (first ids: {missing_b[:3]})")
    if not shared:
        return

    # gold_mode agreement + confusion
    mode_agree = sum(1 for i in shared if a[i]["gold_mode"] == b[i]["gold_mode"])
    print(f"\ngold_mode agreement: {_fmt_pct(mode_agree, len(shared))}")
    modes, conf = _mode_confusion(shared, a, b)
    print(f"\nconfusion  rows={args.label_a}  cols={args.label_b}")
    print("           " + " ".join(f"{m:>9s}" for m in modes))
    for ma in modes:
        row = [conf.get((ma, mb), 0) for mb in modes]
        print(f"  {ma:8s} " + " ".join(f"{v:>9d}" for v in row))

    # Abstain rate per side
    ab_a = sum(1 for i in shared if a[i]["gold_mode"] == "abstain")
    ab_b = sum(1 for i in shared if b[i]["gold_mode"] == "abstain")
    print(f"\nabstain rate  {args.label_a}: {_fmt_pct(ab_a, len(shared))}   {args.label_b}: {_fmt_pct(ab_b, len(shared))}")

    # Per-category agreement
    print("\nper-category agreement (both-present or both-absent):")
    for c in CATS:
        agree = sum(1 for i in shared if a[i]["categories"].get(c, False) == b[i]["categories"].get(c, False))
        a_pos = sum(1 for i in shared if a[i]["categories"].get(c, False))
        b_pos = sum(1 for i in shared if b[i]["categories"].get(c, False))
        print(f"  {c:18s} agree={_fmt_pct(agree, len(shared))}   {args.label_a}_pos={a_pos}  {args.label_b}_pos={b_pos}")

    # Mean severity per category
    def _mean_sev(side: dict[str, dict], cat: str) -> float:
        vals = [side[i]["severity"].get(cat, 0) for i in shared if side[i]["categories"].get(cat, False)]
        return sum(vals) / len(vals) if vals else 0.0
    print("\nmean severity | present-only:")
    for c in CATS:
        sa = _mean_sev(a, c)
        sb = _mean_sev(b, c)
        if sa == 0 and sb == 0:
            continue
        print(f"  {c:18s} {args.label_a}={sa:.2f}   {args.label_b}={sb:.2f}")

    # Show disagreeing examples
    diffs = [i for i in shared if a[i]["gold_mode"] != b[i]["gold_mode"]]
    if diffs:
        print(f"\ndisagreeing items ({len(diffs)} total; showing up to {args.show}):")
        for i in diffs[:args.show]:
            prompt = a[i]["prompt"][:200].replace("\n", " ")
            print(f"\n  id={i}   source={a[i]['source']}")
            print(f"    prompt: {prompt}{'...' if len(a[i]['prompt']) > 200 else ''}")
            print(f"    {args.label_a}: mode={a[i]['gold_mode']}  cats={[k for k,v in a[i]['categories'].items() if v]}")
            print(f"              note={a[i]['notes']}")
            print(f"    {args.label_b}: mode={b[i]['gold_mode']}  cats={[k for k,v in b[i]['categories'].items() if v]}")
            print(f"              note={b[i]['notes']}")


if __name__ == "__main__":
    main()
