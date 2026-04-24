"""Download source datasets for the PrivateGate label pipeline.

Stores each dataset as JSONL in `data/raw/<name>/items.jsonl` using a
normalized row schema: `{id, prompt, source, source_meta}`. Downstream
scripts (sample.py, auto_label.py) consume this shape.

Targets:
  starter — small sample per dataset; used to validate the pipeline.
  full    — large sample sufficient to build the 50k weak-supervision pool
            plus the 5k gold draft.

All three starter datasets are publicly accessible on HuggingFace without
an access token. Gated datasets (WildChat-1M, LMSYS-Chat-1M, i2b2) are
intentionally omitted — add them later if the user provides credentials.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


def _write_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def download_oasst1(out_dir: Path, limit: int | None) -> int:
    """OpenAssistant/oasst1 — realistic user prompts (unlabeled)."""
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    rows = []
    for item in ds:
        if item.get("role") != "prompter":
            continue
        if item.get("lang") != "en":
            continue
        rows.append({
            "id": f"oasst1-{item['message_id']}",
            "prompt": item["text"],
            "source": "oasst1",
            "source_meta": {"lang": "en", "parent": item.get("parent_id")},
        })
        if limit and len(rows) >= limit:
            break
    return _write_jsonl(rows, out_dir / "oasst1" / "items.jsonl")


def download_ai4privacy(out_dir: Path, limit: int | None) -> int:
    """ai4privacy/pii-masking-200k — labeled PII spans on synthetic sentences.

    Used both as source prompts (for the weak-supervision pool) and as
    ground-truth spans for cross-checking Claude's span predictions.
    """
    ds = load_dataset("ai4privacy/pii-masking-200k", split="train")
    rows = []
    for i, item in enumerate(ds):
        lang = item.get("language") or item.get("LANGUAGE")
        if lang and str(lang).lower() not in {"english", "en"}:
            continue
        prompt = item.get("source_text") or item.get("unmasked_text") or ""
        if not prompt:
            continue
        rows.append({
            "id": f"ai4priv-{i}",
            "prompt": prompt,
            "source": "ai4privacy",
            "source_meta": {
                "masked_text": item.get("mask_text") or item.get("masked_text"),
                "privacy_mask": item.get("privacy_mask"),
                "language": str(lang) if lang else "English",
            },
        })
        if limit and len(rows) >= limit:
            break
    return _write_jsonl(rows, out_dir / "ai4privacy" / "items.jsonl")


def download_jailbreakbench(out_dir: Path, limit: int | None) -> int:
    """JailbreakBench/JBB-Behaviors — prompt-injection / jailbreak attempts."""
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    rows = []
    for split_name in list(ds.keys()):
        for i, item in enumerate(ds[split_name]):
            prompt = item.get("Goal") or item.get("Behavior") or item.get("goal") or ""
            if not prompt:
                continue
            rows.append({
                "id": f"jbb-{split_name}-{i}",
                "prompt": prompt,
                "source": "jailbreakbench",
                "source_meta": {
                    "split": split_name,
                    "category": item.get("Category") or item.get("category"),
                },
            })
            if limit and len(rows) >= limit:
                break
    return _write_jsonl(rows, out_dir / "jailbreakbench" / "items.jsonl")


DOWNLOADERS = {
    "oasst1": download_oasst1,
    "ai4privacy": download_ai4privacy,
    "jailbreakbench": download_jailbreakbench,
}


STARTER_LIMITS: dict[str, int | None] = {
    "oasst1": 2000,
    "ai4privacy": 2000,
    "jailbreakbench": None,   # small enough; take all
}

FULL_LIMITS: dict[str, int | None] = {
    "oasst1": 80_000,
    "ai4privacy": 50_000,
    "jailbreakbench": None,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["starter", "full"], default="starter")
    ap.add_argument("--datasets", nargs="+", default=list(DOWNLOADERS))
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    out_dir = Path(args.out)
    limits = STARTER_LIMITS if args.target == "starter" else FULL_LIMITS

    for name in args.datasets:
        if name not in DOWNLOADERS:
            print(f"[skip] unknown dataset: {name}")
            continue
        try:
            n = DOWNLOADERS[name](out_dir, limits.get(name))
            print(f"[ok]  {name}: {n} rows → {out_dir / name / 'items.jsonl'}")
        except Exception as e:
            print(f"[err] {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
