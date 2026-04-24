"""Apply a Labeler (Claude API or local vLLM) to a JSONL of prompts.

Emits one schema-validated privategate-gold record per labeled prompt. Flags:

  --backend {claude,vllm}        which labeler to use
  --model MODEL                  backend-specific model id
  --base-url URL                 for vllm backend (default http://localhost:8000/v1)
  --thinking / --no-thinking     for vllm backend; controls Qwen chain-of-thought
  --workers N                    client-side concurrency (server batches internally)
  --sample N                     label only the first N items (for pilots)

Usage (vLLM + thinking off, pilot):
  python -m tests.data.auto_label \\
      --in  data/sampled/pilot_50.jsonl \\
      --out data/weak_labels/pilot_50_thinking_off.jsonl \\
      --backend vllm --base-url http://localhost:8765/v1 --no-thinking --workers 8

Usage (Claude, small trial):
  export ANTHROPIC_API_KEY=...
  python -m tests.data.auto_label \\
      --in  data/sampled/pilot_50.jsonl \\
      --out data/weak_labels/claude_pilot.jsonl \\
      --backend claude --sample 50
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from .labeler import ClaudeLabeler, Labeler, VLLMLabeler
from .schema import REQUIRED_CATEGORIES, validate_item


def normalize(label: dict, source_item: dict, labeler_name: str) -> dict:
    """Merge a Labeler output with the source prompt into a full gold record."""
    cats = dict(label.get("categories") or {})
    for c in REQUIRED_CATEGORIES:
        cats.setdefault(c, False)
    # Enforce schema invariant: none <=> no other categories true.
    any_other = any(cats[c] for c in REQUIRED_CATEGORIES if c != "none")
    cats["none"] = not any_other
    severity = {
        k: int(v)
        for k, v in (label.get("severity") or {}).items()
        if isinstance(v, (int, float))
    }
    return {
        "id": source_item["id"],
        "prompt": source_item["prompt"],
        "source": source_item["source"],
        "categories": cats,
        "severity": severity,
        "spans": label.get("spans") or [],
        "gold_mode": label.get("gold_mode") or "plaintext",
        "notes": label.get("notes") or "",
        "annotators": [labeler_name],
        "adjudicated": False,
    }


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield json.loads(line)


def run(
    labeler: Labeler,
    in_path: Path,
    out_path: Path,
    workers: int,
    limit: int | None = None,
) -> tuple[int, int]:
    items = list(iter_jsonl(in_path))
    if limit is not None:
        items = items[:limit]
    if not items:
        raise RuntimeError(f"no items to label in {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    with out_path.open("w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(labeler.label, it["prompt"]): it for it in items}
        for fut in as_completed(futures):
            src = futures[fut]
            label = fut.result()
            if label is None:
                n_fail += 1
                continue
            record = normalize(label, src, labeler.name)
            try:
                validate_item(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as e:
                print(f"  ! schema-invalid label for {src['id']}: {e}", file=sys.stderr)
                n_fail += 1
    return n_ok, n_fail


def build_labeler(args) -> Labeler:
    if args.backend == "claude":
        model = args.model or "claude-opus-4-7"
        return ClaudeLabeler(model=model)
    elif args.backend == "vllm":
        model = args.model or "Qwen/Qwen3.5-122B-A10B-FP8"
        return VLLMLabeler(
            model=model,
            base_url=args.base_url,
            thinking=args.thinking,
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown backend {args.backend}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--backend", choices=["claude", "vllm"], required=True)
    ap.add_argument("--model", default=None, help="backend-specific model id")
    ap.add_argument("--base-url", default="http://localhost:8000/v1",
                    help="OpenAI-compat endpoint for vllm backend")
    thinking_group = ap.add_mutually_exclusive_group()
    thinking_group.add_argument("--thinking", dest="thinking", action="store_true",
                                help="vLLM: enable Qwen3 <think>...</think> reasoning")
    thinking_group.add_argument("--no-thinking", dest="thinking", action="store_false",
                                help="vLLM: disable thinking + enable guided JSON (default)")
    ap.set_defaults(thinking=False)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sample", type=int, default=None,
                    help="label only the first N items")
    args = ap.parse_args()

    labeler = build_labeler(args)
    n_ok, n_fail = run(
        labeler, Path(args.in_path), Path(args.out_path),
        args.workers, args.sample,
    )
    print(f"labeler={labeler.name}  ok={n_ok}  fail={n_fail}  out={args.out_path}")


if __name__ == "__main__":
    main()
