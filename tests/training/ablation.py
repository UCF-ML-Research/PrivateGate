"""D5 base-model ablation driver.

Runs the same training pipeline across four architectures:
  (a) ModernBERT-base           — fine-tuned
  (b) DeBERTa-v3-small          — fine-tuned
  (c) MiniLM-L12                — fine-tuned
  (d) MiniLM-L6 + MLP heads     — frozen encoder, R2-Router recipe
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import TrainConfig
from .models import get_spec
from .train import run_training

ABLATION_KEYS: list[str] = [
    "modernbert",
    "deberta-v3-small",
    "minilm-l12",
    "minilm-l6-frozen-mlp",
]


def run_d5_ablation(
    train_jsonl: str,
    val_jsonl: str,
    output_root: str,
    epochs: int = 3,
    stage: str = "gold",
    keys: list[str] | None = None,
    epochs_overrides: dict[str, int] | None = None,
) -> dict[str, dict]:
    """Train each variant in turn; write per-variant summary and a top-level
    `ablation_summary.json` comparing them.
    """
    results: dict[str, dict] = {}
    epochs_overrides = epochs_overrides or {}
    for key in keys or ABLATION_KEYS:
        spec = get_spec(key)
        # Frozen-encoder variant trains heads only; it's fine to give it more
        # epochs since each pass is ~free compared to fine-tuning.
        eps = epochs_overrides.get(key, 15 if spec.freeze_encoder else epochs)
        cfg = TrainConfig(
            base_model=key,
            train_jsonl=train_jsonl,
            val_jsonl=val_jsonl,
            output_dir=f"{output_root.rstrip('/')}/{key}",
            epochs=eps,
            stage=stage,
        )
        print(f"\n===== ablation: {key}  (epochs={eps}, frozen={spec.freeze_encoder}) =====",
              flush=True)
        results[key] = run_training(cfg)

    leaderboard = sorted(
        (
            {
                "base_model": k,
                "best_macro_f1": r.get("best_macro_f1", -1),
                "best_epoch": r.get("best_epoch", -1),
                "total_time_sec": r.get("total_time_sec", -1),
                "output_dir": r.get("output_dir", ""),
            }
            for k, r in results.items()
        ),
        key=lambda x: x["best_macro_f1"],
        reverse=True,
    )
    Path(output_root).mkdir(parents=True, exist_ok=True)
    with (Path(output_root) / "ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"leaderboard": leaderboard, "results": results}, f, indent=2)
    return results
