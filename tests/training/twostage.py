"""Two-stage training: weak-supervision warmup then gold fine-tune.

Stage 1 trains each variant on the 50 k weak-label pool (Qwen-thinking-OFF
labels on 50 k OASST1/AI4Privacy/synthetic prompts). Stage 2 resumes from the
stage-1 `best.pt` and fine-tunes on the 3.7 k adjudicated gold train split.

Both stages eval on the same gold val split — the stage-1 run is effectively
a representation-pretraining pass whose macroF1 is incidental; the stage-2
final number is what we compare to the single-stage D5 baseline.

Layout written:
    <output_root>/<variant>/stage_weak/best.pt  (+ history.json, summary.json)
    <output_root>/<variant>/stage_gold/best.pt  (+ history.json, summary.json)
    <output_root>/<variant>/best.pt             (copy of stage_gold/best.pt so
                                                 `run_calibration.py --root`
                                                 picks it up unchanged)
    <output_root>/<variant>/tokenizer/          (copy of stage_gold tokenizer)
    <output_root>/twostage_summary.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .ablation import ABLATION_KEYS
from .config import TrainConfig
from .models import get_spec
from .train import run_training


# Epoch budgets per stage. Fine-tuned variants: 2 weak + 4 gold. Frozen-
# encoder variant trains heads only, so weak pass is cheap → give it more.
DEFAULT_EPOCHS_WEAK = {
    "modernbert": 2,
    "deberta-v3-small": 2,
    "minilm-l12": 2,
    "minilm-l6-frozen-mlp": 4,
}
DEFAULT_EPOCHS_GOLD = {
    "modernbert": 4,
    "deberta-v3-small": 4,
    "minilm-l12": 4,
    "minilm-l6-frozen-mlp": 15,
}


def run_twostage_variant(
    key: str,
    weak_jsonl: str,
    gold_train_jsonl: str,
    val_jsonl: str,
    variant_out: str,
    epochs_weak: int,
    epochs_gold: int,
) -> dict:
    """Run weak→gold for a single variant. Returns a merged summary dict."""
    variant_out_p = Path(variant_out)
    variant_out_p.mkdir(parents=True, exist_ok=True)
    stage_weak_dir = variant_out_p / "stage_weak"
    stage_gold_dir = variant_out_p / "stage_gold"

    # --- Stage 1: weak-supervision warmup --------------------------------
    cfg_weak = TrainConfig(
        base_model=key,
        train_jsonl=weak_jsonl,
        val_jsonl=val_jsonl,
        output_dir=str(stage_weak_dir),
        epochs=epochs_weak,
        stage="weak",
    )
    print(f"\n---- [{key}] stage 1 (weak): {epochs_weak} epochs on {weak_jsonl} ----",
          flush=True)
    weak_summary = run_training(cfg_weak)
    weak_ckpt = stage_weak_dir / "best.pt"
    if not weak_ckpt.is_file():
        raise RuntimeError(f"stage 1 did not produce a checkpoint: {weak_ckpt}")

    # --- Stage 2: gold fine-tune, warm-started from stage 1 --------------
    cfg_gold = TrainConfig(
        base_model=key,
        train_jsonl=gold_train_jsonl,
        val_jsonl=val_jsonl,
        output_dir=str(stage_gold_dir),
        epochs=epochs_gold,
        stage="gold",
        init_from=str(weak_ckpt),
    )
    print(f"\n---- [{key}] stage 2 (gold): {epochs_gold} epochs on {gold_train_jsonl} ----",
          flush=True)
    gold_summary = run_training(cfg_gold)

    # Expose stage-2 artifacts at the variant root so that
    # `run_calibration.py --root artifacts/classifier/d5_twostage` works without
    # changes: it scans top-level subdirs for a `best.pt`.
    src_best = stage_gold_dir / "best.pt"
    src_tok  = stage_gold_dir / "tokenizer"
    if src_best.is_file():
        shutil.copy2(src_best, variant_out_p / "best.pt")
    if src_tok.is_dir():
        dst_tok = variant_out_p / "tokenizer"
        if dst_tok.exists():
            shutil.rmtree(dst_tok)
        shutil.copytree(src_tok, dst_tok)

    return {
        "variant": key,
        "stage_weak": weak_summary,
        "stage_gold": gold_summary,
        "best_macro_f1_weak": weak_summary.get("best_macro_f1", -1),
        "best_macro_f1_gold": gold_summary.get("best_macro_f1", -1),
        "output_dir": str(variant_out_p),
    }


def run_twostage(
    weak_jsonl: str,
    gold_train_jsonl: str,
    val_jsonl: str,
    output_root: str,
    keys: list[str] | None = None,
    epochs_weak_overrides: dict[str, int] | None = None,
    epochs_gold_overrides: dict[str, int] | None = None,
) -> dict[str, dict]:
    """Two-stage training across variants; writes per-variant and top-level
    summaries.
    """
    keys = keys or list(ABLATION_KEYS)
    epochs_weak_overrides = epochs_weak_overrides or {}
    epochs_gold_overrides = epochs_gold_overrides or {}

    results: dict[str, dict] = {}
    for key in keys:
        spec = get_spec(key)  # validates early
        e_w = epochs_weak_overrides.get(key, DEFAULT_EPOCHS_WEAK[key])
        e_g = epochs_gold_overrides.get(key, DEFAULT_EPOCHS_GOLD[key])
        variant_out = f"{output_root.rstrip('/')}/{key}"
        print(f"\n===== twostage: {key} (frozen={spec.freeze_encoder}, "
              f"weak_epochs={e_w}, gold_epochs={e_g}) =====", flush=True)
        results[key] = run_twostage_variant(
            key=key,
            weak_jsonl=weak_jsonl,
            gold_train_jsonl=gold_train_jsonl,
            val_jsonl=val_jsonl,
            variant_out=variant_out,
            epochs_weak=e_w,
            epochs_gold=e_g,
        )

    leaderboard = sorted(
        (
            {
                "base_model": k,
                "best_macro_f1_gold": r.get("best_macro_f1_gold", -1),
                "best_macro_f1_weak": r.get("best_macro_f1_weak", -1),
                "output_dir": r.get("output_dir", ""),
            }
            for k, r in results.items()
        ),
        key=lambda x: x["best_macro_f1_gold"],
        reverse=True,
    )
    Path(output_root).mkdir(parents=True, exist_ok=True)
    with (Path(output_root) / "twostage_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"leaderboard": leaderboard, "results": results}, f, indent=2)
    return results
