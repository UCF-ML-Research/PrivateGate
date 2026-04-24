"""Smoke tests for the training scaffold (no torch required)."""
from __future__ import annotations

import json

import pytest

from ..training import (
    ABLATION_KEYS,
    DistillConfig,
    MODEL_REGISTRY,
    TrainConfig,
    get_spec,
    run_d5_ablation,
    run_distillation,
    run_training,
    run_twostage,
)


def _write_gold_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


def _sample_item(id_: str, gold_mode: str, pii: bool = False) -> dict:
    cats = {
        "none": not pii,
        "pii": pii,
        "phi": False, "pci": False, "secret": False,
        "ip_confidential": False, "regulated_eu": False, "regulated_us": False,
        "injection": False,
    }
    return {
        "id": id_,
        "prompt": f"prompt {id_}",
        "source": "synthetic",
        "categories": cats,
        "severity": {"pii": 2} if pii else {},
        "spans": [],
        "gold_mode": gold_mode,
    }


def test_model_registry_has_all_d5_candidates():
    assert set(ABLATION_KEYS) == set(MODEL_REGISTRY)
    assert {
        "modernbert",
        "deberta-v3-small",
        "minilm-l12",
        "minilm-l6-frozen-mlp",
    } == set(MODEL_REGISTRY)


def test_frozen_variant_spec_matches_r2_recipe():
    spec = get_spec("minilm-l6-frozen-mlp")
    assert spec.freeze_encoder is True
    assert spec.hidden_dims == (256, 128, 64)


def test_get_spec_rejects_unknown_key():
    with pytest.raises(ValueError):
        get_spec("bogus-model")


def test_train_config_has_required_fields():
    # Guards the public dataclass surface used by the sbatch wrapper.
    cfg = TrainConfig(
        base_model="deberta-v3-small",
        train_jsonl="/tmp/_t.jsonl",
        val_jsonl="/tmp/_v.jsonl",
        output_dir="/tmp/_o",
    )
    assert cfg.epochs > 0
    assert 0 < cfg.lr < 1
    assert cfg.head_lr >= cfg.lr
    assert cfg.class_weighting in {"none", "inverse_freq", "manual"}
    # init_from is the two-stage warm-start hook; default must be None so the
    # existing single-stage trainer path is unchanged.
    assert cfg.init_from is None


def test_distill_config_defaults_are_sane():
    cfg = DistillConfig(
        teacher_ckpt_dir="/tmp/teacher",
        output_dir="/tmp/_o",
    )
    assert cfg.student_key == "minilm-l6-frozen-mlp"
    assert 0.0 <= cfg.alpha <= 1.0
    assert cfg.temperature > 0
    # Default student must be the frozen-encoder variant — any other choice
    # breaks the "same ship size as the existing MiniLM-L6 ckpt" invariant.
    assert get_spec(cfg.student_key).freeze_encoder is True


def test_twostage_and_distill_callables_exposed():
    # Surface-level check; full runs are exercised by the sbatch scripts.
    assert callable(run_twostage)
    assert callable(run_distillation)


def test_new_cli_entrypoints_importable():
    # Each CLI must have a `main()` callable so the sbatch `-m` invocations
    # succeed. Smoke-import only; no args parsed.
    from ..training.run_twostage import main as twostage_main
    from ..training.run_distill import main as distill_main
    from ..data.splice_boost import main as splice_main
    assert callable(twostage_main)
    assert callable(distill_main)
    assert callable(splice_main)


# The full training loop (torch + HF model download + GPU) is exercised by
# scripts/slurm/train_d5.sbatch on the B200 partition. Running it here would
# require torch/cuda and would pull a ~100MB encoder on first call, so we
# keep the unit suite data-only.
