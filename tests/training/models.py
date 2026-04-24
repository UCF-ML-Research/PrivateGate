"""Base-model registry for the D5 ablation.

D5 resolution: run the same training pipeline with each of (a)(b)(c)(d) and
compare. Keep the specs here so that the ablation driver enumerates them
instead of hard-coding strings.

  (a) ModernBERT-base   — ~149M params, SOTA 2024 encoder, 8k context, fine-tuned
  (b) DeBERTa-v3-small  —  ~44M params, strong mid-size baseline, fine-tuned
  (c) MiniLM-L12        —  ~33M params, fastest fine-tuned CPU inference
  (d) MiniLM-L6 frozen  —  R2-Router-style: frozen encoder + 9 independent
                           3-layer MLP heads. Smallest & fastest to train;
                           lowest CPU latency; direct port of R2-Router.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseModelSpec:
    key: str
    hf_id: str
    max_length: int
    approx_params_m: int
    cpu_infer_ms_target: int        # p95 target on t3.medium-class CPU
    freeze_encoder: bool = False    # True → R2-style frozen-encoder + MLP heads
    hidden_dims: tuple[int, ...] = (256, 128, 64)   # MLP head shape
    amp_ok: bool = True             # False ⇒ force fp32 (DeBERTa-v3 needs this;
                                    # its disentangled attention underflows in
                                    # bf16 → NaN logits).


MODEL_REGISTRY: dict[str, BaseModelSpec] = {
    "modernbert": BaseModelSpec(
        key="modernbert",
        hf_id="answerdotai/ModernBERT-base",
        max_length=8192,
        approx_params_m=149,
        cpu_infer_ms_target=120,
    ),
    "deberta-v3-small": BaseModelSpec(
        key="deberta-v3-small",
        hf_id="microsoft/deberta-v3-small",
        max_length=512,
        approx_params_m=44,
        cpu_infer_ms_target=60,
        amp_ok=False,
    ),
    "minilm-l12": BaseModelSpec(
        key="minilm-l12",
        hf_id="microsoft/MiniLM-L12-H384-uncased",
        max_length=512,
        approx_params_m=33,
        cpu_infer_ms_target=40,
    ),
    "minilm-l6-frozen-mlp": BaseModelSpec(
        key="minilm-l6-frozen-mlp",
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        max_length=256,
        approx_params_m=22,
        cpu_infer_ms_target=20,
        freeze_encoder=True,
        hidden_dims=(256, 128, 64),
    ),
}


def get_spec(key: str) -> BaseModelSpec:
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown base model {key!r}; known: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key]
