"""Training configuration — single source of truth for a run.

Stage values:
  "weak"        — warmup on auto-labeled (LLM-labeled) data
  "gold"        — fine-tune on human-adjudicated gold set
  "adversarial" — adversarial augmentation fine-tune
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_CATEGORIES: list[str] = [
    "none", "pii", "phi", "pci", "secret",
    "ip_confidential", "regulated_eu", "regulated_us", "injection",
]


@dataclass
class TrainConfig:
    base_model: str                       # key in MODEL_REGISTRY
    train_jsonl: str
    val_jsonl: str
    output_dir: str
    categories: list[str] = field(default_factory=lambda: list(DEFAULT_CATEGORIES))
    epochs: int = 3
    batch_size: int = 16
    # Fine-tuned encoders use a smaller LR (2e-5); the frozen-encoder variant
    # trains only MLP heads and can use a larger LR (1e-3) like R2-Router.
    lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    class_weighting: str = "inverse_freq"    # "none" | "inverse_freq" | "manual"
    manual_class_weights: dict[str, float] = field(default_factory=dict)
    pos_weight_cap: float = 20.0             # clamp inverse-freq weights
    eval_every_epoch: bool = True
    save_best: bool = True                   # keep only the best-macro-F1 ckpt
    num_workers: int = 2
    fp16: bool = True
    seed: int = 1337
    stage: str = "gold"
    # Optional path to a prior `best.pt` to warm-start from. When set, the
    # encoder + head/heads state dicts are loaded into the freshly-constructed
    # classifier *before* training begins. Used by the two-stage weak→gold
    # recipe. `None` keeps the existing behaviour (train from the HF init).
    init_from: str | None = None
