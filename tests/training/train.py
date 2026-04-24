"""Single-run trainer (one base model, one stage).

Two architectures, selected via `BaseModelSpec.freeze_encoder`:

  * Fine-tuned encoder (default)
      HF AutoModel → [CLS] / mean-pool hidden → single Linear head → 9 logits.
      All encoder weights update; small LR (2e-5) with linear warmup + cosine.

  * Frozen encoder + MLP heads  (R2-Router recipe, D5(d))
      Encoder is frozen (requires_grad=False on every param). Nine independent
      3-layer MLPs `[256,128,64]` with ReLU + Sigmoid output, one per class.
      Only the heads train; larger LR (1e-3); no warmup needed.

Loss: per-class BCE with logits, pos_weight from inverse class frequency on the
training split (capped at `config.pos_weight_cap`).

Eval: macro-F1, per-class F1, macro AUROC. Best ckpt by val macro-F1 is kept.

torch/transformers imports are lazy so the rest of the harness can run without
a GPU stack.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

from .config import TrainConfig
from .dataset import class_frequencies, load_split, to_multilabel
from .models import BaseModelSpec, get_spec


# ----------------------------------------------------------------------------
# torch-aware internals (imported lazily so pure-data callers never pay the
# 2 GB-of-RAM torch import tax).
# ----------------------------------------------------------------------------

def _build_heads(hidden_size: int, n_labels: int, hidden_dims, dropout: float):
    import torch.nn as nn

    class MLPHead(nn.Module):
        def __init__(self, in_dim: int, hidden_dims, dropout: float):
            super().__init__()
            dims = [in_dim, *hidden_dims]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dims[-1], 1))   # scalar logit per class
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return nn.ModuleList(
        [MLPHead(hidden_size, hidden_dims, dropout) for _ in range(n_labels)]
    )


def _mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class _PrivacyClassifier:
    """Encoder + head(s). Works for both fine-tuned and frozen-encoder modes."""

    def __init__(self, spec: BaseModelSpec, n_labels: int):
        import torch
        from torch import nn
        from transformers import AutoModel

        self.spec = spec
        self.n_labels = n_labels
        # Force fp32 weight loading regardless of the checkpoint's saved dtype.
        # Some HF checkpoints (e.g. microsoft/deberta-v3-small in transformers
        # 5.x) load as fp16, which breaks the fp32 classifier head with
        # "mat1 and mat2 must have the same dtype". Casting inside autocast
        # then relies on AMP context rather than on checkpoint dtype.
        try:
            self.encoder = AutoModel.from_pretrained(spec.hf_id, dtype=torch.float32)
        except TypeError:
            # Older transformers (<5.0) used torch_dtype.
            self.encoder = AutoModel.from_pretrained(spec.hf_id, torch_dtype=torch.float32)
        hidden_size = self.encoder.config.hidden_size

        if spec.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
            self.heads = _build_heads(
                hidden_size=hidden_size,
                n_labels=n_labels,
                hidden_dims=spec.hidden_dims,
                dropout=0.1,
            )
            self.head = None
        else:
            self.heads = None
            self.head = nn.Linear(hidden_size, n_labels)

        self._torch = torch
        self._nn = nn

    def to(self, device):
        self.encoder.to(device)
        if self.heads is not None:
            self.heads.to(device)
        if self.head is not None:
            self.head.to(device)
        return self

    def parameters_trainable(self):
        params = []
        if not self.spec.freeze_encoder:
            params += list(self.encoder.parameters())
        if self.heads is not None:
            params += list(self.heads.parameters())
        if self.head is not None:
            params += list(self.head.parameters())
        return params

    def forward_logits(self, input_ids, attention_mask):
        torch = self._torch
        if self.spec.freeze_encoder:
            with torch.no_grad():
                enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = _mean_pool(enc_out.last_hidden_state, attention_mask)

        if self.heads is not None:
            logits = torch.stack([h(pooled) for h in self.heads], dim=1)
        else:
            logits = self.head(pooled)
        return logits

    def train_mode(self):
        if not self.spec.freeze_encoder:
            self.encoder.train()
        if self.heads is not None:
            self.heads.train()
        if self.head is not None:
            self.head.train()

    def eval_mode(self):
        self.encoder.eval()
        if self.heads is not None:
            self.heads.eval()
        if self.head is not None:
            self.head.eval()

    def state_dict(self):
        sd = {}
        if not self.spec.freeze_encoder:
            sd["encoder"] = self.encoder.state_dict()
        if self.heads is not None:
            sd["heads"] = self.heads.state_dict()
        if self.head is not None:
            sd["head"] = self.head.state_dict()
        return sd


class _JsonlDataset:
    def __init__(self, items, tokenizer, categories, max_length):
        self.items = items
        self.tokenizer = tokenizer
        self.categories = categories
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        # item is a GoldItem (frozen dataclass)
        enc = self.tokenizer(
            item.prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        labels = to_multilabel(item, self.categories)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


def _collate(batch, pad_id: int):
    import torch

    max_len = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.zeros((len(batch), len(batch[0]["labels"])), dtype=torch.float)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
        mask[i, :n] = torch.tensor(b["attention_mask"], dtype=torch.long)
        labels[i] = torch.tensor(b["labels"], dtype=torch.float)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def _compute_metrics(logits_np, labels_np, categories: list[str], threshold: float = 0.5) -> dict:
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    preds = (logits_np >= 0.0).astype(int)        # BCE-with-logits threshold = sigmoid(0)=0.5
    labels = labels_np.astype(int)

    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    per_class_f1 = {
        c: float(f1_score(labels[:, i], preds[:, i], zero_division=0))
        for i, c in enumerate(categories)
    }

    # AUROC per class (falls back to NaN for classes with only one label present)
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    per_class_auroc: dict[str, float] = {}
    aurocs = []
    for i, c in enumerate(categories):
        if len(set(labels[:, i])) < 2:
            per_class_auroc[c] = float("nan")
            continue
        auc = float(roc_auc_score(labels[:, i], probs[:, i]))
        per_class_auroc[c] = auc
        aurocs.append(auc)
    macro_auroc = float(np.mean(aurocs)) if aurocs else float("nan")

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "macro_auroc": macro_auroc,
        "per_class_f1": per_class_f1,
        "per_class_auroc": per_class_auroc,
    }


def _pos_weight_from_freq(freqs: dict[str, int], categories: list[str], n: int, cap: float):
    import torch

    # pos_weight = #neg / #pos, capped; class totally absent → cap
    weights = []
    for c in categories:
        p = max(1, freqs.get(c, 0))
        w = (n - p) / p
        weights.append(min(w, cap))
    return torch.tensor(weights, dtype=torch.float)


def _set_seed(seed: int):
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------

def run_training(config: TrainConfig) -> dict:
    """Run one training stage end-to-end and return a result summary."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    spec = get_spec(config.base_model)
    _set_seed(config.seed)

    train_items = load_split(config.train_jsonl)
    val_items = load_split(config.val_jsonl)

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    freqs = class_frequencies(train_items, config.categories)
    n_labels = len(config.categories)

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    model = _PrivacyClassifier(spec=spec, n_labels=n_labels)
    # Optional warm start from a prior checkpoint (two-stage weak→gold etc.).
    # Skipped silently when `init_from` is None or the file is absent.
    if getattr(config, "init_from", None):
        init_path = Path(config.init_from)
        if init_path.is_file():
            init_ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
            init_state = init_ckpt.get("state_dict", init_ckpt)
            if "encoder" in init_state:
                model.encoder.load_state_dict(init_state["encoder"])
            if init_state.get("head") is not None and model.head is not None:
                model.head.load_state_dict(init_state["head"])
            if init_state.get("heads") is not None and model.heads is not None:
                model.heads.load_state_dict(init_state["heads"])
            print(f"[init] warm-started from {init_path}", flush=True)
        else:
            print(f"[init] WARNING: init_from={init_path} not found; "
                  f"training from HF init", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = _JsonlDataset(train_items, tokenizer, config.categories, spec.max_length)
    val_ds   = _JsonlDataset(val_items,   tokenizer, config.categories, spec.max_length)

    def _wrap(batch): return _collate(batch, pad_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              collate_fn=_wrap, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(32, config.batch_size), shuffle=False,
                            collate_fn=_wrap, num_workers=config.num_workers, pin_memory=True)

    # Class-weighted BCE ---------------------------------------------------
    if config.class_weighting == "inverse_freq":
        pos_weight = _pos_weight_from_freq(freqs, config.categories, len(train_items),
                                           config.pos_weight_cap).to(device)
    elif config.class_weighting == "manual":
        pos_weight = torch.tensor(
            [config.manual_class_weights.get(c, 1.0) for c in config.categories],
            dtype=torch.float,
        ).to(device)
    else:
        pos_weight = None

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer ------------------------------------------------------------
    if spec.freeze_encoder:
        # Heads-only: use head_lr, no warmup, no weight-decay on biases.
        params = [p for p in model.parameters_trainable() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=config.head_lr, weight_decay=config.weight_decay)
        scheduler = None
    else:
        # Fine-tune encoder: small LR + linear warmup.
        params = model.parameters_trainable()
        optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        total_steps = max(1, len(train_loader) * config.epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * config.warmup_ratio),
            num_training_steps=total_steps,
        )

    # bfloat16 over float16: matches fp32 numeric range, so no GradScaler /
    # unscale step is needed. B200 has first-class bf16 throughput and this
    # sidesteps the "Attempting to unscale FP16 gradients" failure that
    # DeBERTa-v3's disentangled attention triggers under fp16+GradScaler.
    # Per-spec opt-out: `amp_ok=False` forces fp32 for models whose low-
    # precision paths produce NaN (DeBERTa-v3 relative-position encoding).
    use_amp = config.fp16 and device.type == "cuda" and spec.amp_ok
    amp_dtype = torch.bfloat16

    # Training loop --------------------------------------------------------
    history: list[dict] = []
    best_macro_f1 = -1.0
    best_epoch = -1
    t_start = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train_mode()
        running = 0.0
        n_batches = 0
        t_epoch = time.time()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model.forward_logits(input_ids, attention_mask)
                    loss = criterion(logits, labels)
            else:
                logits = model.forward_logits(input_ids, attention_mask)
                loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters_trainable() if p.requires_grad],
                config.max_grad_norm,
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running += float(loss.detach().item())
            n_batches += 1

        train_loss = running / max(1, n_batches)

        # Val eval --------------------------------------------------------
        model.eval_mode()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"]
                if use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = model.forward_logits(input_ids, attention_mask)
                else:
                    logits = model.forward_logits(input_ids, attention_mask)
                logits = logits.float().cpu()
                all_logits.append(logits.numpy())
                all_labels.append(labels.numpy())

        import numpy as np
        logits_np = np.concatenate(all_logits, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        metrics = _compute_metrics(logits_np, labels_np, config.categories)

        dt = time.time() - t_epoch
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": metrics["macro_f1"],
            "val_micro_f1": metrics["micro_f1"],
            "val_macro_auroc": metrics["macro_auroc"],
            "val_per_class_f1": metrics["per_class_f1"],
            "val_per_class_auroc": metrics["per_class_auroc"],
            "elapsed_sec": round(dt, 1),
        }
        history.append(log_entry)
        print(
            f"[epoch {epoch}/{config.epochs}] loss={train_loss:.4f}  "
            f"macroF1={metrics['macro_f1']:.4f}  microF1={metrics['micro_f1']:.4f}  "
            f"AUROC={metrics['macro_auroc']:.4f}  ({dt:.1f}s)",
            flush=True,
        )

        # Checkpoint best -----------------------------------------------
        if config.save_best and metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            ckpt_path = out / "best.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "categories": config.categories,
                    "base_model": spec.key,
                    "hf_id": spec.hf_id,
                    "freeze_encoder": spec.freeze_encoder,
                    "hidden_dims": list(spec.hidden_dims),
                    "epoch": epoch,
                    "macro_f1": best_macro_f1,
                },
                ckpt_path,
            )
            tokenizer.save_pretrained(out / "tokenizer")

    total_time = time.time() - t_start

    # Persist history + summary ------------------------------------------
    with (out / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "status": "ok",
        "stage": config.stage,
        "base_model": spec.key,
        "hf_id": spec.hf_id,
        "freeze_encoder": spec.freeze_encoder,
        "max_length": spec.max_length,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_labels": n_labels,
        "class_frequencies": freqs,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "last_metrics": history[-1] if history else None,
        "total_time_sec": round(total_time, 1),
        "output_dir": str(out),
        "device": str(device),
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
