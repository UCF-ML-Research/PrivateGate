"""Knowledge distillation: ModernBERT teacher → MiniLM-L6 frozen-MLP student.

The student must stay at 4.9 MB (same shape as the existing frozen-MLP D5
checkpoint) so the deployed pipeline is unchanged; we only hope to lift its
val/test macroF1 closer to the teacher by training the nine MLP heads against
teacher-provided soft targets on top of the 50k+3.7k combined corpus.

Loss
----
Per-class Bernoulli KD: each logit is an independent 2-way distribution, so
the "soft" KD term is a soft-target BCE rather than a class-softmax KL.

    soft_t = sigmoid(teacher_logits / T)     # target probability per class
    hard_bce  = BCE_with_logits(student_logits, hard_labels, pos_weight)
    soft_bce  = BCE_with_logits(student_logits / T, soft_t)   # no pos_weight
    loss      = α * hard_bce + (1 - α) * (T ** 2) * soft_bce

Defaults α=0.5, T=2.0. `T**2` matches Hinton (2015) so KD gradient magnitude
is comparable to the hard term.

Tokenization
------------
Student and teacher share the SAME tokenizer (the student's MiniLM-L6
tokenizer, max_len 256). This matches the deployment path (the deployed
student never sees ModernBERT tokens) and keeps the two encoders on identical
inputs so soft-target BCE is meaningful.

The teacher's encoder was trained at max_len 8192, so it will be mildly
out-of-distribution on <=256-token inputs — that's intentional and acceptable:
the student will never see longer inputs either.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_CATEGORIES
from .dataset import class_frequencies, load_split, to_multilabel
from .models import get_spec
from .train import (
    _collate,
    _compute_metrics,
    _PrivacyClassifier,
    _pos_weight_from_freq,
    _set_seed,
)


@dataclass
class DistillConfig:
    """Config for the KD run. Mirrors TrainConfig shape where applicable."""
    teacher_ckpt_dir: str
    student_key: str = "minilm-l6-frozen-mlp"
    weak_jsonl: str = "data/weak_labels/pool.jsonl"
    gold_train_jsonl: str = "data/gold/train.jsonl"
    val_jsonl: str = "data/gold/val.jsonl"
    output_dir: str = "artifacts/classifier/distilled/minilm-l6-frozen-mlp"

    categories: list[str] = field(default_factory=lambda: list(DEFAULT_CATEGORIES))
    epochs: int = 10
    batch_size: int = 32
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    class_weighting: str = "inverse_freq"
    pos_weight_cap: float = 20.0

    # KD knobs
    alpha: float = 0.5          # hard-label weight
    temperature: float = 2.0

    num_workers: int = 2
    fp16: bool = True
    seed: int = 1337


class _TeacherStudentDS:
    """Dataset that emits (input_ids, attention_mask, labels) for items drawn
    from a mixed pool (weak + gold). Tokenizes with the student tokenizer —
    the teacher then runs on the same ids inside the training loop."""
    def __init__(self, items, tokenizer, categories, max_length):
        self.items = items
        self.tokenizer = tokenizer
        self.categories = categories
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
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


def _load_teacher(teacher_ckpt_dir: str):
    """Returns the teacher `_PrivacyClassifier` in eval mode, with weights
    loaded from <teacher_ckpt_dir>/best.pt. Caller is responsible for .to()
    and AMP context."""
    import torch

    ckpt_path = Path(teacher_ckpt_dir) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    t_spec = get_spec(ckpt["base_model"])
    t_categories = list(ckpt["categories"])

    teacher = _PrivacyClassifier(spec=t_spec, n_labels=len(t_categories))
    state = ckpt["state_dict"]
    if "encoder" in state:
        teacher.encoder.load_state_dict(state["encoder"])
    if state.get("head") is not None and teacher.head is not None:
        teacher.head.load_state_dict(state["head"])
    if state.get("heads") is not None and teacher.heads is not None:
        teacher.heads.load_state_dict(state["heads"])
    # Teacher is forward-only; keep it fully frozen.
    for p in teacher.encoder.parameters():
        p.requires_grad = False
    if teacher.head is not None:
        for p in teacher.head.parameters():
            p.requires_grad = False
    if teacher.heads is not None:
        for p in teacher.heads.parameters():
            p.requires_grad = False
    return teacher, t_spec, t_categories


def run_distillation(config: DistillConfig) -> dict:
    """Run KD for one student. Returns a summary dict and writes best.pt +
    tokenizer/ + history.json + summary.json under config.output_dir."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    _set_seed(config.seed)

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    student_spec = get_spec(config.student_key)
    if not student_spec.freeze_encoder:
        raise ValueError(
            f"distill expects a frozen-encoder student; got {config.student_key}"
            " (freeze_encoder=False)"
        )

    # --- Data -----------------------------------------------------------
    weak_items = load_split(config.weak_jsonl)
    gold_items = load_split(config.gold_train_jsonl)
    combined = list(weak_items) + list(gold_items)
    random.Random(config.seed).shuffle(combined)
    val_items = load_split(config.val_jsonl)

    # Category frequencies — use combined corpus so pos_weight reflects the
    # training distribution, not just the tiny gold split.
    freqs = class_frequencies(combined, config.categories)
    n_labels = len(config.categories)

    # --- Tokenizer (STUDENT's; teacher sees the same tokens) ------------
    tokenizer = AutoTokenizer.from_pretrained(student_spec.hf_id)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # --- Models ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher, t_spec, t_categories = _load_teacher(config.teacher_ckpt_dir)
    if t_categories != config.categories:
        raise ValueError(
            f"teacher categories {t_categories} don't match student categories "
            f"{config.categories}"
        )
    teacher.to(device)
    teacher.eval_mode()

    student = _PrivacyClassifier(spec=student_spec, n_labels=n_labels)
    student.to(device)

    # --- DataLoaders ----------------------------------------------------
    train_ds = _TeacherStudentDS(combined, tokenizer, config.categories,
                                 student_spec.max_length)
    val_ds   = _TeacherStudentDS(val_items, tokenizer, config.categories,
                                 student_spec.max_length)

    def _wrap(batch): return _collate(batch, pad_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              collate_fn=_wrap, num_workers=config.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(32, config.batch_size), shuffle=False,
                            collate_fn=_wrap, num_workers=config.num_workers,
                            pin_memory=True)

    # --- Loss & optimizer ----------------------------------------------
    if config.class_weighting == "inverse_freq":
        pos_weight = _pos_weight_from_freq(freqs, config.categories, len(combined),
                                           config.pos_weight_cap).to(device)
    else:
        pos_weight = None
    hard_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Soft-target BCE: no pos_weight (the soft target is already fractional
    # and skewing it by inverse freq double-counts the class weighting).
    soft_criterion = torch.nn.BCEWithLogitsLoss()

    params = [p for p in student.parameters_trainable() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.head_lr,
                                  weight_decay=config.weight_decay)

    # Teacher runs under bf16 AMP when possible (ModernBERT supports it);
    # student forward is a single no-grad encoder + 9 small MLPs.
    use_amp_teacher = config.fp16 and device.type == "cuda" and t_spec.amp_ok
    use_amp_student = config.fp16 and device.type == "cuda" and student_spec.amp_ok
    amp_dtype = torch.bfloat16

    T = float(config.temperature)
    alpha = float(config.alpha)

    history: list[dict] = []
    best_macro_f1 = -1.0
    best_epoch = -1
    t_start = time.time()

    for epoch in range(1, config.epochs + 1):
        student.train_mode()
        teacher.eval_mode()

        running_total = 0.0
        running_hard = 0.0
        running_soft = 0.0
        n_batches = 0
        t_epoch = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Teacher forward (no grad). Cast back to fp32 before the loss so
            # the BCE math is stable regardless of AMP dtype.
            with torch.no_grad():
                if use_amp_teacher:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        teacher_logits = teacher.forward_logits(input_ids, attention_mask)
                else:
                    teacher_logits = teacher.forward_logits(input_ids, attention_mask)
                teacher_logits = teacher_logits.float()
                soft_targets = torch.sigmoid(teacher_logits / T)

            optimizer.zero_grad(set_to_none=True)
            if use_amp_student:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    student_logits = student.forward_logits(input_ids, attention_mask)
                    student_logits = student_logits.float()
                    hard = hard_criterion(student_logits, labels)
                    soft = soft_criterion(student_logits / T, soft_targets)
            else:
                student_logits = student.forward_logits(input_ids, attention_mask)
                hard = hard_criterion(student_logits, labels)
                soft = soft_criterion(student_logits / T, soft_targets)

            loss = alpha * hard + (1.0 - alpha) * (T ** 2) * soft
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
            optimizer.step()

            running_total += float(loss.detach().item())
            running_hard  += float(hard.detach().item())
            running_soft  += float(soft.detach().item())
            n_batches += 1

        train_loss = running_total / max(1, n_batches)
        train_hard = running_hard  / max(1, n_batches)
        train_soft = running_soft  / max(1, n_batches)

        # --- Val ---------------------------------------------------------
        student.eval_mode()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels_cpu = batch["labels"]
                if use_amp_student:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        logits = student.forward_logits(input_ids, attention_mask)
                else:
                    logits = student.forward_logits(input_ids, attention_mask)
                all_logits.append(logits.float().cpu().numpy())
                all_labels.append(labels_cpu.numpy())

        logits_np = np.concatenate(all_logits, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        metrics = _compute_metrics(logits_np, labels_np, config.categories)

        dt = time.time() - t_epoch
        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_hard_bce": train_hard,
            "train_soft_bce": train_soft,
            "val_macro_f1": metrics["macro_f1"],
            "val_micro_f1": metrics["micro_f1"],
            "val_macro_auroc": metrics["macro_auroc"],
            "val_per_class_f1": metrics["per_class_f1"],
            "val_per_class_auroc": metrics["per_class_auroc"],
            "elapsed_sec": round(dt, 1),
        }
        history.append(log_entry)
        print(
            f"[epoch {epoch}/{config.epochs}] loss={train_loss:.4f} "
            f"hard={train_hard:.4f} soft={train_soft:.4f}  "
            f"macroF1={metrics['macro_f1']:.4f}  microF1={metrics['micro_f1']:.4f}  "
            f"AUROC={metrics['macro_auroc']:.4f}  ({dt:.1f}s)",
            flush=True,
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            ckpt_path = out / "best.pt"
            torch.save(
                {
                    "state_dict": student.state_dict(),
                    "categories": config.categories,
                    "base_model": student_spec.key,
                    "hf_id": student_spec.hf_id,
                    "freeze_encoder": student_spec.freeze_encoder,
                    "hidden_dims": list(student_spec.hidden_dims),
                    "epoch": epoch,
                    "macro_f1": best_macro_f1,
                    # Provenance so downstream tooling can detect KD-trained
                    # ckpts. Safe to ignore; calibrate_checkpoint doesn't read it.
                    "distillation": {
                        "teacher_ckpt_dir": str(Path(config.teacher_ckpt_dir).resolve()),
                        "teacher_base_model": t_spec.key,
                        "alpha": alpha,
                        "temperature": T,
                    },
                },
                ckpt_path,
            )
            tokenizer.save_pretrained(out / "tokenizer")

    total_time = time.time() - t_start
    with (out / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "status": "ok",
        "mode": "distillation",
        "student_base_model": student_spec.key,
        "teacher_base_model": t_spec.key,
        "teacher_ckpt_dir": str(Path(config.teacher_ckpt_dir).resolve()),
        "alpha": alpha,
        "temperature": T,
        "n_train": len(combined),
        "n_train_weak": len(weak_items),
        "n_train_gold": len(gold_items),
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
