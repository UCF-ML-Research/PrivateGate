"""Temperature scaling + per-class threshold calibration.

Post-hoc calibration for the multi-label classifier. Two complementary knobs:

  * Per-class **temperature** `T_c > 0`: rescales logits before sigmoid so the
    predicted probabilities match empirical frequencies. Fit by minimising BCE
    on the held-out val set via L-BFGS-B in log-space (T = exp(τ)).

  * Per-class **threshold** `θ_c ∈ (0, 1)`: used instead of the implicit 0.5
    cutoff when binarising. Fit by sweeping a grid and picking the θ that
    maximises that class's F1 on the val set — especially useful for rare
    classes where the default 0.5 is too conservative.

The calibration is saved next to each checkpoint as `calibration.json`:

    {
      "temperatures":     [T_c for c in categories],
      "thresholds":       [θ_c for c in categories],
      "metrics_uncalibrated": {...},   # on val (and test if provided)
      "metrics_calibrated":   {...},
      "categories": [...]
    }

The classifier head itself is NOT retrained — only the two scalar vectors.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def _sigmoid(x):
    import numpy as np
    # Stable sigmoid
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def compute_logits_for_ckpt(ckpt_dir, data_jsonl: str, device: str | None = None):
    """Load a trained checkpoint and run it on `data_jsonl`, returning
    (logits [N,C], labels [N,C], categories, meta).
    """
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from .dataset import load_split, to_multilabel
    from .models import get_spec
    from .train import _PrivacyClassifier, _collate

    ckpt_path = Path(ckpt_dir) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    categories = list(ckpt["categories"])
    spec = get_spec(ckpt["base_model"])

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = _PrivacyClassifier(spec=spec, n_labels=len(categories))
    state = ckpt["state_dict"]
    if "encoder" in state:
        model.encoder.load_state_dict(state["encoder"])
    if state.get("head") is not None and model.head is not None:
        model.head.load_state_dict(state["head"])
    if state.get("heads") is not None and model.heads is not None:
        model.heads.load_state_dict(state["heads"])
    model.to(device)
    model.eval_mode()

    tokenizer = AutoTokenizer.from_pretrained(Path(ckpt_dir) / "tokenizer")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    items = load_split(data_jsonl)

    class _DS:
        def __len__(self): return len(items)
        def __getitem__(self, i):
            enc = tokenizer(items[i].prompt, truncation=True,
                            max_length=spec.max_length, padding=False)
            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": to_multilabel(items[i], categories),
            }

    from torch.utils.data import DataLoader

    def wrap(batch): return _collate(batch, pad_id=pad_id)
    loader = DataLoader(_DS(), batch_size=32, shuffle=False, collate_fn=wrap,
                        num_workers=0)

    logits_chunks, label_chunks = [], []
    use_amp = device.type == "cuda" and spec.amp_ok
    amp_dtype = torch.bfloat16
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    lg = model.forward_logits(ids, mask)
            else:
                lg = model.forward_logits(ids, mask)
            logits_chunks.append(lg.float().cpu().numpy())
            label_chunks.append(batch["labels"].numpy())

    logits = np.concatenate(logits_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    meta = {
        "base_model": spec.key,
        "hf_id": spec.hf_id,
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_macro_f1": ckpt.get("macro_f1"),
    }
    return logits, labels, categories, meta


def _bce_loss_np(logits, labels):
    """Per-class mean BCE loss. Returns vector [C]."""
    import numpy as np
    # log(1 + exp(-x*sign))-ish, numerically stable
    probs = _sigmoid(logits)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return -(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)).mean(axis=0)


def fit_temperature(logits, labels, init: float = 1.0, bounds=(0.25, 8.0)):
    """Per-class temperature fit by L-BFGS-B (BCE loss on val)."""
    import numpy as np
    from scipy.optimize import minimize_scalar

    C = logits.shape[1]
    temps = np.ones(C, dtype=np.float64)
    for c in range(C):
        lg = logits[:, c]
        y = labels[:, c]
        # If the class has no positive or negative samples, temperature doesn't
        # matter — keep it at 1.
        if len(set(y)) < 2:
            continue

        def nll(T):
            p = _sigmoid(lg / T)
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

        res = minimize_scalar(nll, bounds=bounds, method="bounded",
                              options={"xatol": 1e-4})
        temps[c] = float(res.x)
    return temps


def fit_thresholds(probs, labels, grid_min: float = 0.05, grid_max: float = 0.95,
                   n_grid: int = 91):
    """Per-class threshold that maximises F1 on `probs`, `labels`."""
    import numpy as np
    from sklearn.metrics import f1_score

    C = probs.shape[1]
    thresh = np.full(C, 0.5, dtype=np.float64)
    grid = np.linspace(grid_min, grid_max, n_grid)
    for c in range(C):
        y = labels[:, c]
        if len(set(y)) < 2:
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (probs[:, c] >= t).astype(int)
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresh[c] = best_t
    return thresh


def ece_multilabel(probs, labels, n_bins: int = 15):
    """Macro-averaged ECE over classes. Per class: expected absolute gap
    between predicted confidence and empirical frequency across `n_bins`
    equal-width bins. Missing bins contribute 0.
    """
    import numpy as np

    C = probs.shape[1]
    eces = []
    for c in range(C):
        p = probs[:, c]
        y = labels[:, c]
        if len(set(y)) < 2:
            eces.append(float("nan"))
            continue
        ece = 0.0
        for i in range(n_bins):
            lo = i / n_bins
            hi = (i + 1) / n_bins
            mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
            if not mask.any():
                continue
            conf = p[mask].mean()
            acc = y[mask].mean()
            ece += (mask.sum() / len(p)) * abs(conf - acc)
        eces.append(float(ece))
    finite = [e for e in eces if not math.isnan(e)]
    return (sum(finite) / len(finite) if finite else float("nan")), eces


def classification_metrics(logits, labels, categories, thresholds=None):
    """Compute macroF1, microF1, per-class F1, AUROC, ECE."""
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score

    probs = _sigmoid(logits)
    if thresholds is None:
        preds = (probs >= 0.5).astype(int)
    else:
        preds = (probs >= thresholds[None, :]).astype(int)

    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    micro_f1 = float(f1_score(labels, preds, average="micro", zero_division=0))
    per_class_f1 = {
        c: float(f1_score(labels[:, i], preds[:, i], zero_division=0))
        for i, c in enumerate(categories)
    }
    per_class_auroc = {}
    aurocs = []
    for i, c in enumerate(categories):
        if len(set(labels[:, i])) < 2:
            per_class_auroc[c] = float("nan")
            continue
        auc = float(roc_auc_score(labels[:, i], probs[:, i]))
        per_class_auroc[c] = auc
        aurocs.append(auc)
    macro_auroc = float(sum(aurocs) / len(aurocs)) if aurocs else float("nan")
    ece, per_class_ece = ece_multilabel(probs, labels)
    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_auroc": macro_auroc,
        "macro_ece": ece,
        "per_class_f1": per_class_f1,
        "per_class_auroc": per_class_auroc,
        "per_class_ece": dict(zip(categories, per_class_ece)),
    }


def calibrate_checkpoint(ckpt_dir, val_jsonl: str, test_jsonl: str | None = None,
                         device: str | None = None) -> dict:
    """Full pipeline for one variant: compute val logits → fit T → fit θ →
    compute test metrics if provided → write calibration.json.
    """
    import numpy as np

    ckpt_dir = Path(ckpt_dir)

    # Val pass (used for fitting + reporting)
    val_logits, val_labels, categories, meta = compute_logits_for_ckpt(
        ckpt_dir, val_jsonl, device=device
    )
    uncalibrated_val = classification_metrics(val_logits, val_labels, categories)

    temperatures = fit_temperature(val_logits, val_labels)
    val_scaled = val_logits / temperatures[None, :]
    val_probs_cal = _sigmoid(val_scaled)
    thresholds = fit_thresholds(val_probs_cal, val_labels)

    calibrated_val = classification_metrics(
        val_scaled, val_labels, categories, thresholds=thresholds
    )

    result = {
        "ckpt_dir": str(ckpt_dir),
        "categories": categories,
        "temperatures": temperatures.tolist(),
        "thresholds": thresholds.tolist(),
        "meta": meta,
        "metrics_uncalibrated_val": uncalibrated_val,
        "metrics_calibrated_val": calibrated_val,
    }

    if test_jsonl:
        test_logits, test_labels, _, _ = compute_logits_for_ckpt(
            ckpt_dir, test_jsonl, device=device
        )
        uncalibrated_test = classification_metrics(test_logits, test_labels, categories)
        test_scaled = test_logits / temperatures[None, :]
        calibrated_test = classification_metrics(
            test_scaled, test_labels, categories, thresholds=thresholds
        )
        result["metrics_uncalibrated_test"] = uncalibrated_test
        result["metrics_calibrated_test"] = calibrated_test

    with (ckpt_dir / "calibration.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result
