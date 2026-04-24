"""LLM-based synthesis of rare-class prompts.

Raw OASST1 / AI4Privacy traffic under-represents the high-severity classes
(PHI, PCI, secret, ip_confidential, regulated_eu/us, injection). This script
asks a large LLM to *generate* plausible prompts for a target class so we can
oversample during training.

Two backends:
  --backend vllm   (default) — local vLLM OpenAI-compat server (e.g. Qwen3.5).
                                No external API cost, no API key needed.
  --backend claude          — Anthropic API; requires ANTHROPIC_API_KEY.

Output rows match the same `{id, prompt, source, source_meta}` shape as
`download.py`. Labels are NOT attached here — run `auto_label.py` afterwards
so the same labeler that will be evaluated also labels these.

Usage (vLLM + Qwen, typical):
  python -m tests.data.synthesize_rare \\
      --out data/synthetic \\
      --per-class 600 \\
      --classes phi pci secret ip_confidential regulated_eu regulated_us injection \\
      --backend vllm \\
      --model Qwen/Qwen3.5-122B-A10B-FP8 \\
      --base-url http://localhost:8765/v1

Usage (Claude):
  export ANTHROPIC_API_KEY=...
  python -m tests.data.synthesize_rare --backend claude --model claude-opus-4-7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol

DEFAULT_VLLM_MODEL = "Qwen/Qwen3.5-122B-A10B-FP8"
DEFAULT_CLAUDE_MODEL = "claude-opus-4-7"
DEFAULT_VLLM_BASE_URL = "http://localhost:8765/v1"

CLASS_PROMPTS: dict[str, str] = {
    "phi": (
        "Generate realistic but FICTIONAL user prompts that would be sent to a "
        "general-purpose chatbot and that contain Protected Health Information "
        "(HIPAA 18 identifiers: MRN, diagnosis codes like ICD-10, provider names, "
        "dates of service, prescribed medications tied to a named individual). "
        "Vary styles: brief questions, long descriptions, medical-record excerpts, "
        "family member health summaries. Use made-up names and plausible but "
        "invalid MRNs."
    ),
    "pci": (
        "Generate realistic but FICTIONAL user prompts that contain payment-card "
        "data (full PAN-shaped numbers using Luhn-compliant test ranges like "
        "4111111111111111, CVV/CVV2, expiration dates, cardholder names, BIN "
        "ranges, or references to 'track data' / service codes). Vary intent: "
        "user asking for help with a chargeback, developer debugging, fraud "
        "pattern question."
    ),
    "secret": (
        "Generate realistic but FICTIONAL user prompts that contain credentials: "
        "API keys (sk-..., pk_live_..., ghp_..., xoxb-..., AKIA..., "
        "AIza...), OAuth tokens, JWT-shaped tokens, database connection strings "
        "with passwords, SSH private-key snippets, 24-word crypto mnemonics. "
        "Formats must LOOK real but be fabricated test values."
    ),
    "ip_confidential": (
        "Generate realistic user prompts from an employee asking about "
        "unreleased product strategy, pre-announcement roadmap, sealed "
        "financial results, acquisition/M&A plans, proprietary source code "
        "snippets marked 'internal only', or trade-secret manufacturing "
        "process details. Invent a plausible but FICTIONAL company name."
    ),
    "regulated_eu": (
        "Generate realistic user prompts that contain GDPR Art. 9 special-category "
        "data: explicit statements about a named person's race/ethnicity, "
        "religious belief, political affiliation, trade-union membership, sexual "
        "orientation, biometric data (fingerprint/face descriptors), or non-HIPAA "
        "health info. Use fictional names. Mix benign contexts and sensitive "
        "ones."
    ),
    "regulated_us": (
        "Generate realistic user prompts that describe US-export-controlled or "
        "CJIS-regulated content: ITAR/EAR-controlled technical data (e.g., "
        "rocket propulsion parameters, cryptographic implementation details "
        "above mass-market, missile guidance), ECCN classifications, criminal "
        "justice record queries (NCIC, rap sheet, case file content). All "
        "fictional, plausible phrasing."
    ),
    "injection": (
        "Generate realistic prompt-injection / jailbreak attempts a user or an "
        "adversarial agent might send: 'ignore previous instructions', DAN-style "
        "role override, hidden-instruction attempts ('the real instructions are: "
        "...'), system-prompt extraction attempts, indirect injection via "
        "pasted documents. Vary sophistication and length."
    ),
}

GENERATION_INSTRUCTIONS = """Return ONLY a JSON array of N distinct prompts. Each array element is a single-string user prompt (plain text, no escaped markdown). Do not include explanations, numbering, or keys. Example shape:

["prompt text one here", "prompt text two here", "prompt text three here"]

Requirements:
- Prompts must be DIVERSE in length, style, register, and domain.
- Mix of short (1 sentence) and long (multi-paragraph) prompts.
- Content must be fully fictional but plausibly real.
- Do NOT include any real person's PII or real secrets.
"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


class SynthClient(Protocol):
    """Minimal surface synth_batch() needs — either backend supplies it."""
    name: str

    def generate(self, system: str, user: str, max_tokens: int) -> str: ...


class ClaudeSynthClient:
    def __init__(self, model: str):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install anthropic") from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.name = f"claude/{model}"

    def generate(self, system: str, user: str, max_tokens: int) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text"))


class VLLMSynthClient:
    """Talks to a local vLLM OpenAI-compat server. Uses guided JSON schema
    (array of strings) when thinking is off so Qwen can't deviate from the
    shape the caller expects to json.loads()."""

    def __init__(self, model: str, base_url: str):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai") from e
        self._client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
        self.name = f"vllm/{model}"

    def generate(self, system: str, user: str, max_tokens: int) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.9,          # diverse sampling for synthesis
            top_p=0.95,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "PromptList",
                    "schema": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content or ""


def synth_batch(client: SynthClient, class_key: str, batch_size: int, max_retries: int = 3) -> list[str]:
    system = CLASS_PROMPTS[class_key]
    user = f"Generate {batch_size} prompts. {GENERATION_INSTRUCTIONS}"
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            raw_text = client.generate(system, user, max_tokens=8000)
            raw = _strip_fences(raw_text)
            arr = json.loads(raw)
            if isinstance(arr, list):
                return [str(p) for p in arr if isinstance(p, str) and p.strip()]
        except json.JSONDecodeError:
            time.sleep(0.5)
        except Exception:
            time.sleep(backoff)
            backoff *= 2
    return []


def run_class(client: SynthClient, class_key: str, total: int, batch_size: int, workers: int = 2) -> list[dict]:
    n_batches = (total + batch_size - 1) // batch_size
    results: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(synth_batch, client, class_key, batch_size) for _ in range(n_batches)]
        for fut in as_completed(futures):
            results.extend(fut.result())
    # Dedup + trim
    seen, out = set(), []
    for p in results:
        key = p.strip()[:500]
        if key in seen:
            continue
        seen.add(key)
        out.append(p.strip())
        if len(out) >= total:
            break
    return [
        {
            "id": f"syn-{class_key}-{i:05d}",
            "prompt": p,
            "source": "synthetic",
            "source_meta": {"target_class": class_key, "generator": client.name},
        }
        for i, p in enumerate(out)
    ]


def build_client(args) -> SynthClient:
    if args.backend == "claude":
        model = args.model or DEFAULT_CLAUDE_MODEL
        return ClaudeSynthClient(model=model)
    elif args.backend == "vllm":
        model = args.model or DEFAULT_VLLM_MODEL
        return VLLMSynthClient(model=model, base_url=args.base_url)
    else:  # pragma: no cover
        raise ValueError(f"unknown backend {args.backend}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--classes", nargs="+", default=list(CLASS_PROMPTS))
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--backend", choices=["vllm", "claude"], default="vllm",
                    help="LLM backend. vllm uses a local OpenAI-compat server (default Qwen).")
    ap.add_argument("--model", default=None,
                    help="Model id; backend-specific default used if omitted.")
    ap.add_argument("--base-url", default=DEFAULT_VLLM_BASE_URL,
                    help="OpenAI-compat endpoint (vllm backend only).")
    args = ap.parse_args()

    client = build_client(args)
    print(f"[synth] backend={args.backend}  model={getattr(client, 'model', '?')}  name={client.name}")

    out_dir = Path(args.out)
    for class_key in args.classes:
        if class_key not in CLASS_PROMPTS:
            print(f"[skip] unknown class: {class_key}", file=sys.stderr)
            continue
        rows = run_class(client, class_key, args.per_class, args.batch_size, args.workers)
        path = out_dir / f"{class_key}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[ok]  {class_key}: {len(rows)} prompts → {path}")


if __name__ == "__main__":
    main()
