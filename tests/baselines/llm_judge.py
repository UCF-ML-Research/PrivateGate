"""LLM-judge router (B5) — STUBBED, blocked on D5.

D5 choices:
  (a) GPT-4o-mini via OpenAI API
  (b) Claude Haiku via Anthropic API
  (c) Llama-Guard-3-8B (local; requires GPU)
  (d) Phi-3.5-mini (local, CPU-feasible)

When resolved, implement `_call_judge()` and remove the NotImplementedError.
Remember: sending the raw prompt to a cloud-hosted judge is itself a privacy
leak unless that judge runs in a TEE. Note the caveat in the paper.
"""
from __future__ import annotations

from time import perf_counter

from privategate.types import Mode

from ..harness.schemas import GoldItem, RouterDecision

JUDGE_PROMPT = """You are a privacy routing judge. Given USER_PROMPT, decide \
which privacy-preserving inference mode it should use:

- "plaintext": no privacy concern, safe to send as-is to a cloud LLM
- "ldp": contains personal / quasi-private information; use local differential privacy
- "he": contains strictly-regulated or never-releasable information; use homomorphic encryption
- "abstain": cannot confidently decide

Respond with ONLY the lowercase mode label.

USER_PROMPT:
<<<
{prompt}
>>>
MODE:"""

_LABEL_TO_MODE: dict[str, Mode] = {
    "plaintext": Mode.PLAINTEXT,
    "ldp": Mode.PERTURB,
    "he": Mode.HE_SMPC,
    "abstain": Mode.ABSTAIN,
}


class LLMJudge:
    """B5 — stubbed until D5 is resolved."""

    name = "llm_judge"

    def __init__(self, model: str = "TBD"):
        self.model = model

    def _call_judge(self, prompt: str) -> str:
        raise NotImplementedError(
            "D5 unresolved: pick a judge LLM (GPT-4o-mini / Claude Haiku / "
            "Llama-Guard-3-8B / Phi-3.5) and implement the API call here."
        )

    def route(self, item: GoldItem) -> RouterDecision:
        t0 = perf_counter()
        label = self._call_judge(item.prompt).strip().lower()
        mode = _LABEL_TO_MODE.get(label, Mode.ABSTAIN)
        return RouterDecision(
            item_id=item.id,
            mode=mode,
            latency_ms=(perf_counter() - t0) * 1000.0,
            rationale=f"llm_judge[{self.model}]={label}",
        )
