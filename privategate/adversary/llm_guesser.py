"""LLM-based guesser attacker (plan §5.3).

Builds a reconstruction prompt and asks an LLM to recover any sensitive
content from the transformed query. The model is injected so the unit
tests don't need network or weights; production plugs in the configured
adversarial model.

A deterministic ``naive_pattern_model`` is shipped as the default so the
runner is usable end-to-end without any external model. It mimics the
easy wins a real attacker gets from a transformed payload: extract any
literal digit groups, email-like strings, and obvious credential prefixes
that survived the rewrite. If a defense actually redacted those, this
attacker recovers nothing — which is the desired property.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from privategate.adversary.base import Attacker, AttackResult

ModelFn = Callable[[str], str]


_DIGIT_GROUP_RE = re.compile(r"\b[\d][\d\- ]{4,}[\d]\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CREDENTIAL_RE = re.compile(r"\b(?:sk|pk|api|key|token)[-_][A-Za-z0-9]{8,}\b", re.IGNORECASE)


def naive_pattern_model(prompt: str) -> str:
    """A deterministic stand-in for an LLM attacker.

    Looks at whatever text the prompt asked the model to attack and
    returns one guess per line. Real LLMs do this and quite a bit more;
    the point of the stub is to give the runner a working default that
    fails on properly-redacted text and succeeds on leaks.
    """
    target = _extract_target(prompt)
    guesses: list[str] = []
    guesses.extend(m.group(0) for m in _CREDENTIAL_RE.finditer(target))
    guesses.extend(m.group(0) for m in _EMAIL_RE.finditer(target))
    guesses.extend(m.group(0) for m in _DIGIT_GROUP_RE.finditer(target))
    return "\n".join(guesses)


def _extract_target(prompt: str) -> str:
    marker = "TRANSFORMED:"
    if marker in prompt:
        return prompt.split(marker, 1)[1]
    return prompt


def _build_prompt(transformed: str) -> str:
    return (
        "You are auditing a redacted message for any sensitive information "
        "that may have been left in. List one guess per line.\n\n"
        f"TRANSFORMED:{transformed}\n"
    )


def _parse(response: str) -> list[str]:
    return [line.strip() for line in response.splitlines() if line.strip()]


class LLMGuesser(Attacker):
    name = "llm_guesser"

    def __init__(self, model_fn: Optional[ModelFn] = None) -> None:
        self._model = model_fn or naive_pattern_model

    def attack(self, transformed: str) -> AttackResult:
        prompt = _build_prompt(transformed)
        response = self._model(prompt)
        return AttackResult(attacker=self.name, predictions=_parse(response))
