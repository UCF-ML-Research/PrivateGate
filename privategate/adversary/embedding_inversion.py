"""Embedding-inversion attacker (plan §5.3).

A real attacker would: encode the transformed text with a public sentence
encoder, run a vec2text-style inverter on the embedding, and read off
the recovered tokens. We don't ship `vec2text`/`torch` in the default
install — and unit tests must stay hermetic — so this module accepts an
injected reconstructor function.

The default reconstructor is the **identity**: returns the transformed
text as the recovered text. This is intentionally honest: for short
queries vec2text recovers most of the input, so any plaintext that
remains in the outbound payload is fair game for the scorer. If a
defense (e.g. PrivateGate) has redacted a span, the identity
reconstructor will also fail to recover it — which is the right
accounting under the threat model.
"""
from __future__ import annotations

from typing import Callable, Optional

from privategate.adversary.base import Attacker, AttackResult

Reconstructor = Callable[[str], list[str]]


def _identity_reconstructor(text: str) -> list[str]:
    return [text] if text else []


class EmbeddingInverter(Attacker):
    name = "embedding_inversion"

    def __init__(self, reconstructor: Optional[Reconstructor] = None) -> None:
        self._reconstruct = reconstructor or _identity_reconstructor

    def attack(self, transformed: str) -> AttackResult:
        return AttackResult(attacker=self.name, predictions=list(self._reconstruct(transformed)))

    @classmethod
    def from_vec2text(cls, model_name: str = "ielab/vec2text-gtr-base") -> "EmbeddingInverter":
        """Build an inverter backed by a real `vec2text` model.

        Lazily imports the optional dependency so the rest of the package
        keeps working without it.
        """
        try:
            import vec2text  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "vec2text is not installed; install it to run real embedding inversion"
            ) from exc

        corrector = vec2text.load_pretrained_corrector(model_name)

        def _reconstruct(text: str) -> list[str]:
            recovered = vec2text.invert_strings(  # type: ignore[attr-defined]
                strings=[text],
                corrector=corrector,
            )
            return list(recovered)

        return cls(reconstructor=_reconstruct)
