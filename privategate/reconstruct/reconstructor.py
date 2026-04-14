from __future__ import annotations

from privategate.reconstruct.fuzzy_resolver import resolve_paraphrased
from privategate.reconstruct.string_resolver import resolve_verbatim


class Reconstructor:
    def reconstruct(self, response: str, placeholder_map: dict[str, str]) -> str:
        step1 = resolve_verbatim(response, placeholder_map)
        step2 = resolve_paraphrased(step1, placeholder_map)
        return step2
