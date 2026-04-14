"""ML span detector.

Designed so the underlying tagger is *injected*. This keeps unit tests
hermetic (no model download in CI) while letting a real HuggingFace
pipeline plug in unchanged via `from_huggingface(...)`.

A `Tagger` is any callable `text -> list[TaggedEntity]`. `TaggedEntity`
mirrors the dict that `transformers.pipeline("ner", aggregation_strategy=
"simple")` returns: ``{"start": int, "end": int, "word": str,
"entity_group": str, "score": float}``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional

from privategate.types import Category, RiskLevel, Span


@dataclass(frozen=True)
class TaggedEntity:
    start: int
    end: int
    word: str
    entity_group: str
    score: float = 1.0


Tagger = Callable[[str], Iterable[TaggedEntity]]


# Default mapping from common HF NER label sets onto our categories.
DEFAULT_LABEL_MAP: Mapping[str, Category] = {
    "PER": Category.IDENTIFIER,
    "PERSON": Category.IDENTIFIER,
    "LOC": Category.PERSONAL_CONTEXT,
    "LOCATION": Category.PERSONAL_CONTEXT,
    "GPE": Category.PERSONAL_CONTEXT,
    "ORG": Category.PERSONAL_CONTEXT,
    "ORGANIZATION": Category.PERSONAL_CONTEXT,
    "MISC": Category.PERSONAL_CONTEXT,
    "DATE": Category.IDENTIFIER,
    "EMAIL": Category.IDENTIFIER,
    "PHONE": Category.IDENTIFIER,
    "ID": Category.IDENTIFIER,
    "MEDICAL": Category.MEDICAL,
    "DISEASE": Category.MEDICAL,
    "CONDITION": Category.MEDICAL,
    "MONEY": Category.FINANCIAL,
    "CARD": Category.FINANCIAL,
    "CREDENTIAL": Category.CREDENTIAL,
    "TOKEN": Category.CREDENTIAL,
}

_DEFAULT_RISK: Mapping[Category, RiskLevel] = {
    Category.IDENTIFIER: RiskLevel.HIGH,
    Category.MEDICAL: RiskLevel.HIGH,
    Category.FINANCIAL: RiskLevel.HIGH,
    Category.CREDENTIAL: RiskLevel.CRITICAL,
    Category.PERSONAL_CONTEXT: RiskLevel.MEDIUM,
}


class MLDetector:
    def __init__(
        self,
        tagger: Tagger,
        label_map: Optional[Mapping[str, Category]] = None,
        score_threshold: float = 0.5,
    ) -> None:
        self._tagger = tagger
        self._label_map = dict(label_map) if label_map is not None else dict(DEFAULT_LABEL_MAP)
        self._threshold = score_threshold

    def detect(self, text: str) -> list[Span]:
        if not text:
            return []
        out: list[Span] = []
        for ent in self._tagger(text):
            if ent.score < self._threshold:
                continue
            label = ent.entity_group.upper()
            category = self._label_map.get(label)
            if category is None:
                continue
            risk = _DEFAULT_RISK[category]
            if ent.end <= ent.start or not text[ent.start:ent.end]:
                continue
            out.append(
                Span(
                    start=ent.start,
                    end=ent.end,
                    text=text[ent.start:ent.end],
                    category=category,
                    risk=risk,
                    source="ml",
                )
            )
        return out

    @classmethod
    def from_huggingface(
        cls,
        model_name: str = "dslim/bert-base-NER",
        score_threshold: float = 0.5,
        label_map: Optional[Mapping[str, Category]] = None,
    ) -> "MLDetector":
        """Build an MLDetector backed by a real HF NER pipeline.

        Imports ``transformers`` lazily so the rest of the package can be
        used (and tested) without the optional ML dependency installed.
        """
        from transformers import pipeline  # type: ignore

        ner = pipeline("ner", model=model_name, aggregation_strategy="simple")

        def _tag(text: str) -> list[TaggedEntity]:
            return [
                TaggedEntity(
                    start=int(e["start"]),
                    end=int(e["end"]),
                    word=str(e["word"]),
                    entity_group=str(e["entity_group"]),
                    score=float(e.get("score", 1.0)),
                )
                for e in ner(text)
            ]

        return cls(tagger=_tag, label_map=label_map, score_threshold=score_threshold)
