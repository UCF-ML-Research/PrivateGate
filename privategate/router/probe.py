"""Semantic-dependency probe (plan §5.2).

For a single span we build three variants of the query:

  - original: the user query verbatim
  - masked:   the span replaced with its category token (e.g. ``[MEDICAL]``)
  - abstract: the span replaced with its abstract-action superclass

We run the proxy model on each variant and measure how much the answer
diverges from the original. If divergence exceeds the threshold the span
is *semantic-critical*: removing it changes the task the model has to
perform, not merely the surface form of the answer. Such spans must be
routed to the secure path even when the policy table would have allowed
masking.

The proxy model is injected as a callable so unit tests don't need a
real LLM. Production use plugs in either the configured local model or
a cheap on-device proxy.
"""
from __future__ import annotations

from typing import Callable, Optional

from privategate.rewriter.actions import apply_abstract, apply_mask
from privategate.router.divergence import Embedder, answer_divergence
from privategate.types import ProbeResult, Span

ProxyModel = Callable[[str], str]


def _replace_span(text: str, span: Span, replacement: str) -> str:
    return text[: span.start] + replacement + text[span.end :]


def three_variant_probe(
    query: str,
    span: Span,
    proxy_model: ProxyModel,
    embedder: Optional[Embedder] = None,
    threshold: float = 0.3,
) -> ProbeResult:
    if span.start < 0 or span.end > len(query):
        raise ValueError("span offsets are outside the query")

    masked_query = _replace_span(query, span, apply_mask(span))
    abstract_query = _replace_span(query, span, apply_abstract(span))

    answer_orig = proxy_model(query)
    answer_masked = proxy_model(masked_query)
    answer_abstract = proxy_model(abstract_query)

    div_masked = answer_divergence(answer_orig, answer_masked, embedder=embedder)
    div_abstract = answer_divergence(answer_orig, answer_abstract, embedder=embedder)
    divergence = max(div_masked, div_abstract)

    return ProbeResult(
        span=span,
        divergence=divergence,
        semantic_critical=divergence > threshold,
    )
