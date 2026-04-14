from __future__ import annotations


def resolve_verbatim(response: str, placeholder_map: dict[str, str]) -> str:
    """Replace verbatim placeholder tokens with their original values.

    The client controls which fields are revealed back to the user; this
    function intentionally does not redact anything — that decision belongs
    to the higher-level Reconstructor.
    """
    out = response
    for token, original in placeholder_map.items():
        out = out.replace(token, original)
    return out
