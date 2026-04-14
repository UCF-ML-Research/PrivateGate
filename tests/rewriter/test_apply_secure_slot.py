from privategate.rewriter.actions import apply_secure_slot
from privategate.rewriter.placeholder_map import PlaceholderMap
from privategate.types import Category, RiskLevel, Span


def test_secure_slot_returns_token_and_records_original():
    pmap = PlaceholderMap()
    s = Span(0, 19, "sk-abcdefghij1234567", Category.CREDENTIAL, RiskLevel.CRITICAL)
    token = apply_secure_slot(s, pmap)
    assert token.startswith("[SLOT_") and token.endswith("]")
    assert pmap.as_dict()[token] == "sk-abcdefghij1234567"
