from privategate.rewriter.actions import apply_mask
from privategate.types import Category, RiskLevel, Span


def test_mask_uses_category_token():
    s = Span(0, 11, "123-45-6789", Category.IDENTIFIER, RiskLevel.HIGH)
    assert apply_mask(s) == "[IDENTIFIER]"
