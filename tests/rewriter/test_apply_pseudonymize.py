from faker import Faker

from privategate.rewriter.actions import apply_pseudonymize
from privategate.rewriter.placeholder_map import PlaceholderMap
from privategate.types import Category, RiskLevel, Span


def test_pseudonymize_email_returns_email():
    faker = Faker()
    Faker.seed(42)
    pmap = PlaceholderMap()
    s = Span(0, 9, "a@b.co", Category.IDENTIFIER, RiskLevel.MEDIUM)
    out = apply_pseudonymize(s, faker, pmap)
    assert "@" in out
    assert "a@b.co" in pmap.as_dict().values()


def test_pseudonymize_credit_card_returns_digits():
    faker = Faker()
    pmap = PlaceholderMap()
    s = Span(0, 19, "4111-1111-1111-1111", Category.FINANCIAL, RiskLevel.HIGH)
    out = apply_pseudonymize(s, faker, pmap)
    assert any(c.isdigit() for c in out)
