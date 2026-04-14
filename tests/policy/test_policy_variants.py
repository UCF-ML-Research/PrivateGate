from privategate.policy.table import (
    load_default_policy,
    load_permissive_policy,
    load_strict_policy,
)
from privategate.types import Action, Category, RiskLevel


def test_strict_table_routes_everything_to_secure():
    table = load_strict_policy()
    for cat in Category:
        for risk in RiskLevel:
            assert table.lookup(cat, risk) is Action.SECURE_SLOT


def test_permissive_keeps_personal_context():
    table = load_permissive_policy()
    assert table.lookup(Category.PERSONAL_CONTEXT, RiskLevel.HIGH) is Action.KEEP


def test_permissive_still_locks_critical_credentials():
    table = load_permissive_policy()
    assert table.lookup(Category.CREDENTIAL, RiskLevel.CRITICAL) is Action.SECURE_SLOT


def test_default_table_unchanged():
    table = load_default_policy()
    assert table.lookup(Category.IDENTIFIER, RiskLevel.HIGH) is Action.PSEUDONYMIZE
