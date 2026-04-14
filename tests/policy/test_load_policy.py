from privategate.policy.table import load_default_policy
from privategate.types import Action, Category, RiskLevel


def test_default_policy_loads():
    table = load_default_policy()
    assert table.lookup(Category.CREDENTIAL, RiskLevel.LOW) is Action.MASK
    assert table.lookup(Category.CREDENTIAL, RiskLevel.CRITICAL) is Action.SECURE_SLOT
    assert table.lookup(Category.IDENTIFIER, RiskLevel.HIGH) is Action.PSEUDONYMIZE
    assert table.lookup(Category.MEDICAL, RiskLevel.HIGH) is Action.ABSTRACT


def test_unknown_category_fails_closed():
    table = load_default_policy()
    # use a fabricated risk that isn't in the table by removing the entry
    rules = {k: dict(v) for k, v in table.rules.items()}
    del rules[Category.IDENTIFIER][RiskLevel.LOW]
    from privategate.policy.table import PolicyTable
    t2 = PolicyTable(rules=rules)
    assert t2.lookup(Category.IDENTIFIER, RiskLevel.LOW) is Action.SECURE_SLOT
