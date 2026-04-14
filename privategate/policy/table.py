from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from privategate.types import Action, Category, RiskLevel


@dataclass(frozen=True)
class PolicyTable:
    rules: dict[Category, dict[RiskLevel, Action]]

    def lookup(self, category: Category, risk: RiskLevel) -> Action:
        cat_rules = self.rules.get(category)
        if cat_rules is None:
            return Action.SECURE_SLOT  # fail closed
        action = cat_rules.get(risk)
        if action is None:
            return Action.SECURE_SLOT  # fail closed
        return action


def load_policy(path: Path | str) -> PolicyTable:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "table" not in raw:
        raise ValueError("policy file must have a top-level 'table' key")
    parsed: dict[Category, dict[RiskLevel, Action]] = {}
    for cat_name, risk_map in raw["table"].items():
        category = Category(cat_name)
        parsed[category] = {
            RiskLevel(risk_name): Action(action_name)
            for risk_name, action_name in risk_map.items()
        }
    return PolicyTable(rules=parsed)


def load_default_policy() -> PolicyTable:
    return load_policy(Path(__file__).parent / "default_policy.yaml")


def load_strict_policy() -> PolicyTable:
    return load_policy(Path(__file__).parent / "strict_policy.yaml")


def load_permissive_policy() -> PolicyTable:
    return load_policy(Path(__file__).parent / "permissive_policy.yaml")
