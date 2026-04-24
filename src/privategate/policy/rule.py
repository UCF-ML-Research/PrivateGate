from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Union

from ..types import Mode

RuleWhen = Union[str, Callable[..., bool]]


@dataclass(frozen=True)
class Rule:
    """Declarative routing rule.

    `when` can be either a CEL expression string (evaluated by a future
    CELPolicyEngine) or a Python callable `(Request, list[Signal]) -> bool`
    (used by the v1 reference policy). Rules are composed by `PolicyEngine`;
    conflict resolution is highest-priority-wins with stable ordering.
    """

    id: str
    when: Any  # RuleWhen — dataclass frozen + Union sometimes trips mypy; keep Any at runtime
    route: Mode
    priority: int = 0
    rationale: str = ""


@dataclass(frozen=True)
class RuleMatch:
    rule: Rule
    matched: bool
    reason: str
