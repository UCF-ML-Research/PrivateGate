"""Common attacker interface.

Every attacker takes a string (the **outbound payload** that left the
client and is therefore visible to the server / adversary) and returns a
list of candidate plaintext predictions. The recovery scorer (see
`scorer.py`) decides which gold spans are considered recovered.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AttackResult:
    attacker: str
    predictions: list[str] = field(default_factory=list)


class Attacker(ABC):
    name: str = "attacker"

    @abstractmethod
    def attack(self, transformed: str) -> AttackResult:
        ...
