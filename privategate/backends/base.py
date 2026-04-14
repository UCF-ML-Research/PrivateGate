from __future__ import annotations

from abc import ABC, abstractmethod


class Backend(ABC):
    name: str = "backend"

    @abstractmethod
    def complete(self, prompt: str) -> str:
        ...
