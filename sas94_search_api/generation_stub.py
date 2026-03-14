from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    model: str | None = None
    temperature: float = 0.1
    insecure: bool = False
