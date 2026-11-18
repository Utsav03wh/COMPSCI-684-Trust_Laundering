"""Compactor registry. Every family and every RQ3 defense registers here under a name that
configs refer to as `family:`. Defenses (defenses/) register themselves the same way."""

from __future__ import annotations

from collections.abc import Callable

from compaction.abstractive import AbstractiveCompactor, SummarizeFn
from compaction.base import CompactionConfig, CompactionResult, Compactor
from compaction.truncation import TruncationCompactor

# name -> factory(summarize_fn) -> Compactor
REGISTRY: dict[str, Callable[[SummarizeFn | None], Compactor]] = {
    "truncation": lambda _s: TruncationCompactor(),
    "abstractive_generic": lambda s: AbstractiveCompactor(_need(s), "generic", "abstractive_generic"),
    "abstractive_agent": lambda s: AbstractiveCompactor(_need(s), "agent_style", "abstractive_agent"),
    # TODO(Utsav): "extractive" (LLMLingua-2, rate 0.5, tool/assistant text only)
    # TODO(Tanisha): "isolated", "ppc" from defenses/
}


def _need(s: SummarizeFn | None) -> SummarizeFn:
    if s is None:
        raise ValueError("this compactor needs a summarize function")
    return s


def register(name: str, factory: Callable[[SummarizeFn | None], Compactor]) -> None:
    if name in REGISTRY:
        raise ValueError(f"compactor {name!r} already registered")
    REGISTRY[name] = factory


def build_compactor(cfg: CompactionConfig, summarize: SummarizeFn | None = None) -> Compactor:
    if cfg.family not in REGISTRY:
        raise KeyError(f"unknown compaction family {cfg.family!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[cfg.family](summarize)


__all__ = [
    "CompactionConfig",
    "CompactionResult",
    "Compactor",
    "REGISTRY",
    "build_compactor",
    "register",
]
