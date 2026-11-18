"""Truncation / eviction (RQ2 family i, negative control): no rewrite, roles preserved.
Keeps the system prompt, the first user message and the last k turns; drops the rest.

Note: with the default m = 0 trigger the poisoned tool result is inside the last turn, so
truncation keeps it verbatim. That is intended: this family tests "no rewrite, no re-trust".
"""

from __future__ import annotations

from collections.abc import Sequence

from agentdojo.types import ChatMessage

from compaction.base import CompactionConfig, CompactionResult, Compactor, last_k_turns, split_system


class TruncationCompactor(Compactor):
    name = "truncation"

    def compact(self, messages: Sequence[ChatMessage], cfg: CompactionConfig) -> CompactionResult:
        k = cfg.keep_last_k if cfg.keep_last_k > 0 else 2
        system, rest = split_system(messages)
        first_user = next((m for m in rest if m["role"] == "user"), None)
        tail = last_k_turns(rest, k)

        out: list[ChatMessage] = []
        if system is not None:
            out.append(system)
        if first_user is not None and not any(m is first_user for m in tail):
            out.append(first_user)
        out.extend(tail)

        return CompactionResult(
            messages=out,
            summary_text=None,
            meta={"family": self.name, "keep_last_k": k, "n_in": len(messages), "n_out": len(out)},
        )
