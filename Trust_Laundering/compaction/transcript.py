"""Verbatim-transcript control: no LLM, no rewriting.

The history is flattened into the same text transcript the summarizer reads and inserted
exactly where an abstractive summary would go (same role, same header). Compared with an
abstractive family on the same pairs, this separates
    "the summarizer rewrote the injection into the user's task list"   (abstractive only)
from
    "the history was reformatted and moved into a new message/role"   (both).
Note the raw injection text is still present here, now inside that message.
Added after the audit showed truncation removes nothing on short AgentDojo histories.
"""

from __future__ import annotations

from collections.abc import Sequence

from agentdojo.types import ChatMessage

from compaction.base import (
    CompactionConfig,
    CompactionResult,
    Compactor,
    insert_summary,
    last_k_turns,
    render_transcript,
    split_system,
    user_messages,
)


class TranscriptCompactor(Compactor):
    name = "transcript"

    def compact(self, messages: Sequence[ChatMessage], cfg: CompactionConfig) -> CompactionResult:
        system, rest = split_system(messages)
        tail = last_k_turns(rest, cfg.keep_last_k)
        to_flatten = rest[: len(rest) - len(tail)] if tail else rest
        transcript = render_transcript(to_flatten)
        preserved = user_messages(to_flatten) if cfg.keep_user_verbatim else []
        new_messages = insert_summary(system, preserved, transcript, cfg.role, cfg.summary_header, tail)
        return CompactionResult(
            messages=new_messages,
            summary_text=transcript,  # so marker / payload metrics apply unchanged
            meta={"family": self.name, "role": cfg.role, "n_in": len(messages), "n_out": len(new_messages),
                  "transcript_chars": len(transcript)},
        )
