"""Abstractive compaction: an LLM rewrites the history into a summary that re-enters the
context in a configurable role. Covers RQ2 families (iii) generic and (iv) agent-style;
which one is decided by cfg.prompt_id."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from agentdojo.types import ChatMessage

from compaction.base import (
    CompactionConfig,
    CompactionResult,
    Compactor,
    assistant_blocks,
    insert_summary,
    last_k_turns,
    render_transcript,
    split_system,
    user_messages,
)

PROMPTS_DIR = Path(__file__).parent / "prompts"

# (system_prompt, user_text, max_tokens) -> summary text
SummarizeFn = Callable[[str, str, int], str]


def load_prompt(prompt_id: str) -> str:
    path = PROMPTS_DIR / f"{prompt_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"no compaction prompt {path}")
    lines = path.read_text().splitlines()
    # Lines starting with '#' are provenance notes for us, not part of the prompt.
    return "\n".join(l for l in lines if not l.startswith("#")).strip()


class AbstractiveCompactor(Compactor):
    name = "abstractive"

    def __init__(self, summarize: SummarizeFn, default_prompt: str, name: str = "abstractive") -> None:
        self.summarize = summarize
        self.default_prompt = default_prompt
        self.name = name

    def compact(self, messages: Sequence[ChatMessage], cfg: CompactionConfig) -> CompactionResult:
        system, rest = split_system(messages)

        tail = last_k_turns(rest, cfg.keep_last_k)
        to_summarize = rest[: len(rest) - len(tail)] if tail else rest

        prompt_id = cfg.prompt_id or self.default_prompt
        prompt = load_prompt(prompt_id)
        transcript = render_transcript(to_summarize)
        summary = self.summarize(prompt, transcript, cfg.max_summary_tokens).strip()

        preserved = user_messages(to_summarize) if cfg.keep_user_verbatim else []
        new_messages = insert_summary(system, preserved, summary, cfg.role, cfg.summary_header, tail)

        return CompactionResult(
            messages=new_messages,
            summary_text=summary,
            meta={
                "family": self.name,
                "prompt_id": prompt_id,
                "role": cfg.role,
                "n_in": len(messages),
                "n_out": len(new_messages),
                "n_turns_summarized": sum(1 for b in assistant_blocks(to_summarize) if b[0]["role"] == "assistant"),
                "transcript_chars": len(transcript),
            },
        )
