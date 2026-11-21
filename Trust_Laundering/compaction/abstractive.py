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
    # The leading block of '#' lines is provenance for us, not prompt text. Only the leading block:
    # ported prompts contain Markdown headings ('## Objective') that must survive.
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    return "\n".join(lines[i:]).strip()


# How a ported harness wraps its summary when reinserting it (summary -> message body).
Wrapper = Callable[[str], str]


def codex_wrapper(summary: str) -> str:
    # codex-rs/core/src/compact.rs: format!("{SUMMARY_PREFIX}\n{summary_suffix}")
    return f"{load_prompt('codex_summary_prefix')}\n{summary}"


def opencode_wrapper(summary: str) -> str:
    # packages/core/src/session/compaction.ts: checkpoint sent as one user message (no recent-context here)
    return f"<conversation-checkpoint>\n<summary>{summary}</summary>\n</conversation-checkpoint>"


class AbstractiveCompactor(Compactor):
    name = "abstractive"

    def __init__(
        self,
        summarize: SummarizeFn,
        default_prompt: str,
        name: str = "abstractive",
        wrapper: Wrapper | None = None,
        min_summary_tokens: int = 0,
    ) -> None:
        """wrapper: ported harnesses put their own framing around the summary instead of
        cfg.summary_header. min_summary_tokens: long structured templates get cut off at the
        plan's 512-token default, so ported prompts raise the floor."""
        self.summarize = summarize
        self.default_prompt = default_prompt
        self.name = name
        self.wrapper = wrapper
        self.min_summary_tokens = min_summary_tokens

    def compact(self, messages: Sequence[ChatMessage], cfg: CompactionConfig) -> CompactionResult:
        system, rest = split_system(messages)

        tail = last_k_turns(rest, cfg.keep_last_k)
        to_summarize = rest[: len(rest) - len(tail)] if tail else rest

        prompt_id = cfg.prompt_id or self.default_prompt
        prompt = load_prompt(prompt_id)
        transcript = render_transcript(to_summarize)
        max_tokens = max(cfg.max_summary_tokens, self.min_summary_tokens)
        summary = self.summarize(prompt, transcript, max_tokens).strip()

        preserved = user_messages(to_summarize) if cfg.keep_user_verbatim else []
        if self.wrapper is not None:
            body, header = self.wrapper(summary), ""
        else:
            body, header = summary, cfg.summary_header
        new_messages = insert_summary(system, preserved, body, cfg.role, header, tail)

        return CompactionResult(
            messages=new_messages,
            summary_text=summary,
            meta={
                "family": self.name,
                "prompt_id": prompt_id,
                "role": cfg.role,
                "max_summary_tokens": max_tokens,
                "n_in": len(messages),
                "n_out": len(new_messages),
                "n_turns_summarized": sum(1 for b in assistant_blocks(to_summarize) if b[0]["role"] == "assistant"),
                "transcript_chars": len(transcript),
            },
        )
