"""Compactor interface shared by every compaction family and every RQ3 defense.

Contract (agreed Day 1):
    compact(messages, cfg) -> CompactionResult(messages, summary_text, meta)

`messages` is an AgentDojo message list (list of ChatMessage TypedDicts). A compactor
must return a list that is valid to send to an OpenAI-style chat endpoint:
  * at most one system message, and it comes first;
  * every `tool` message directly follows the assistant message whose tool call it answers.
Use the helpers in this module (`insert_summary`, `assistant_blocks`) to keep that true.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)

SummaryRole = Literal["user", "system", "assistant", "tool"]

SUMMARY_TOOL_NAME = "get_history_summary"
SUMMARY_TOOL_CALL_ID = "call_history_summary"


@dataclass(frozen=True)
class CompactionConfig:
    family: str = "abstractive_agent"
    """Registry name of the compactor (see compaction/__init__.py)."""
    role: SummaryRole = "user"
    """Role the summary re-enters the context in (abstractive families only)."""
    keep_user_verbatim: bool = False
    """Re-insert the user's original messages, unchanged, before the summary."""
    keep_last_k: int = 0
    """Abstractive: keep the last k turns verbatim after the summary.
    Truncation: number of turns to keep. A turn is one assistant message plus its tool results."""
    prompt_id: str | None = None
    """File name (without .txt) in compaction/prompts/. None = the family's default prompt."""
    max_summary_tokens: int = 512
    summary_header: str = "Summary of the conversation so far:"
    """Text placed before the summary when it is inserted."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CompactionConfig:
        return cls(**d)


@dataclass
class CompactionResult:
    messages: list[ChatMessage]
    summary_text: str | None
    """The text the compactor wrote (None for families that do not write one, e.g. truncation)."""
    meta: dict[str, Any] = field(default_factory=dict)
    """Free-form diagnostics: message counts, tokens, prompt id, ..."""


class Compactor(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def compact(self, messages: Sequence[ChatMessage], cfg: CompactionConfig) -> CompactionResult: ...


# --------------------------------------------------------------------------- helpers


def text_of(message: ChatMessage) -> str:
    content = message.get("content")
    if not content:
        return ""
    return get_text_content_as_str(content)


def split_system(messages: Sequence[ChatMessage]) -> tuple[ChatSystemMessage | None, list[ChatMessage]]:
    """Return (first system message or None, every other message)."""
    if messages and messages[0]["role"] == "system":
        return messages[0], list(messages[1:])  # type: ignore[return-value]
    return None, list(messages)


def user_messages(messages: Sequence[ChatMessage]) -> list[ChatUserMessage]:
    return [m for m in messages if m["role"] == "user"]  # type: ignore[misc]


def assistant_blocks(messages: Sequence[ChatMessage]) -> list[list[ChatMessage]]:
    """Group a system-free message list into blocks that each start at an assistant message
    (together with the tool results that follow it). Messages before the first assistant
    message form block 0. Dropping or keeping whole blocks never orphans a tool result."""
    blocks: list[list[ChatMessage]] = [[]]
    for m in messages:
        if m["role"] == "assistant":
            blocks.append([m])
        else:
            blocks[-1].append(m)
    return [b for b in blocks if b]


def last_k_turns(messages: Sequence[ChatMessage], k: int) -> list[ChatMessage]:
    """The last k assistant turns (assistant message + its tool results), pairing-safe."""
    if k <= 0:
        return []
    turns = [b for b in assistant_blocks(messages) if b[0]["role"] == "assistant"]
    return [m for b in turns[-k:] for m in b]


def render_transcript(messages: Sequence[ChatMessage], include_system: bool = False) -> str:
    """Plain-text transcript for a summarizer. Role tags are kept so the summarizer *could*
    track provenance; whether it does is part of what we measure."""
    lines: list[str] = []
    for m in messages:
        role = m["role"]
        if role == "system" and not include_system:
            continue
        if role == "assistant":
            text = text_of(m)
            if text:
                lines.append(f"[assistant]\n{text}")
            for tc in m.get("tool_calls") or []:
                lines.append(f"[assistant called tool] {tc.function}({json.dumps(dict(tc.args), default=str)})")
        elif role == "tool":
            name = m["tool_call"].function if m.get("tool_call") else "unknown"
            body = m.get("error") or text_of(m)
            lines.append(f"[tool result: {name}]\n{body}")
        else:
            lines.append(f"[{role}]\n{text_of(m)}")
    return "\n\n".join(lines)


def insert_summary(
    system: ChatSystemMessage | None,
    preserved_users: Sequence[ChatUserMessage],
    summary: str,
    role: SummaryRole,
    header: str,
    tail: Sequence[ChatMessage] = (),
) -> list[ChatMessage]:
    """Build [system, *preserved_users, <summary in role>, *tail] with valid pairing.

    role="system" appends the summary to the existing system prompt (Llama/Qwen templates
    expect exactly one system message). role="tool" uses a synthetic get_history_summary call.
    """
    body = f"{header}\n\n{summary}" if header else summary
    out: list[ChatMessage] = []

    if role == "system":
        base = text_of(system) if system is not None else ""
        merged = f"{base}\n\n{body}" if base else body
        out.append(ChatSystemMessage(role="system", content=[text_content_block_from_string(merged)]))
        out.extend(preserved_users)
    else:
        if system is not None:
            out.append(system)
        out.extend(preserved_users)
        if role == "user":
            out.append(ChatUserMessage(role="user", content=[text_content_block_from_string(body)]))
        elif role == "assistant":
            out.append(
                ChatAssistantMessage(role="assistant", content=[text_content_block_from_string(body)], tool_calls=None)
            )
        elif role == "tool":
            call = FunctionCall(function=SUMMARY_TOOL_NAME, args={}, id=SUMMARY_TOOL_CALL_ID)
            out.append(ChatAssistantMessage(role="assistant", content=None, tool_calls=[call]))
            out.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(body)],
                    tool_call_id=SUMMARY_TOOL_CALL_ID,
                    tool_call=call,
                    error=None,
                )
            )
        else:
            raise ValueError(f"unknown summary role {role!r}")

    out.extend(tail)
    return out


def check_well_formed(messages: Sequence[ChatMessage]) -> None:
    """Raise ValueError if the list breaks the system-first or tool-pairing rules."""
    for i, m in enumerate(messages):
        if m["role"] == "system" and i != 0:
            raise ValueError(f"system message at position {i}; must be first and unique")
    open_ids: set[str | None] = set()
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            open_ids = {tc.id for tc in (m.get("tool_calls") or [])}
        elif m["role"] == "tool":
            if m.get("tool_call_id") not in open_ids:
                raise ValueError(f"tool message at {i} (id={m.get('tool_call_id')}) has no matching assistant call")
        else:
            open_ids = set()
