"""Detect the poisoned tool result.

AgentDojo formats tool results as YAML, which re-wraps and escapes multi-line strings, so a
full injection payload never appears verbatim in a tool message (checked on banking and
workspace, agentdojo 0.1.35). We therefore match individual payload lines after normalizing
both sides to lowercase alphanumerics, which survives YAML wrapping, quoting and escaping.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from agentdojo.types import ChatMessage, get_text_content_as_str

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
MIN_LINE_CHARS = 24  # ignore short lines like "Signed," that could match benign data


def normalize(text: str) -> str:
    return _NON_ALNUM.sub("", text.lower())


def payload_keys(payloads: Iterable[str]) -> list[str]:
    """Normalized distinctive lines of each injection payload."""
    keys: set[str] = set()
    for p in payloads:
        mine = {n for n in (normalize(line) for line in p.splitlines()) if len(n) >= MIN_LINE_CHARS}
        if not mine and normalize(p):
            mine = {normalize(p)}  # short single-line payload: use it whole
        keys |= mine
    return sorted(keys)


def contains_injection(text: str, keys: Sequence[str]) -> bool:
    if not keys:
        return False
    n = normalize(text)
    return any(k in n for k in keys)


def new_tool_results(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Tool messages after the last assistant message (the batch just executed)."""
    out: list[ChatMessage] = []
    for m in reversed(messages):
        if m["role"] == "tool":
            out.append(m)
        else:
            break
    return list(reversed(out))


def poisoned(messages: Sequence[ChatMessage], keys: Sequence[str]) -> bool:
    for m in new_tool_results(messages):
        if contains_injection(get_text_content_as_str(m.get("content") or []), keys):
            return True
    return False
