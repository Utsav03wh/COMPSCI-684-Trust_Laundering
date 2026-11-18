"""CompactionElement: sits inside ToolsExecutionLoop between ToolsExecutor and the LLM.

    ToolsExecutionLoop([ToolsExecutor(), CompactionElement(...), llm])

It watches each batch of tool results for the injection (see trigger.py). Once the trigger
has been seen and `m` further tool batches have run, it either
  * mode="compact": replaces the message list with the compactor's output, or
  * mode="fork":    raises ForkSignal with the current state, so the runner can snapshot and
                    run both branches from an identical prefix (see fork.py).
It fires at most once per run. State lives in `extra_args["compaction"]` (a fresh dict is
returned each time; AgentDojo's default `extra_args={}` is a shared mutable default).

The full, uncompacted history is kept in that state: AgentDojo rebuilds the tool-call trace
for scoring from the final message list, which compaction would otherwise truncate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage

from compaction.base import CompactionConfig, Compactor, check_well_formed
from pipeline.trigger import poisoned

STATE_KEY = "compaction"
TRIGGER_KEYS = "trigger_keys"  # extra_args key: normalized payload lines for this run


class ForkSignal(Exception):
    def __init__(self, query: str, runtime: FunctionsRuntime, env: Env, messages: Sequence[ChatMessage], extra_args: dict):
        super().__init__("fork point reached")
        self.query, self.runtime, self.env, self.messages, self.extra_args = query, runtime, env, list(messages), extra_args


def _n_tool_batches(messages: Sequence[ChatMessage]) -> int:
    return sum(1 for m in messages if m["role"] == "assistant" and m.get("tool_calls"))


def apply_compaction(
    messages: Sequence[ChatMessage], compactor: Compactor, cfg: CompactionConfig, state: dict[str, Any]
) -> tuple[list[ChatMessage], dict[str, Any]]:
    """Run the compactor and record what it did. Shared by the in-loop and fork paths."""
    result = compactor.compact(messages, cfg)
    check_well_formed(result.messages)
    new_state = {
        **state,
        "fired": True,
        "compaction_step": _n_tool_batches(messages),
        "summary_text": result.summary_text,
        "meta": result.meta,
        "pre_messages": list(messages),
        "n_compacted": len(result.messages),
    }
    return result.messages, new_state


class CompactionElement(BasePipelineElement):
    def __init__(
        self,
        compactor: Compactor | None,
        cfg: CompactionConfig | None,
        m: int = 0,
        mode: Literal["compact", "fork", "observe"] = "compact",
    ) -> None:
        if mode == "compact" and (compactor is None or cfg is None):
            raise ValueError("mode='compact' needs a compactor and a config")
        self.compactor, self.cfg, self.m, self.mode = compactor, cfg, m, mode

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        state = dict(extra_args.get(STATE_KEY) or {})
        if state.get("fired") or state.get("forked"):
            return query, runtime, env, messages, extra_args

        if "trigger_batch" not in state and poisoned(messages, extra_args.get(TRIGGER_KEYS, [])):
            state["trigger_batch"] = _n_tool_batches(messages)

        if "trigger_batch" in state and _n_tool_batches(messages) - state["trigger_batch"] >= self.m:
            if self.mode == "fork":
                state["forked"] = True
                raise ForkSignal(query, runtime, env, messages, {**extra_args, STATE_KEY: state})
            if self.mode == "compact":
                assert self.compactor is not None and self.cfg is not None
                messages, state = apply_compaction(messages, self.compactor, self.cfg, state)
            else:  # observe: record the trigger point only
                state["observed_step"] = _n_tool_batches(messages)

        return query, runtime, env, messages, {**extra_args, STATE_KEY: state}


def full_trace(final_messages: Sequence[ChatMessage], extra_args: dict) -> list[ChatMessage]:
    """The uncompacted history: messages before compaction + everything generated after it."""
    state = extra_args.get(STATE_KEY) or {}
    if not state.get("fired"):
        return list(final_messages)
    return [*state["pre_messages"], *final_messages[state["n_compacted"]:]]
