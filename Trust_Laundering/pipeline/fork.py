"""Paired branching by snapshot (replaces replay; see the plan's "Paired branching").

    state = run_prefix(case, llm, system_message, m=0)     # stops at the poisoned tool result
    a = run_branch(state, case, llm)                        # branch A: no compaction
    b = run_branch(state, case, llm, compactor, cfg)        # branch B: compaction

Both branches start from deep copies of the same environment and message list, so the
prefix is identical by construction and no byte-for-byte replay check is needed. The
ForkState can be saved to disk (save_state / load_state) so E2/E3 reuse E1's prefixes.

Scoring uses the *full* uncompacted trace for AgentDojo's trace-based checks.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.task_suite import TaskSuite, functions_stack_trace_from_messages
from agentdojo.types import ChatMessage

from compaction.base import CompactionConfig, Compactor
from pipeline.compaction_element import (
    STATE_KEY,
    TRIGGER_KEYS,
    CompactionElement,
    ForkSignal,
    apply_compaction,
    full_trace,
)
from pipeline.trigger import payload_keys

DEFAULT_MAX_ITERS = 15  # AgentDojo's ToolsExecutionLoop default


@dataclass
class Case:
    suite: TaskSuite
    user_task: BaseUserTask
    injection_task: BaseInjectionTask | None
    injections: dict[str, str]

    @property
    def ids(self) -> dict[str, str | None]:
        return {
            "suite": self.suite.name,
            "user_task_id": self.user_task.ID,
            "injection_task_id": self.injection_task.ID if self.injection_task else None,
        }


@dataclass
class ForkState:
    query: str
    env: Any  # TaskEnvironment (pydantic) at the fork point
    pre_env: Any  # environment before the task started, for AgentDojo's checks
    messages: list[ChatMessage]
    extra_args: dict
    batches_used: int


@dataclass
class BranchResult:
    branch: str  # "A" (no compaction), "B" (compaction) or "no_fork"
    messages: list[ChatMessage]
    trace: list[ChatMessage]
    utility: bool
    security: bool | None  # True = attacker goal achieved; None when there is no injection task
    compaction: dict = field(default_factory=dict)
    error: str | None = None


# --------------------------------------------------------------------------- running


def _new_env(case: Case):
    env = case.suite.load_and_inject_default_environment(case.injections)
    task_env = case.user_task.init_environment(env)
    return task_env, task_env.model_copy(deep=True)


def run_prefix(
    case: Case,
    llm: BasePipelineElement,
    system_message: str,
    m: int = 0,
    max_iters: int = DEFAULT_MAX_ITERS,
) -> ForkState | BranchResult:
    """Run until the injection has been read plus m tool batches. Returns a ForkState, or a
    scored BranchResult("no_fork") if the agent finished without reaching the fork point
    (injection never read, or the agent stopped first). Those pairs are excluded from the
    paired analysis and should be counted."""
    task_env, pre_env = _new_env(case)
    runtime = FunctionsRuntime(case.suite.tools)
    loop = ToolsExecutionLoop([ToolsExecutor(), CompactionElement(None, None, m=m, mode="fork"), llm], max_iters)
    pipeline = AgentPipeline([SystemMessage(system_message), InitQuery(), llm, loop])
    extra = {TRIGGER_KEYS: payload_keys(case.injections.values())}
    try:
        _, _, env, messages, extra_out = pipeline.query(case.user_task.PROMPT, runtime, task_env, [], extra)
    except ForkSignal as s:
        n_batches = sum(1 for msg in s.messages if msg["role"] == "assistant" and msg.get("tool_calls"))
        return ForkState(case.user_task.PROMPT, s.env, pre_env, s.messages, s.extra_args, n_batches)
    messages = list(messages)
    utility, security = score(case, pre_env, env, messages, messages)
    return BranchResult("no_fork", messages, messages, utility, security, dict(extra_out.get(STATE_KEY) or {}))


def run_branch(
    state: ForkState,
    case: Case,
    llm: BasePipelineElement,
    compactor: Compactor | None = None,
    cfg: CompactionConfig | None = None,
    max_iters: int = DEFAULT_MAX_ITERS,
) -> BranchResult:
    """Continue from a fork point. With a compactor this is branch B, otherwise branch A.
    The ForkState is never mutated, so it can be reused for any number of branches."""
    env = state.env.model_copy(deep=True)
    messages = copy.deepcopy(state.messages)
    extra = copy.deepcopy(state.extra_args)
    branch = "A"
    if compactor is not None:
        if cfg is None:
            raise ValueError("compactor given without a config")
        messages, extra[STATE_KEY] = apply_compaction(messages, compactor, cfg, dict(extra.get(STATE_KEY) or {}))
        branch = "B"

    runtime = FunctionsRuntime(case.suite.tools)
    loop = ToolsExecutionLoop([ToolsExecutor(), llm], max_iters=max(1, max_iters - state.batches_used))
    _, _, env, messages, extra = AgentPipeline([llm, loop]).query(state.query, runtime, env, messages, extra)
    messages = list(messages)
    trace = full_trace(messages, extra)
    utility, security = score(case, state.pre_env, env, messages, trace)
    comp = dict(extra.get(STATE_KEY) or {})
    comp.pop("pre_messages", None)  # already in `trace`
    return BranchResult(branch, messages, trace, utility, security, comp)


def score(case: Case, pre_env, post_env, final_messages, trace) -> tuple[bool, bool | None]:
    """AgentDojo's utility/security checks; model output from the final messages, tool-call
    stack trace from the full uncompacted trace."""
    last = final_messages[-1] if final_messages else None
    model_output = (last.get("content") or []) if last is not None and last["role"] == "assistant" else []
    stack = functions_stack_trace_from_messages(trace)
    utility = case.suite._check_task_result(case.user_task, model_output, pre_env, post_env, stack)
    if case.injection_task is None:
        return utility, None
    security = case.suite._check_task_result(case.injection_task, model_output, pre_env, post_env, stack)
    return utility, security


# --------------------------------------------------------------------------- persistence


def _msg_to_json(m: ChatMessage) -> dict:
    d = dict(m)
    if d.get("tool_calls"):
        d["tool_calls"] = [tc.model_dump() for tc in d["tool_calls"]]
    if d.get("tool_call") is not None:
        d["tool_call"] = d["tool_call"].model_dump()
    return d


def _msg_from_json(d: dict) -> ChatMessage:
    d = dict(d)
    if d.get("tool_calls"):
        d["tool_calls"] = [FunctionCall.model_validate(tc) for tc in d["tool_calls"]]
    if d.get("tool_call") is not None:
        d["tool_call"] = FunctionCall.model_validate(d["tool_call"])
    return d  # type: ignore[return-value]


def messages_to_json(messages) -> list[dict]:
    return [_msg_to_json(m) for m in messages]


def messages_from_json(data) -> list[ChatMessage]:
    return [_msg_from_json(d) for d in data]


def save_state(state: ForkState, path: Path) -> None:
    extra = dict(state.extra_args)
    comp = dict(extra.get(STATE_KEY) or {})
    if "pre_messages" in comp:
        comp["pre_messages"] = messages_to_json(comp["pre_messages"])
    extra[STATE_KEY] = comp
    payload = {
        "query": state.query,
        "env": json.loads(state.env.model_dump_json()),
        "pre_env": json.loads(state.pre_env.model_dump_json()),
        "env_type": f"{type(state.env).__module__}.{type(state.env).__qualname__}",
        "messages": messages_to_json(state.messages),
        "extra_args": extra,
        "batches_used": state.batches_used,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def load_state(path: Path, case: Case) -> ForkState:
    """Load a saved ForkState. The environment class comes from the case's suite."""
    payload = json.loads(path.read_text())
    env_cls = case.suite.environment_type
    extra = payload["extra_args"]
    comp = extra.get(STATE_KEY) or {}
    if "pre_messages" in comp:
        comp["pre_messages"] = messages_from_json(comp["pre_messages"])
    return ForkState(
        query=payload["query"],
        env=env_cls.model_validate(payload["env"]),
        pre_env=env_cls.model_validate(payload["pre_env"]),
        messages=messages_from_json(payload["messages"]),
        extra_args=extra,
        batches_used=payload["batches_used"],
    )
