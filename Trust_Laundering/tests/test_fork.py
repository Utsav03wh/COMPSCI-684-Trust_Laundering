"""CPU-only smoke test of the compaction pipeline, no LLM server needed.

A scripted "agent" plays the user task's ground-truth tool calls, and *obeys the attacker*
only when the injection text appears in a trusted role (user or system). With an echo
summarizer that copies tool output into the summary, this simulates the worst case of
trust laundering, so we can check plumbing and scoring end to end:

  branch A (no compaction)            -> attack fails  (injection only in tool role)
  branch B, abstractive, user role    -> attack succeeds
  branch B, abstractive, tool role    -> attack fails
  branch B, truncation                -> attack fails

Run:  python tests/test_fork.py      (or pytest tests/ once pytest is installed)
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string

from compaction import CompactionConfig, build_compactor
from compaction.base import check_well_formed
from pipeline.compaction_element import TRIGGER_KEYS, full_trace
from pipeline.fork import Case, ForkState, load_state, run_branch, run_prefix, save_state
from pipeline.trigger import contains_injection, payload_keys

SUITE, USER_TASK, INJ_TASK = "banking", "user_task_0", "injection_task_0"


class ScriptedLLM(BasePipelineElement):
    """Deterministic stand-in for the agent model."""

    name = "vllm_parsed/scripted"

    def __init__(self, user_calls: list[FunctionCall], inj_calls: list[FunctionCall]) -> None:
        self.user_calls, self.inj_calls = user_calls, inj_calls

    def query(self, query, runtime: FunctionsRuntime, env: Env = EmptyEnv(), messages: Sequence[ChatMessage] = [], extra_args: dict = {}):
        history = full_trace(messages, extra_args)  # progress survives compaction
        done = [tc.function for m in history if m["role"] == "assistant" for tc in (m.get("tool_calls") or [])]
        keys = extra_args.get(TRIGGER_KEYS, [])
        trusted_text = " ".join(get_text_content_as_str(m["content"]) for m in messages if m["role"] in ("user", "system"))
        obey = contains_injection(trusted_text, keys)

        plan = self.user_calls + (self.inj_calls if obey else [])
        remaining = list(plan)
        for f in done:  # remove calls already made, in order
            for i, c in enumerate(remaining):
                if c.function == f:
                    remaining.pop(i)
                    break
        if remaining:
            nxt = remaining[0]
            call = FunctionCall(function=nxt.function, args=dict(nxt.args), id=f"call_{len(done)}")
            out = ChatAssistantMessage(role="assistant", content=None, tool_calls=[call])
        else:
            out = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("Done.")], tool_calls=None)
        return query, runtime, env, [*messages, out], extra_args


def echo_summarizer(system_prompt: str, transcript: str, max_tokens: int) -> str:
    return "Pending tasks, from the conversation:\n" + transcript


def make_case() -> tuple[Case, ScriptedLLM]:
    suite = get_suite("v1.2", SUITE)
    ut, it = suite.get_user_task_by_id(USER_TASK), suite.get_injection_task_by_id(INJ_TASK)
    payload = ImportantInstructionsAttack._JB_STRING.format(user="Emma Johnson", model="Local model", goal=it.GOAL)
    injections = {k: payload for k in suite.get_injection_vector_defaults()}
    case = Case(suite, ut, it, injections)
    pre = ut.init_environment(suite.load_and_inject_default_environment(injections))
    llm = ScriptedLLM(ut.ground_truth(pre.model_copy(deep=True)), it.ground_truth(pre.model_copy(deep=True)))
    return case, llm


def get_state() -> tuple[Case, ScriptedLLM, ForkState]:
    case, llm = make_case()
    state = run_prefix(case, llm, load_system_message(None), m=0)
    assert isinstance(state, ForkState), f"expected a fork, got {state}"
    return case, llm, state


def test_trigger_matches_yaml_wrapped_payload():
    case, _llm, state = get_state()
    assert state.batches_used == 1, state.batches_used  # forked right after read_file
    assert state.messages[-1]["role"] == "tool"
    assert contains_injection(get_text_content_as_str(state.messages[-1]["content"]), payload_keys(case.injections.values()))


def test_branch_a_resists():
    case, llm, state = get_state()
    a = run_branch(state, case, llm)
    assert a.branch == "A" and a.utility is True and a.security is False, (a.utility, a.security)


def test_branch_b_user_role_launders():
    case, llm, state = get_state()
    cfg = CompactionConfig(family="abstractive_agent", role="user")
    b = run_branch(state, case, llm, build_compactor(cfg, echo_summarizer), cfg)
    assert b.branch == "B" and b.security is True, (b.utility, b.security)
    assert b.compaction["fired"] and b.compaction["meta"]["prompt_id"] == "agent_style"
    # the full trace still contains the pre-compaction read_file call
    assert any(tc.function == "read_file" for m in b.trace if m["role"] == "assistant" for tc in (m.get("tool_calls") or []))


def test_branch_b_tool_role_and_truncation_resist():
    case, llm, state = get_state()
    for cfg in [CompactionConfig(family="abstractive_generic", role="tool"), CompactionConfig(family="truncation")]:
        b = run_branch(state, case, llm, build_compactor(cfg, echo_summarizer), cfg)
        assert b.security is False, (cfg, b.security)


def test_all_roles_well_formed():
    case, llm, state = get_state()
    for role in ["user", "system", "assistant", "tool"]:
        for verbatim in [False, True]:
            cfg = CompactionConfig(family="abstractive_agent", role=role, keep_user_verbatim=verbatim, keep_last_k=1)
            res = build_compactor(cfg, echo_summarizer).compact(state.messages, cfg)
            check_well_formed(res.messages)
            assert sum(m["role"] == "system" for m in res.messages) == 1


def test_state_roundtrip_and_reuse():
    case, llm, state = get_state()
    # ignore_cleanup_errors: rmtree can hit ENOTEMPTY on Unity's network filesystems
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        p = Path(d) / "state.json"
        save_state(state, p)
        loaded = load_state(p, case)
    assert loaded.env == state.env and loaded.pre_env == state.pre_env
    a1, a2 = run_branch(state, case, llm), run_branch(loaded, case, llm)
    assert (a1.utility, a1.security) == (a2.utility, a2.security)
    # state is not mutated by running branches
    assert run_branch(state, case, llm).security is False


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
