"""LLM element for a vLLM OpenAI-compatible server with native tool calling, plus the
summarizer function the abstractive compactors use.

Why not AgentDojo's OpenAILLM directly (agentdojo 0.1.35):
  1. It sends `temperature or NOT_GIVEN`, so temperature=0.0 is silently dropped and vLLM
     samples with the model's default temperature. We always send it.
  2. It maps role "system" to "developer". Not every vLLM version / chat template accepts
     that, so we map it back to "system".
We reuse AgentDojo's private converters, which is safe because the version is pinned.
"""

from __future__ import annotations

from collections.abc import Sequence

import openai
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.openai_llm import (
    _function_to_openai,
    _message_to_openai,
    _openai_to_assistant_message,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage

# AgentDojo's important_instructions attack looks the pipeline name up in MODEL_NAMES and
# raises for unknown names. "vllm_parsed" maps to "Local model". Keep this prefix.
PIPELINE_NAME_PREFIX = "vllm_parsed"


def make_client(base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY") -> openai.OpenAI:
    return openai.OpenAI(base_url=base_url, api_key=api_key)


def _to_openai(message: ChatMessage, model: str) -> dict:
    m = dict(_message_to_openai(message, model))
    if m.get("role") == "developer":
        m["role"] = "system"
    return m


class VLLMChatLLM(BasePipelineElement):
    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = 1024,
        seed: int | None = 0,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.name = f"{PIPELINE_NAME_PREFIX}/{model}"

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        tools = [_function_to_openai(t) for t in runtime.functions.values()]
        kwargs: dict = dict(
            model=self.model,
            messages=[_to_openai(m, self.model) for m in messages],
            temperature=self.temperature,
        )
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.seed is not None:
            kwargs["seed"] = self.seed
        completion = self.client.chat.completions.create(**kwargs)
        output = _openai_to_assistant_message(completion.choices[0].message)
        return query, runtime, env, [*messages, output], extra_args


def make_summarizer(client: openai.OpenAI, model: str, temperature: float = 0.0, seed: int | None = 0):
    """Return summarize(system_prompt, transcript, max_tokens) -> str for AbstractiveCompactor.
    Same model and endpoint as the agent by default (plan: temperature 0, <= 512 tokens)."""

    def summarize(system_prompt: str, transcript: str, max_tokens: int) -> str:
        kwargs: dict = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"<conversation>\n{transcript}\n</conversation>"},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if seed is not None:
            kwargs["seed"] = seed
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    return summarize
