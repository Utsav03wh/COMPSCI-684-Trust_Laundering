"""Attack families. A1 = AgentDojo's important_instructions (unchanged); A2 = summarizer-targeted.

Both are AgentDojo attack classes with the same constructor, so the runner treats them the
same way (plan, interface 3). The pipeline/LLM passed as `target_pipeline` must have a name
starting with "vllm_parsed" (see pipeline/llm.py), or A1 raises.
"""

from __future__ import annotations

from agentdojo.attacks.important_instructions_attacks import ImportantInstructionsAttack

from attacks.a2_summarizer import A2_TEMPLATES, SummarizerTargetedAttack

ATTACKS = {
    "A1": ImportantInstructionsAttack,
    "A2": SummarizerTargetedAttack,
}

__all__ = ["A2_TEMPLATES", "ATTACKS", "SummarizerTargetedAttack"]
