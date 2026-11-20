# Trust Laundering: code

Does LLM context compaction promote injected tool-output text into trusted instructions?
Built on [AgentDojo](https://github.com/ethz-spylab/agentdojo) 0.1.35, benchmark version `v1.2`.
Proposal: `Project_Proposal.pdf` in this folder.

## Setup (Unity)

```bash
cd Trust_Laundering
module load conda/latest && conda activate trustlaundering
PIP_NO_CACHE_DIR=1 pip install -r requirements.txt   # only when requirements change
python tests/test_fork.py                            # CPU-only smoke test, ~1 min, login-node safe
```

## Layout

| Path | Owner | What |
|---|---|---|
| `compaction/base.py` | Utsav | `Compactor` interface: `compact(messages, cfg) -> CompactionResult` |
| `compaction/{truncation,abstractive}.py`, `prompts/` | Utsav | Compaction families; register new ones in `compaction/__init__.py` |
| `pipeline/compaction_element.py` | Utsav | In-loop element: trigger on the injection, then compact or fork |
| `pipeline/fork.py` | Utsav | `run_prefix` → `ForkState` → `run_branch` (A / B), scoring, save/load |
| `pipeline/trigger.py` | Utsav | Injection detection (normalized line match; YAML re-wraps payloads) |
| `pipeline/llm.py` | Utsav | vLLM chat element (sends temperature=0, system role) + summarizer |
| `tests/test_fork.py` | Utsav | Scripted-agent smoke test of the whole branch path |
| `attacks/` | Siddharth | A1 (`important_instructions`), A2 summarizer-targeted (draft templates), goal markers |
| `runner/explore_rq1.py` | — | RQ1 exploration: branch A vs B per pair, JSONL + traces, prints ASR / delayed detonation / laundering |
| `runner/slurm/explore_rq1.sbatch` | — | Serves Qwen3-8B with vLLM (env `vllm-gptoss`) on one GPU and runs the exploration |

## RQ1 exploration

```bash
cd Trust_Laundering
sbatch runner/slurm/explore_rq1.sbatch                          # smoke test: 4 banking pairs x {A1, A2}
LIMIT=0 sbatch --time=03:00:00 runner/slurm/explore_rq1.sbatch  # all 144 banking pairs
python runner/explore_rq1.py --summarize <results>/raw/<run>.jsonl
```
Results go to `/scratch3/workspace/udhanuka_umass_edu-cs684/trust_laundering/results` (workspace expires 2026-12-14).

## Usage sketch

```python
from pipeline.llm import make_client, VLLMChatLLM, make_summarizer
from pipeline.fork import Case, ForkState, run_prefix, run_branch
from compaction import CompactionConfig, build_compactor

client = make_client("http://<gpu-node>:8000/v1")
llm = VLLMChatLLM(client, "<served-model-name>")
cfg = CompactionConfig(family="abstractive_agent", role="user")
compactor = build_compactor(cfg, make_summarizer(client, "<served-model-name>"))

state = run_prefix(case, llm, system_message, m=0)
if isinstance(state, ForkState):
    a = run_branch(state, case, llm)                  # no compaction
    b = run_branch(state, case, llm, compactor, cfg)  # compaction
```

## AgentDojo gotchas (0.1.35)

- `OpenAILLM` drops `temperature=0.0` (`temperature or NOT_GIVEN`) and sends system as `developer`; use `pipeline.llm.VLLMChatLLM`.
- `important_instructions` raises for unknown model names; keep the pipeline name prefixed `vllm_parsed`.
- Tool outputs are YAML, so injection payloads are never verbatim in tool messages; use `pipeline.trigger`.
- Scoring reads the tool-call trace from the final message list; `run_branch` scores on the full uncompacted trace.
