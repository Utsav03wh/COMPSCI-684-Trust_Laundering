"""Benign utility: run each user task with no attack and no compaction.

Used to (1) check a model qualifies (plan: solves >= 40% of benign tasks) and
(2) build the case-set filter: only user tasks the model solves are kept for RQ1/RQ2.
Writes <out-dir>/raw/benign_<model>_<timestamp>.jsonl, one line per (suite, user task, sample).

    python runner/benign_utility.py --base-url http://127.0.0.1:8000/v1 --model qwen3-8b \
        --suites banking --out-dir /path/to/results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor
from agentdojo.task_suite.load_suites import get_suite

from pipeline.llm import NO_THINKING, VLLMChatLLM, make_client

BENCHMARK_VERSION = "v1.2"


def run_one(suite_name, uid, llm, system_message):
    suite = get_suite(BENCHMARK_VERSION, suite_name)
    pipe = AgentPipeline([SystemMessage(system_message), InitQuery(), llm,
                          ToolsExecutionLoop([ToolsExecutor(), llm])])
    pipe.name = llm.name
    t0 = time.time()
    try:
        env = suite.load_and_inject_default_environment(suite.get_injection_vector_defaults())
        utility, _ = suite.run_task_with_pipeline(pipe, suite.get_user_task_by_id(uid), None, {}, environment=env)
        return dict(suite=suite_name, user_task_id=uid, utility=bool(utility), error=None,
                    latency_s=round(time.time() - t0, 1))
    except Exception as e:  # noqa: BLE001
        return dict(suite=suite_name, user_task_id=uid, utility=False, error=f"{type(e).__name__}: {e}",
                    tb=traceback.format_exc()[-1500:], latency_s=round(time.time() - t0, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--suites", nargs="+", default=["banking"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    raw = args.out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    out = raw / f"benign_{args.model}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    llm = VLLMChatLLM(make_client(args.base_url), args.model, temperature=0.0, extra_body=NO_THINKING)
    system_message = load_system_message(None)
    jobs = [(s, u) for s in args.suites
            for u in sorted(get_suite(BENCHMARK_VERSION, s).user_tasks, key=lambda x: int(x.split("_")[-1]))]
    print(f"benign: {len(jobs)} user tasks -> {out}", flush=True)
    rows = []
    with ThreadPoolExecutor(args.workers) as ex, out.open("w") as f:
        for fut in as_completed([ex.submit(run_one, s, u, llm, system_message) for s, u in jobs]):
            r = dict(fut.result(), model=args.model)
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
    for s in args.suites:
        rs = [r for r in rows if r["suite"] == s]
        ok = sorted((r["user_task_id"] for r in rs if r["utility"]), key=lambda x: int(x.split("_")[-1]))
        print(f"[{s}] benign utility = {len(ok)}/{len(rs)} = {len(ok) / max(len(rs), 1):.1%}  "
              f"errors={sum(bool(r['error']) for r in rs)}")
        print(f"  solved: {' '.join(ok)}")


if __name__ == "__main__":
    main()
