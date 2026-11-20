"""RQ1 exploration: how often does compaction turn a resisted injection into a successful one?

For each (user task, injection task, attack): run to the poisoned tool result, snapshot, then
  branch A: no compaction
  branch B: compaction (default: agent-style summary in the user role)
from the identical snapshot. Writes one JSONL line per branch plus a full trace per pair,
then prints the RQ1 numbers (ASR per branch, delayed-detonation, rescue, laundering rate).

Needs a running vLLM server (see runner/slurm/explore_rq1.sbatch). Example:
    python runner/explore_rq1.py --base-url http://127.0.0.1:8000/v1 --model qwen3-8b \
        --suites banking --attacks A1 A2 --limit 4 --out-dir /path/to/results
Re-print the summary of an existing run:
    python runner/explore_rq1.py --summarize /path/to/results/<run>.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.task_suite.load_suites import get_suite

from attacks import ATTACKS
from attacks.markers import goal_markers, marker_hit
from compaction import CompactionConfig, build_compactor
from pipeline.fork import BranchResult, Case, ForkState, messages_to_json, run_branch, run_prefix
from pipeline.llm import NO_THINKING, VLLMChatLLM, make_client, make_summarizer
from pipeline.trigger import contains_injection, payload_keys

BENCHMARK_VERSION = "v1.2"


def build_pairs(suites, user_tasks, injection_tasks, limit, seed):
    pairs = []
    for sname in suites:
        suite = get_suite(BENCHMARK_VERSION, sname)
        uts = user_tasks or sorted(suite.user_tasks, key=lambda x: int(x.split("_")[-1]))
        its = injection_tasks or sorted(suite.injection_tasks, key=lambda x: int(x.split("_")[-1]))
        pairs += [(sname, u, i) for u in uts for i in its]
    if limit:
        random.Random(seed).shuffle(pairs)  # spread a small limit across tasks
        pairs = sorted(pairs[:limit])
    return pairs


def run_pair(job, llm, compactor, cfg, system_message, m, model, run_id, traces_dir):
    sname, uid, iid, attack_name, injections = job
    suite = get_suite(BENCHMARK_VERSION, sname)
    ut, it = suite.get_user_task_by_id(uid), suite.get_injection_task_by_id(iid)
    clean = ut.init_environment(suite.load_and_inject_default_environment(suite.get_injection_vector_defaults()))
    markers = goal_markers(it, ut, clean)
    keys = payload_keys(injections.values())
    base = dict(
        run_id=run_id, experiment="explore_rq1", seed=0, suite=sname, user_task_id=uid,
        injection_task_id=iid, attack=attack_name, model=model, m=m, n_markers=len(markers),
    )
    comp_fields = dict(family=cfg.family, summary_role=cfg.role, keep_user_verbatim=cfg.keep_user_verbatim)
    t0 = time.time()
    try:
        case = Case(suite, ut, it, injections)
        state = run_prefix(case, llm, system_message, m=m)
        if isinstance(state, BranchResult):  # injection never read, or agent stopped first
            return [dict(base, branch="no_fork", utility=state.utility, security=state.security,
                         latency_s=round(time.time() - t0, 1), error=None)]
        a = run_branch(state, case, llm)
        b = run_branch(state, case, llm, compactor, cfg)
    except Exception as e:  # noqa: BLE001 - record and continue
        return [dict(base, branch="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-2000:])]

    trace_path = traces_dir / f"{sname}__{uid}__{iid}__{attack_name}.json"
    trace_path.write_text(json.dumps({"A": messages_to_json(a.trace), "B": messages_to_json(b.trace),
                                      "B_final_context": messages_to_json(b.messages)}, default=str))
    summary = b.compaction.get("summary_text")
    rows = []
    for r in (a, b):
        rows.append(dict(
            base, **comp_fields, branch=r.branch, fork_step=state.batches_used,
            utility=r.utility, security=r.security,
            compaction_fired=bool(r.compaction.get("fired")),
            summary_text=summary if r.branch == "B" else None,
            marker_hit=marker_hit(summary, markers) if r.branch == "B" else None,
            payload_copied=contains_injection(summary or "", keys) if r.branch == "B" else None,
            latency_s=round(time.time() - t0, 1), error=r.error, trace_path=str(trace_path),
        ))
    return rows


def summarize(path: Path) -> None:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    by = defaultdict(dict)
    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        key = (r["suite"], r["user_task_id"], r["injection_task_id"], r["attack"])
        counts[r["attack"]][r["branch"]] += 1
        if r["branch"] in ("A", "B"):
            by[key][r["branch"]] = r
    print(f"\n=== {path.name}  ({len(rows)} rows)")
    for attack in sorted(counts):
        pairs = [v for k, v in by.items() if k[3] == attack and "A" in v and "B" in v]
        n = len(pairs)
        c = counts[attack]
        print(f"\n[{attack}] paired={n}  no_fork={c.get('no_fork', 0)}  errors={c.get('error', 0)}")
        if not n:
            continue
        asr_a = sum(p["A"]["security"] for p in pairs) / n
        asr_b = sum(p["B"]["security"] for p in pairs) / n
        a_fail = [p for p in pairs if not p["A"]["security"]]
        a_succ = [p for p in pairs if p["A"]["security"]]
        dd = sum(p["B"]["security"] for p in a_fail)
        rescue = sum(not p["B"]["security"] for p in a_succ)
        laund = sum(bool(p["B"]["marker_hit"]) for p in pairs)
        with_markers = sum(p["B"]["n_markers"] > 0 for p in pairs)
        util_a = sum(p["A"]["utility"] for p in pairs) / n
        util_b = sum(p["B"]["utility"] for p in pairs) / n
        print(f"  ASR  A (no compaction) = {asr_a:6.1%}   B (compaction) = {asr_b:6.1%}")
        print(f"  delayed detonation     = {dd}/{len(a_fail)}" + (f" = {dd / len(a_fail):.1%}" if a_fail else ""))
        print(f"  rescue                 = {rescue}/{len(a_succ)}" + (f" = {rescue / len(a_succ):.1%}" if a_succ else ""))
        print(f"  laundering (marker)    = {laund}/{with_markers} summaries with markers")
        print(f"  payload copied verbatim= {sum(bool(p['B']['payload_copied']) for p in pairs)}/{n}")
        print(f"  utility A = {util_a:.1%}   B = {util_b:.1%}")
        print(f"  discordant pairs for McNemar: A fail->B succ = {dd}, A succ->B fail = {rescue}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summarize", type=Path, help="only print the summary of an existing JSONL")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen3-8b", help="served model name")
    ap.add_argument("--suites", nargs="+", default=["banking"])
    ap.add_argument("--user-tasks", nargs="*")
    ap.add_argument("--injection-tasks", nargs="*")
    ap.add_argument("--attacks", nargs="+", default=["A1", "A2"], choices=sorted(ATTACKS))
    ap.add_argument("--limit", type=int, default=0, help="max (user, injection) pairs; 0 = all")
    ap.add_argument("--family", default="abstractive_agent")
    ap.add_argument("--role", default="user", choices=["user", "system", "assistant", "tool"])
    ap.add_argument("--keep-user-verbatim", action="store_true")
    ap.add_argument("--m", type=int, default=0, help="tool batches after the poisoned one before compacting")
    ap.add_argument("--workers", type=int, default=4, help="concurrent pairs (vLLM batches them)")
    ap.add_argument("--out-dir", type=Path, required=False, default=Path("results"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return

    run_id = f"explore_rq1_{datetime.now():%Y%m%d_%H%M%S}"
    raw_dir, traces_dir = args.out_dir / "raw", args.out_dir / "traces" / run_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{run_id}.jsonl"
    (raw_dir / f"{run_id}.args.json").write_text(json.dumps(vars(args), default=str, indent=2))

    client = make_client(args.base_url)
    llm = VLLMChatLLM(client, args.model, temperature=0.0, extra_body=NO_THINKING)
    cfg = CompactionConfig(family=args.family, role=args.role, keep_user_verbatim=args.keep_user_verbatim)
    compactor = build_compactor(cfg, make_summarizer(client, args.model, extra_body=NO_THINKING))
    system_message = load_system_message(None)

    # Build injections in the main thread (attack classes cache candidates; not thread-safe).
    jobs = []
    attack_objs = {}
    for sname, uid, iid in build_pairs(args.suites, args.user_tasks, args.injection_tasks, args.limit, args.seed):
        suite = get_suite(BENCHMARK_VERSION, sname)
        for a in args.attacks:
            if (sname, a) not in attack_objs:
                attack_objs[(sname, a)] = ATTACKS[a](suite, llm)
            try:
                inj = attack_objs[(sname, a)].attack(suite.get_user_task_by_id(uid), suite.get_injection_task_by_id(iid))
            except ValueError as e:  # user task not injectable
                print(f"skip {sname}/{uid}: {e}")
                continue
            jobs.append((sname, uid, iid, a, inj))

    print(f"{run_id}: {len(jobs)} jobs, workers={args.workers}, out={out_path}", flush=True)
    lock, done, t0 = Lock(), 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex, out_path.open("a") as f:
        futs = [ex.submit(run_pair, j, llm, compactor, cfg, system_message, args.m, args.model, run_id, traces_dir) for j in jobs]
        for fut in as_completed(futs):
            rows = fut.result()
            with lock:
                for r in rows:
                    f.write(json.dumps(r, default=str) + "\n")
                f.flush()
                done += 1
                tag = ",".join(f"{r['branch']}:{'S' if r.get('security') else '-'}" for r in rows)
                print(f"[{done}/{len(jobs)} {time.time() - t0:6.0f}s] {rows[0]['user_task_id']} {rows[0]['injection_task_id']} "
                      f"{rows[0]['attack']} {tag}", flush=True)
    summarize(out_path)


if __name__ == "__main__":
    main()
