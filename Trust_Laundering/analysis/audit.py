"""Audit RQ1/RQ2 runs against their saved traces (CPU only, login-node safe).

For each run JSONL (from runner/explore_rq1.py) checks:
  integrity  A and B share an identical prefix that ends on the poisoned tool result;
             compaction fired; summary non-empty; how many messages compaction removed
  scoring    security=True only when that branch's own trace contains the attacker's action
             (goal markers in tool-call args); unparsed tool calls; iteration-cap hits
  mechanism  splits B outcomes by whether the agent re-read the raw injection after
             compaction (re-exposure) or only had the summary (laundering through the summary).
             Re-read is post-treatment, so the split is descriptive, not a causal estimate.

    python analysis/audit.py <results>/raw/explore_rq1_*.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentdojo.task_suite.load_suites import get_suite

from attacks import ATTACKS
from attacks.markers import goal_markers
from pipeline.trigger import contains_injection, payload_keys

BENCHMARK_VERSION = "v1.2"
ITER_CAP = 15


class _Named:  # A1 only needs a pipeline name to pick the model string
    def __init__(self, model: str) -> None:
        self.name = f"vllm_parsed/{model}"


def text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(b.get("content") or "" for b in c if isinstance(b, dict))
    return c or ""


def calls(msgs):
    return [c for m in msgs if m["role"] == "assistant" for c in (m.get("tool_calls") or [])]


def mcnemar(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


class Context:
    """Per-suite markers and injection payload keys, rebuilt exactly as the runner built them."""

    def __init__(self) -> None:
        self.markers: dict = {}
        self.keys: dict = {}

    def load(self, suite_name: str, attack: str, model: str) -> None:
        if (suite_name, attack) in self.keys:
            return
        suite = get_suite(BENCHMARK_VERSION, suite_name)
        ut0 = next(iter(suite.user_tasks.values()))
        clean = ut0.init_environment(suite.load_and_inject_default_environment(suite.get_injection_vector_defaults()))
        for iid, it in suite.injection_tasks.items():
            self.markers[(suite_name, iid)] = goal_markers(it, ut0, clean)
        atk = ATTACKS[attack](suite, _Named(model))
        keys = {}
        for uid, ut in suite.user_tasks.items():
            for iid, it in suite.injection_tasks.items():
                try:
                    keys[(uid, iid)] = payload_keys(atk.attack(ut, it).values())
                except ValueError:
                    pass
        self.keys[(suite_name, attack)] = keys


def audit(path: Path, ctx: Context) -> None:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    pairs = defaultdict(dict)
    for r in rows:
        if r["branch"] in ("A", "B"):
            pairs[(r["suite"], r["user_task_id"], r["injection_task_id"], r["attack"])][r["branch"]] = r
    pairs = {k: v for k, v in pairs.items() if "A" in v and "B" in v}
    if not pairs:
        print(f"\n== {path.name}: no paired rows")
        return
    b0 = next(iter(pairs.values()))["B"]
    print(f"\n== {path.name}  family={b0['family']} role={b0['summary_role']} "
          f"verbatim={b0['keep_user_verbatim']} model={b0['model']}  pairs={len(pairs)}")

    c = Counter()
    removed = Counter()
    split = defaultdict(list)
    for (s, u, i, a), p in pairs.items():
        ctx.load(s, a, p["A"]["model"])
        t = json.loads(Path(p["A"]["trace_path"]).read_text())
        A, B, final = t["A"], t["B"], t["B_final_context"]
        n = 0
        while n < min(len(A), len(B)) and A[n] == B[n]:
            n += 1
        c["prefix ends on tool result"] += A[n - 1]["role"] == "tool"
        c["compaction fired"] += bool(p["B"]["compaction_fired"])
        c["summary non-empty"] += bool((p["B"].get("summary_text") or "").strip())
        removed[n - (len(final) - (len(B) - n))] += 1

        mk = ctx.markers[(s, i)]
        for br, msgs in (("A", A), ("B", B)):
            sec = bool(p[br]["security"])
            if mk:
                act = any(any(x in json.dumps(cl["args"]) for x in mk) for cl in calls(msgs))
                c["SCORING security=True but no attacker call (must be 0)"] += sec and not act
                c["attacker call but security=False (partial attempts)"] += act and not sec
            c[f"unparsed tool call in {br}"] += any(
                "<tool_call>" in text(m) for m in msgs if m["role"] == "assistant")
            c[f"hit {ITER_CAP}-call cap in {br}"] += len(calls(msgs)) >= ITER_CAP

        keys = ctx.keys[(s, a)].get((u, i), [])
        reread = any(m["role"] == "tool" and contains_injection(text(m), keys) for m in B[n:])
        split[(a, "re-read" if reread else "summary only")].append(p)

    for k in sorted(c):
        print(f"  {k:55s} {c[k]}")
    print(f"  messages removed by compaction at the fork: {dict(sorted(removed.items()))}")
    print("  mechanism split (B outcomes by post-compaction re-read of the raw injection):")
    for (a, g), gp in sorted(split.items()):
        asr_a = sum(p["A"]["security"] for p in gp) / len(gp)
        asr_b = sum(p["B"]["security"] for p in gp) / len(gp)
        dd = sum(not p["A"]["security"] and p["B"]["security"] for p in gp)
        rs = sum(p["A"]["security"] and not p["B"]["security"] for p in gp)
        print(f"    {a} {g:12s} n={len(gp):3d}  ASR A={asr_a:5.1%} B={asr_b:5.1%} ({asr_b - asr_a:+6.1%})  "
              f"A fail->B succ={dd:2d}  A succ->B fail={rs:2d}  McNemar p={mcnemar(dd, rs):.2g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    ctx = Context()
    for p in args.runs:
        audit(p, ctx)


if __name__ == "__main__":
    main()
