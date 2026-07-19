"""Coding stage-1 generation: solo pass@1 baselines on HumanEval.

One independent completion per (model, problem) — no council structures. For
each cell the model generates a completion, we extract the code, and we run it
against the hidden unit tests. This yields the per-model pass@1 and the
pass/fail vector per problem that the coding council (run_coding_council.py)
re-aggregates. The council's job downstream is SELECTION among these member
solutions, so the pass/fail bool recorded here is the ground truth the council
is graded against.

Records: {model, task_id, completion, passed, outcome, cost, provider, error,
timestamp} with outcome in passed|failed|refused|no_answer|error. A failed
unit test is a valid outcome (error stays None so it is NOT retried on
resume); the test traceback is kept separately in exec_error. NO fallbacks:
a refused or unparseable generation is recorded as refused/no_answer, never
substituted.

Usage:
    python -m experiments.run_coding_stage1 --smoke        # 2 problems
    python -m experiments.run_coding_stage1 --n 164        # full HumanEval
    python -m experiments.run_coding_stage1 --dry-run
    python -m experiments.run_coding_stage1 --analyze-only

Checkpointed exactly like run_phase2_scout: atomic write + .bak, resume from
completed (model, task_id) pairs, errored records retry.

SAFETY: executes untrusted model-generated code in a subprocess with a hard
timeout. See backend/execution.py — not a security sandbox; production needs a
container/VM.
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import httpx

from backend.config import COUNCIL_V2_MODELS
from backend.evaluation.humaneval import HumanEvalBenchmark
from backend.execution import check_solution
from backend.model_router import is_anthropic_direct, query_model_routed
from experiments.run_phase2_scout import REFUSAL_MARKERS, load_results, save_results

DEFAULT_OUTPUT_DIR = "experiments/results_coding_stage1"
SMOKE_OUTPUT_DIR = "experiments/results_coding_smoke"
RESULTS_FILENAME = "stage1_results.json"

ANTHROPIC_MAX_CONCURRENT = 4
OPENROUTER_MAX_CONCURRENT = 6
EXEC_TIMEOUT = 10.0
EST_COST_PER_PAID_CALL = 0.05  # dry-run estimate per OpenRouter generation

# Runtime-mutable council composition (main() may override via --council).
COUNCIL: List[str] = list(COUNCIL_V2_MODELS)


def result_key(record: dict) -> tuple:
    return (record["model"], record["task_id"])


async def routed_call(
    client: httpx.AsyncClient, model: str, prompt: str, sems: dict
) -> dict:
    """query_model_routed under the provider-appropriate semaphore."""
    sem = sems["anthropic"] if is_anthropic_direct(model) else sems["openrouter"]
    async with sem:
        return await query_model_routed(client, model, prompt)


def classify_generation(raw: dict, code: str) -> Optional[str]:
    """Pre-execution outcome, or None when the completion should be executed."""
    if raw["error"]:
        return "error"
    fr = (raw.get("native_finish_reason") or raw.get("finish_reason") or "").lower()
    content = (raw.get("content") or "").strip()
    if "refusal" in fr or (
        content and any(m in content.lower()[:300] for m in REFUSAL_MARKERS)
    ):
        return "refused"
    if not content or not (code or "").strip():
        return "no_answer"
    return None


def build_record(model: str, problem, raw: dict, code: str) -> dict:
    """Generate -> classify -> (maybe) execute -> one stage-1 record."""
    outcome = classify_generation(raw, code)
    passed = False
    exec_error = None
    if outcome is None:
        result = check_solution(problem, code, timeout=EXEC_TIMEOUT)
        passed = result["passed"]
        exec_error = result["error"]
        outcome = "passed" if passed else "failed"
    return {
        "model": model,
        "task_id": problem.task_id,
        "benchmark": "HumanEval",
        "completion": code,
        "passed": passed,
        "outcome": outcome,
        # error = transport/call failure only (drives resume retry). A failed
        # unit test keeps error=None; its traceback lives in exec_error.
        "error": raw["error"],
        "exec_error": exec_error,
        "cost": raw["cost"],
        "provider": raw["provider"],
        "finish_reason": raw["finish_reason"],
        "native_finish_reason": raw["native_finish_reason"],
        "timestamp": datetime.now().isoformat(),
    }


async def run_stage1(
    n: Optional[int], output_dir: str, max_concurrent: Optional[int]
) -> List[dict]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / RESULTS_FILENAME

    benchmark = HumanEvalBenchmark()
    problems = benchmark.load_problems(n=n)
    print(f"Loaded {len(problems)} HumanEval problems | council: {COUNCIL}")

    results = load_results(filepath)
    done = {result_key(r) for r in results if not r.get("error")}
    print(f"Resuming with {len(results)} records ({len(done)} clean)")
    results = [r for r in results if result_key(r) in done]  # errored retry

    todo = [
        (model, p)
        for model in COUNCIL
        for p in problems
        if (model, p.task_id) not in done
    ]
    print(f"To run: {len(todo)} generations across {len(COUNCIL)} models")
    if not todo:
        save_results(results, filepath)
        return results

    or_cap = max_concurrent or OPENROUTER_MAX_CONCURRENT
    sems = {
        "anthropic": asyncio.Semaphore(ANTHROPIC_MAX_CONCURRENT),
        "openrouter": asyncio.Semaphore(or_cap),
    }
    lock = asyncio.Lock()
    completed = 0

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=or_cap + ANTHROPIC_MAX_CONCURRENT + 4)
    ) as client:

        async def worker(model: str, problem) -> None:
            nonlocal completed
            prompt = benchmark.build_generation_prompt(problem)
            raw = await routed_call(client, model, prompt, sems)
            code = (
                benchmark.extract_code(raw["content"], problem.entry_point)
                if raw["content"]
                else ""
            )
            # Execution is blocking (subprocess); keep it off the event loop.
            rec = await asyncio.to_thread(build_record, model, problem, raw, code)
            async with lock:
                results.append(rec)
                completed += 1
                if completed % 5 == 0 or completed == len(todo):
                    save_results(results, filepath)
                    spent = sum(r.get("cost") or 0 for r in results)
                    print(
                        f"  {completed}/{len(todo)} done | cumulative cost ${spent:.2f}",
                        flush=True,
                    )

        await asyncio.gather(*(worker(m, p) for m, p in todo))

    save_results(results, filepath)
    return results


def dry_run(n: Optional[int], output_dir: str) -> dict:
    """Print the plan + cost estimate without any calls or execution."""
    benchmark = HumanEvalBenchmark()
    problems = benchmark.load_problems(n=n)
    done = {
        result_key(r)
        for r in load_results(Path(output_dir) / RESULTS_FILENAME)
        if not r.get("error")
    }
    todo = [
        (m, p) for m in COUNCIL for p in problems if (m, p.task_id) not in done
    ]
    paid = sum(1 for m, _ in todo if not is_anthropic_direct(m))
    est_cost = paid * EST_COST_PER_PAID_CALL
    print("DRY RUN — no calls, no execution")
    print(f"  problems:              {len(problems)}")
    print(f"  models:                {len(COUNCIL)}")
    print(f"  generations to run:    {len(todo)}")
    print(f"  paid OpenRouter calls: {paid} (anthropic-direct is $0)")
    print(f"  estimated cost:        ${est_cost:.2f} (@ ${EST_COST_PER_PAID_CALL}/call)")
    return {"n_problems": len(problems), "paid_calls": paid, "est_cost": est_cost}


def analyze(results: List[dict]) -> None:
    print("\n" + "=" * 70)
    print("CODING STAGE-1 ANALYSIS (pass@1)")
    print("=" * 70)
    models = sorted({r["model"] for r in results})
    total_cost = sum(r.get("cost") or 0 for r in results)
    print(f"Records: {len(results)} | total cost ${total_cost:.2f}\n")
    for m in models:
        recs = [r for r in results if r["model"] == m]
        n = len(recs)
        counts: dict = {}
        for r in recs:
            counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
        passed = counts.get("passed", 0)
        pass1 = passed / n if n else 0.0
        mcost = sum(r.get("cost") or 0 for r in recs)
        print(
            f"  {m:38s} pass@1={pass1:5.1%} (n={n})  "
            f"failed={counts.get('failed', 0)} refused={counts.get('refused', 0)} "
            f"no_answer={counts.get('no_answer', 0)} error={counts.get('error', 0)} "
            f"cost=${mcost:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Coding stage-1 HumanEval pass@1")
    parser.add_argument("--smoke", action="store_true", help="2 problems only")
    parser.add_argument("--n", type=int, default=164, help="problems (default 164)")
    parser.add_argument(
        "--council",
        default=None,
        help="comma-separated council model slugs (default: COUNCIL_V2_MODELS)",
    )
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    global COUNCIL
    if args.council:
        COUNCIL = [m.strip() for m in args.council.split(",") if m.strip()]

    n = 2 if args.smoke else args.n
    output_dir = SMOKE_OUTPUT_DIR if args.smoke else args.output_dir

    if args.analyze_only:
        analyze(load_results(Path(output_dir) / RESULTS_FILENAME))
        return
    if args.dry_run:
        dry_run(n, output_dir)
        return

    results = asyncio.run(run_stage1(n, output_dir, args.max_concurrent))
    analyze(results)


if __name__ == "__main__":
    main()
