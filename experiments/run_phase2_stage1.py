"""Phase-2 stage 1: solo baselines on the FULL eligible HLE pool (374 Q).

Same measurement as the scout (one independent answer per (model, question),
no council structures) extended to every text-only multiple-choice HLE
question that survives the pre-registered bio/chem exclusion. This is the
dataset every council arm (A-D, jury, cabinet, court, peer review)
re-aggregates or builds on.

Routing: anthropic/* models (Fable 5 chairman + Opus 4.8 side-arm) go direct
to the Anthropic API on research credits ($0 to the study budget);
GPT-5.6 Sol, Gemini 3.1 Pro, and Grok 4.5 go through OpenRouter.

Scout records are seeded in: the 2026-07-16 scout ran the same prompt/models
on a prefix of the same seeded shuffle, so its clean records for eligible
questions are valid stage-1 records and are not re-bought.

Usage:
    python -m experiments.run_phase2_stage1 --smoke      # 2 Q sanity check
    python -m experiments.run_phase2_stage1              # full pool
    python -m experiments.run_phase2_stage1 --analyze-only

Checkpointed exactly like the scout: atomic writes, .bak fallback, rerunning
resumes from completed (model, question) pairs; errored records retry.
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

import httpx

from backend.evaluation.hle import HLEBenchmark
from backend.model_router import is_anthropic_direct, query_model_routed
from experiments.run_phase2_scout import (
    ALL_MODELS,
    classify_outcome,
    analyze,
    load_results,
    result_key,
    save_results,
)

SCOUT_RESULTS = Path("experiments/results_phase2_scout/scout_results.json")

# Anthropic direct is rate-limited more tightly than OpenRouter fan-out.
ANTHROPIC_MAX_CONCURRENT = 8


def seed_from_scout(results: list, eligible_qids: set) -> list:
    """Adopt clean scout records for still-eligible questions (same prompt,
    same models, same seeded shuffle -> identical measurement, already paid)."""
    have = {result_key(r) for r in results}
    adopted = 0
    for rec in load_results(SCOUT_RESULTS):
        if (
            rec["question_id"] in eligible_qids
            and result_key(rec) not in have
            and not rec.get("error")
        ):
            rec.setdefault("provider", "openrouter")
            rec["seeded_from"] = "phase2_scout"
            results.append(rec)
            have.add(result_key(rec))
            adopted += 1
    print(f"Seeded {adopted} clean scout records")
    return results


async def run_stage1(n, output_dir: str, max_concurrent: int, seed: bool = True) -> list:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / "stage1_results.json"

    benchmark = HLEBenchmark(exclude_categories=HLEBenchmark.PHASE2_EXCLUDED_CATEGORIES)
    questions = benchmark.load_questions(n=n)
    print(f"Loaded {len(questions)} eligible HLE MCQ questions")

    results = load_results(filepath)
    if seed:
        results = seed_from_scout(results, {q.id for q in questions})
    done = {result_key(r) for r in results if not r.get("error")}
    print(f"Resuming with {len(results)} records ({len(done)} clean)")
    results = [r for r in results if result_key(r) in done]

    todo = [
        (model, q)
        for model in ALL_MODELS
        for q in questions
        if (model, q.id) not in done
    ]
    n_anthropic = sum(1 for m, _ in todo if is_anthropic_direct(m))
    print(
        f"To run: {len(todo)} calls "
        f"({n_anthropic} anthropic-direct, {len(todo) - n_anthropic} openrouter)"
    )

    or_sem = asyncio.Semaphore(max_concurrent)
    ant_sem = asyncio.Semaphore(ANTHROPIC_MAX_CONCURRENT)
    lock = asyncio.Lock()
    completed = 0

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_concurrent + ANTHROPIC_MAX_CONCURRENT + 4)
    ) as client:

        async def worker(model: str, q) -> None:
            nonlocal completed
            sem = ant_sem if is_anthropic_direct(model) else or_sem
            async with sem:
                raw = await query_model_routed(client, model, q.text)
            ev = benchmark.evaluate(q, raw["content"]) if raw["content"] else None
            rec = {
                "model": model,
                "question_id": q.id,
                "benchmark": "HLE",
                "category": q.metadata.get("category"),
                "ground_truth": q.ground_truth,
                "predicted": ev.predicted if ev else None,
                "is_correct": bool(ev.is_correct) if ev else False,
                "response": raw["content"],
                "finish_reason": raw["finish_reason"],
                "native_finish_reason": raw["native_finish_reason"],
                "prompt_tokens": raw["prompt_tokens"],
                "completion_tokens": raw["completion_tokens"],
                "reasoning_tokens": raw["reasoning_tokens"],
                "cost": raw["cost"],
                "provider": raw["provider"],
                "error": raw["error"],
                "timestamp": datetime.now().isoformat(),
            }
            rec["outcome"] = classify_outcome(rec)
            async with lock:
                results.append(rec)
                completed += 1
                if completed % 10 == 0 or completed == len(todo):
                    save_results(results, filepath)
                    spent = sum(
                        r.get("cost") or 0
                        for r in results
                        if r.get("provider") != "anthropic-direct"
                    )
                    print(
                        f"  {completed}/{len(todo)} done | OpenRouter spend ${spent:.2f}",
                        flush=True,
                    )

        await asyncio.gather(*(worker(m, q) for m, q in todo))

    save_results(results, filepath)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase-2 stage-1 full-pool run")
    parser.add_argument("--smoke", action="store_true", help="2 questions only")
    parser.add_argument("--n", type=int, default=None, help="questions (default: all eligible)")
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--output-dir", default="experiments/results_phase2_stage1")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if args.analyze_only:
        analyze(load_results(Path(args.output_dir) / "stage1_results.json"))
    else:
        n = 2 if args.smoke else args.n
        outdir = (
            "experiments/results_stage1_smoke" if args.smoke else args.output_dir
        )
        # Smoke skips seeding: the first questions of the shuffle are scout-
        # covered, so a seeded smoke would be a no-op and test nothing.
        results = asyncio.run(
            run_stage1(n, outdir, args.max_concurrent, seed=not args.smoke)
        )
        analyze(results)
