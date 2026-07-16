"""Phase-2 M1 scouting pass: solo stage-1 baselines on HLE for council v2.

One independent answer per (model, question) — no council structures. This
measures the number that gates everything downstream: how many
minority-correct cells (exactly one council member right) exist on HLE.
Also yields the pairwise error-correlation matrix (M2) and the
intra-vendor vs cross-vendor probe (M6, via the Opus 4.8 side-arm).

Usage:
    python -m experiments.run_phase2_scout --smoke        # 5 Q, cost check
    python -m experiments.run_phase2_scout --n 150        # scouting pass
    python -m experiments.run_phase2_scout --analyze-only # reprint analysis

Checkpointed: results saved incrementally (atomic write + .bak); rerunning
resumes from completed (model, question) pairs. Refusals are recorded as
their own outcome category, not errors.
"""

import argparse
import asyncio
import json
import os
import shutil
from datetime import datetime
from itertools import combinations
from pathlib import Path

import httpx

from backend.config import OPENROUTER_API_KEY, OPENROUTER_API_URL
from backend.evaluation.hle import HLEBenchmark
from backend.openrouter import NO_SAMPLING_PARAM_MODELS

# Council v2 (slugs + release dates verified on OpenRouter 2026-07-15) +
# Opus side-arm. Strongest model per vendor: GPT-5.6 Sol (2026-07-09),
# Gemini 3.1 Pro (2026-02-19), Fable 5 (2026-06-09), Grok 4.5 (2026-07-08).
# NOTE: x-ai/grok-4.20 is version 4.2 from MARCH, older than 4.5.
COUNCIL_V2 = [
    "openai/gpt-5.6-sol",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-fable-5",
    "x-ai/grok-4.5",
]
SIDE_ARM = ["anthropic/claude-opus-4.8"]  # intra-vendor probe (M6)
ALL_MODELS = COUNCIL_V2 + SIDE_ARM

MAX_TOKENS = 24000
TIMEOUT_S = 600.0
RETRIES = 5

REFUSAL_MARKERS = (
    "i can't help with that",
    "i cannot help with that",
    "i'm not able to help with",
    "unable to assist with this request",
)


def result_key(record: dict) -> tuple:
    return (record["model"], record["question_id"])


def save_results(results: list, filepath: Path) -> None:
    """Crash-safe write: tmp + fsync + .bak + atomic replace."""
    tmp = filepath.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    if filepath.exists():
        shutil.copyfile(filepath, filepath.with_suffix(".json.bak"))
    os.replace(tmp, filepath)


def load_results(filepath: Path) -> list:
    for candidate in (filepath, filepath.with_suffix(".json.bak")):
        if candidate.exists():
            try:
                with open(candidate, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                print(f"WARNING: {candidate} is corrupt, trying backup")
    return []


async def query_once(client: httpx.AsyncClient, model: str, prompt: str) -> dict:
    """One API call with usage accounting; returns raw fields for the record."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "usage": {"include": True},
    }
    if not any(m in model for m in NO_SAMPLING_PARAM_MODELS):
        payload["temperature"] = 0.0

    last_err = None
    for attempt in range(RETRIES):
        try:
            resp = await client.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=httpx.Timeout(connect=30.0, read=TIMEOUT_S, write=30.0, pool=120.0),
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"API error: {data['error']}")
            choice = data["choices"][0]
            usage = data.get("usage", {}) or {}
            return {
                "content": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "native_finish_reason": choice.get("native_finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                ),
                "cost": usage.get("cost"),
                "error": None,
            }
        except (httpx.TransportError, httpx.HTTPStatusError, RuntimeError, KeyError, ValueError) as e:
            last_err = e
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 400:
                break  # non-retryable
            await asyncio.sleep(min(2 * 2**attempt, 45))
    return {
        "content": "",
        "finish_reason": None,
        "native_finish_reason": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "cost": None,
        "error": str(last_err),
    }


def classify_outcome(rec: dict) -> str:
    """correct | wrong | refused | error | no_answer"""
    if rec["error"]:
        return "error"
    fr = (rec.get("native_finish_reason") or rec.get("finish_reason") or "").lower()
    content = (rec.get("response") or "").strip()
    if "refusal" in fr or (
        content and any(m in content.lower()[:300] for m in REFUSAL_MARKERS)
    ):
        return "refused"
    if not content or rec.get("predicted") == "[no letter found]":
        return "no_answer"
    return "correct" if rec["is_correct"] else "wrong"


async def run_scout(n: int, output_dir: str, max_concurrent: int) -> list:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / "scout_results.json"

    benchmark = HLEBenchmark(
        exclude_categories=HLEBenchmark.PHASE2_EXCLUDED_CATEGORIES
    )
    questions = benchmark.load_questions(n=n)
    print(f"Loaded {len(questions)} HLE MCQ questions")

    results = load_results(filepath)
    done = {result_key(r) for r in results if not r.get("error")}
    print(f"Resuming with {len(results)} records ({len(done)} clean)")
    # Drop errored records so they retry
    results = [r for r in results if result_key(r) in done]

    todo = [
        (model, q)
        for model in ALL_MODELS
        for q in questions
        if (model, q.id) not in done
    ]
    print(f"To run: {len(todo)} calls across {len(ALL_MODELS)} models")

    sem = asyncio.Semaphore(max_concurrent)
    lock = asyncio.Lock()
    completed = 0

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max_concurrent + 4)
    ) as client:

        async def worker(model: str, q) -> None:
            nonlocal completed
            async with sem:
                raw = await query_once(client, model, q.text)
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
                "error": raw["error"],
                "timestamp": datetime.now().isoformat(),
            }
            rec["outcome"] = classify_outcome(rec)
            async with lock:
                results.append(rec)
                completed += 1
                if completed % 10 == 0 or completed == len(todo):
                    save_results(results, filepath)
                    spent = sum(r.get("cost") or 0 for r in results)
                    print(
                        f"  {completed}/{len(todo)} done | cumulative cost ${spent:.2f}",
                        flush=True,
                    )

        await asyncio.gather(*(worker(m, q) for m, q in todo))

    save_results(results, filepath)
    return results


def analyze(results: list) -> None:
    print("\n" + "=" * 70)
    print("PHASE-2 SCOUT ANALYSIS")
    print("=" * 70)

    models = sorted({r["model"] for r in results})
    by_mq = {(r["model"], r["question_id"]): r for r in results}
    qids = sorted({r["question_id"] for r in results})

    total_cost = sum(r.get("cost") or 0 for r in results)
    print(f"Records: {len(results)} | total cost ${total_cost:.2f}\n")

    print("1. PER-MODEL OUTCOMES")
    for m in models:
        recs = [r for r in results if r["model"] == m]
        n = len(recs)
        counts = {}
        for r in recs:
            counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
        acc = counts.get("correct", 0) / n if n else 0
        mcost = sum(r.get("cost") or 0 for r in recs)
        print(
            f"  {m:38s} acc={acc:5.1%} (n={n})  "
            f"wrong={counts.get('wrong',0)} refused={counts.get('refused',0)} "
            f"no_answer={counts.get('no_answer',0)} error={counts.get('error',0)} "
            f"cost=${mcost:.2f}"
        )

    # Agreement profile over the 4 council members only
    print("\n2. AGREEMENT PROFILE (council v2 members only)")
    profile = {}
    minority_cells = []
    for qid in qids:
        recs = [by_mq.get((m, qid)) for m in COUNCIL_V2]
        if any(r is None or r["outcome"] in ("error", "refused", "no_answer") for r in recs):
            continue
        n_right = sum(1 for r in recs if r["is_correct"])
        profile[n_right] = profile.get(n_right, 0) + 1
        if n_right == 1:
            minority_cells.append(qid)
    for k in sorted(profile):
        print(f"  {k}/4 right: {profile[k]}")
    print(f"  MINORITY-CORRECT CELLS (exactly 1 right): {len(minority_cells)}")
    print("  (gate: >= ~40 justifies the full council run)")

    # Pairwise phi incl. side-arm
    print("\n3. PAIRWISE ERROR CORRELATION (phi) + same-wrong rate")
    for m1, m2 in combinations(models, 2):
        pairs = []
        for qid in qids:
            r1, r2 = by_mq.get((m1, qid)), by_mq.get((m2, qid))
            if not r1 or not r2:
                continue
            if r1["outcome"] in ("correct", "wrong") and r2["outcome"] in ("correct", "wrong"):
                pairs.append((r1, r2))
        if len(pairs) < 10:
            continue
        a = sum(1 for r1, r2 in pairs if r1["is_correct"] and r2["is_correct"])
        b = sum(1 for r1, r2 in pairs if r1["is_correct"] and not r2["is_correct"])
        c = sum(1 for r1, r2 in pairs if not r1["is_correct"] and r2["is_correct"])
        d = sum(1 for r1, r2 in pairs if not r1["is_correct"] and not r2["is_correct"])
        denom = ((a + b) * (c + d) * (a + c) * (b + d)) ** 0.5
        phi = (a * d - b * c) / denom if denom else float("nan")
        both_wrong = [(r1, r2) for r1, r2 in pairs if not r1["is_correct"] and not r2["is_correct"]]
        same_wrong = sum(
            1
            for r1, r2 in both_wrong
            if r1["predicted"] and r1["predicted"] == r2["predicted"]
        )
        sw = f"{same_wrong}/{len(both_wrong)}" if both_wrong else "n/a"
        tag = "  <-- INTRA-VENDOR" if ("anthropic" in m1 and "anthropic" in m2) else ""
        print(
            f"  {m1.split('/')[1]:24s} x {m2.split('/')[1]:24s} "
            f"phi={phi:+.3f}  same-wrong={sw}  (n={len(pairs)}){tag}"
        )

    if minority_cells:
        print("\n4. MINORITY-CORRECT CELL DETAIL (who was the lone dissenter?)")
        lone = {}
        for qid in minority_cells:
            for m in COUNCIL_V2:
                r = by_mq.get((m, qid))
                if r and r["is_correct"]:
                    lone[m] = lone.get(m, 0) + 1
        for m, k in sorted(lone.items(), key=lambda x: -x[1]):
            print(f"  {m:38s} lone-correct on {k} questions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase-2 HLE scouting pass")
    parser.add_argument("--smoke", action="store_true", help="5 questions only")
    parser.add_argument("--n", type=int, default=150, help="questions (default 150)")
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument(
        "--output-dir", default="experiments/results_phase2_scout"
    )
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if args.analyze_only:
        analyze(load_results(Path(args.output_dir) / "scout_results.json"))
    else:
        n = 5 if args.smoke else args.n
        outdir = (
            "experiments/results_phase2_smoke" if args.smoke else args.output_dir
        )
        results = asyncio.run(run_scout(n, outdir, args.max_concurrent))
        analyze(results)
