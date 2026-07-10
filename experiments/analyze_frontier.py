"""Analysis for the frontier council experiment.

Produces the three quantities the frontier rerun exists for:

1. Accuracy by structure x benchmark (Hall's headline table, frontier edition),
   with the best-individual-model baseline computed from stage-1 responses.
2. The pairwise ERROR-CORRELATION MATRIX (research direction #1) - the
   Condorcet Jury Theorem independence test Hall's repo never computed.
   Includes a chance-agreement correction for multiple-choice benchmarks
   (same-wrong-answer rate vs the 1/(k-1) guessing floor).
3. CORRECT-MINORITY SURVIVAL (research direction #2, flagship) - on
   questions where exactly one member answered correctly at stage 1, how
   often does each governance structure's final answer preserve it?

Usage:
    python -m experiments.analyze_frontier --results-dir experiments/results_frontier_full
"""

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

from backend.evaluation.aimo import AIMOBenchmark
from backend.evaluation.base import Question
from backend.evaluation.gpqa import GPQABenchmark
from backend.evaluation.gsm8k import GSM8KBenchmark
from backend.evaluation.mmlu_pro import MMLUProBenchmark
from backend.evaluation.truthfulqa import TruthfulQABenchmark

# benchmark name (as stored in results) -> (evaluator instance, n_options)
# n_options is used for the multiple-choice chance-agreement correction;
# None means free-form answers (integer / numeric), where the chance of two
# wrong models colliding on the same wrong answer is ~0.
EVALUATORS = {
    "MMLU-Pro-Math": (MMLUProBenchmark(category="math"), 10),
    "GSM8K": (GSM8KBenchmark(), None),
    "TruthfulQA": (TruthfulQABenchmark(), 2),
    "AIMO": (AIMOBenchmark(), None),
    "GPQA-Diamond": (GPQABenchmark(subset="gpqa_diamond"), 4),
}


def grade_stage1(record):
    """Re-grade each council member's stage-1 response.

    Returns {model: (is_correct, predicted)} or None if the benchmark is
    unknown or stage-1 responses are missing.
    """
    bench_entry = EVALUATORS.get(record["benchmark"])
    stage1 = record.get("stage1_responses")
    if bench_entry is None or not isinstance(stage1, dict):
        return None
    evaluator, _ = bench_entry

    q = Question(
        id=record["question_id"],
        text="",
        ground_truth=str(record["expected"]),
    )
    graded = {}
    for model, response in stage1.items():
        if not isinstance(response, str):
            continue
        result = evaluator.evaluate(q, response)
        graded[model] = (bool(result.is_correct), result.predicted)
    return graded


def phi_coefficient(xs, ys):
    """Pearson correlation of two binary vectors (phi). None if degenerate."""
    n = len(xs)
    if n == 0:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None  # a model was always right or always wrong
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def analyze(results_dir: str):
    path = Path(results_dir) / "pilot_results.json"
    if not path.exists():
        raise SystemExit(f"No results file at {path}")
    with open(path, encoding="utf-8") as f:
        records = json.load(f)

    print("=" * 70)
    print("FRONTIER COUNCIL ANALYSIS")
    print("=" * 70)
    print(f"Records: {len(records)}\n")

    benchmarks = sorted({r["benchmark"] for r in records})
    structures = sorted({r["structure"] for r in records})

    # ------------------------------------------------------------------
    # 1. Accuracy by structure x benchmark
    # ------------------------------------------------------------------
    print("-" * 70)
    print("1. COUNCIL ACCURACY BY STRUCTURE x BENCHMARK")
    print("-" * 70)
    for bench in benchmarks:
        print(f"\n  {bench}:")
        for struct in structures:
            rows = [
                r for r in records
                if r["benchmark"] == bench and r["structure"] == struct
                and r.get("is_correct") is not None
            ]
            if not rows:
                continue
            acc = sum(bool(r["is_correct"]) for r in rows) / len(rows)
            print(f"    {struct:45s} {acc:6.1%}  (n={len(rows)})")

    # ------------------------------------------------------------------
    # Grade stage-1 responses once, keyed by (benchmark, question, structure)
    # ------------------------------------------------------------------
    stage1_graded = {}
    for r in records:
        graded = grade_stage1(r)
        if not graded:
            continue
        # Council members are provider/model slugs. Self-consistency stores
        # its temp-0.7 chairman samples as sample_0..sample_N - those are not
        # council members and must not enter member-level analyses.
        graded = {m: v for m, v in graded.items() if "/" in m}
        if len(graded) >= 2:
            stage1_graded[(r["benchmark"], r["question_id"], r["structure"])] = (
                graded,
                r,
            )

    # Stage 1 is shared across structures (paired design), so per-question
    # quantities must count each question once, not once per structure.
    stage1_by_question = {}
    for (b, q, _s), (graded, r) in stage1_graded.items():
        stage1_by_question.setdefault((b, q), (graded, r))

    # ------------------------------------------------------------------
    # 2. Individual model accuracy (stage-1, pooled across structures)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("2. INDIVIDUAL MODEL ACCURACY (stage-1 responses)")
    print("-" * 70)
    for bench in benchmarks:
        per_model = defaultdict(list)
        for (b, _q), (graded, _r) in stage1_by_question.items():
            if b != bench:
                continue
            for model, (correct, _pred) in graded.items():
                per_model[model].append(correct)
        if not per_model:
            continue
        print(f"\n  {bench}:")
        best = None
        for model, vals in sorted(per_model.items()):
            acc = sum(vals) / len(vals)
            best = max(best or 0.0, acc)
            print(f"    {model:45s} {acc:6.1%}  (n={len(vals)})")
        print(f"    {'BEST INDIVIDUAL (baseline)':45s} {best:6.1%}")

    # ------------------------------------------------------------------
    # 3. Pairwise error-correlation matrix (CJT independence test)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("3. PAIRWISE ERROR CORRELATION (CJT independence test)")
    print("   phi = correlation of correctness indicators (0 = independent)")
    print("   same-wrong = P(identical wrong answer | both wrong) vs chance")
    print("-" * 70)
    for bench in benchmarks:
        n_options = EVALUATORS.get(bench, (None, None))[1]
        # one correctness vector per model over question cells,
        # aligned on cells where all models are present
        cells = [
            (graded, r) for (b, _q), (graded, r) in stage1_by_question.items()
            if b == bench
        ]
        if not cells:
            continue
        models = sorted({m for graded, _ in cells for m in graded})
        aligned = [
            (graded, r) for graded, r in cells
            if all(m in graded for m in models)
        ]
        if len(aligned) < 5:
            print(f"\n  {bench}: too few aligned cells ({len(aligned)}) - skipped")
            continue
        print(f"\n  {bench}  (n={len(aligned)} question-cells):")
        chance = (1.0 / (n_options - 1)) if n_options else 0.0
        for m1, m2 in itertools.combinations(models, 2):
            xs = [int(g[m1][0]) for g, _ in aligned]
            ys = [int(g[m2][0]) for g, _ in aligned]
            phi = phi_coefficient(xs, ys)
            both_wrong = [
                (g[m1][1], g[m2][1])
                for g, _ in aligned
                if not g[m1][0] and not g[m2][0]
            ]
            if both_wrong:
                same_wrong = sum(p1 == p2 for p1, p2 in both_wrong) / len(both_wrong)
                sw_str = f"same-wrong={same_wrong:5.1%} (chance~{chance:5.1%}, n={len(both_wrong)})"
            else:
                sw_str = "same-wrong=n/a (never both wrong)"
            phi_str = f"{phi:+.3f}" if phi is not None else "  n/a "
            short1 = m1.split("/")[-1][:22]
            short2 = m2.split("/")[-1][:22]
            print(f"    {short1:24s} x {short2:24s} phi={phi_str}  {sw_str}")

    # ------------------------------------------------------------------
    # 4. Correct-minority survival (flagship)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("4. CORRECT-MINORITY SURVIVAL (flagship)")
    print("   Cases where EXACTLY ONE member was right at stage 1.")
    print("   Survival = council's final answer was correct anyway.")
    print("-" * 70)
    for bench in benchmarks:
        print(f"\n  {bench}:")
        any_cases = False
        for struct in structures:
            cases = []
            for (b, _q, s), (graded, r) in stage1_graded.items():
                if b != bench or s != struct:
                    continue
                n_correct = sum(c for c, _p in graded.values())
                if n_correct == 1 and len(graded) >= 3:
                    cases.append(bool(r.get("is_correct")))
            if cases:
                any_cases = True
                survival = sum(cases) / len(cases)
                print(
                    f"    {struct:45s} {survival:6.1%} survived  "
                    f"({sum(cases)}/{len(cases)} cases)"
                )
        if not any_cases:
            print("    (no minority-correct cases - benchmark may have ceilinged)")

    # ------------------------------------------------------------------
    # 5. Disagreement inventory (Coasean gate / HLE trigger)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("5. STAGE-1 AGREEMENT PROFILE (0-4 members correct)")
    print("-" * 70)
    for bench in benchmarks:
        counts = defaultdict(int)
        for (b, _q), (graded, _r) in stage1_by_question.items():
            if b != bench:
                continue
            counts[sum(c for c, _p in graded.values())] += 1
        total = sum(counts.values())
        if not total:
            continue
        dist = "  ".join(f"{k}right={counts[k]}" for k in sorted(counts))
        minority = counts.get(1, 0)
        print(f"  {bench:15s} {dist}   -> minority-correct cells: {minority}")
    print(
        "\n  (HLE trigger rule: if total minority-correct cells across AIMO+GPQA"
        "\n   is < ~40, add the HLE text-MCQ subset as phase 2.)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze frontier council results")
    parser.add_argument(
        "--results-dir",
        default="experiments/results_frontier_full",
        help="Directory containing pilot_results.json",
    )
    args = parser.parse_args()
    analyze(args.results_dir)
