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


def grade_answer(record_benchmark, question_id, expected, answer):
    """Grade a bare extracted answer string via the benchmark's own evaluator.

    Wraps the answer in the FINAL ANSWER format every evaluator parses first,
    so normalization (letter case, numeric formats) matches the main pipeline.
    """
    bench_entry = EVALUATORS.get(record_benchmark)
    if bench_entry is None or answer is None or str(answer).strip() == "":
        return False
    evaluator, _ = bench_entry
    q = Question(id=question_id, text="", ground_truth=str(expected))
    result = evaluator.evaluate(q, f"FINAL ANSWER: {answer}")
    return bool(result.is_correct)


def majority_winner(answers):
    """Strict-majority answer among the given strings, or None on a tie.

    Ties are scored as wrong downstream - the conservative choice, noted in
    the section headers.
    """
    counts = defaultdict(int)
    for a in answers:
        counts[str(a).strip().upper()] += 1
    if not counts:
        return None
    best = max(counts.values())
    winners = [a for a, c in counts.items() if c == best]
    return winners[0] if len(winners) == 1 else None


def subset_vote_stats(question_cells, models, sizes=None):
    """Offline majority-vote accuracy + mean pairwise phi for every subset.

    question_cells: list of graded dicts {model: (correct, predicted)} with
    all `models` present, one per question. Returns a list of dicts, one per
    subset of each requested size (default: 3-model subsets - research
    direction #1's actual hypothesis test - plus the full council), each with
    mean individual accuracy, majority-vote accuracy (tie = wrong), and mean
    pairwise phi.
    """
    out = []
    if sizes is None:
        sizes = sorted({3, len(models)})
    for size in sizes:
        for subset in itertools.combinations(sorted(models), size):
            vote_correct = []
            for graded in question_cells:
                winner = majority_winner([graded[m][1] for m in subset])
                correct_answers = {
                    str(graded[m][1]).strip().upper()
                    for m in subset
                    if graded[m][0]
                }
                vote_correct.append(winner is not None and winner in correct_answers)
            indiv = [
                sum(graded[m][0] for graded in question_cells) / len(question_cells)
                for m in subset
            ]
            phis = [
                phi_coefficient(
                    [int(g[m1][0]) for g in question_cells],
                    [int(g[m2][0]) for g in question_cells],
                )
                for m1, m2 in itertools.combinations(subset, 2)
            ]
            phis = [p for p in phis if p is not None]
            out.append(
                {
                    "subset": subset,
                    "size": size,
                    "mean_individual_acc": sum(indiv) / len(indiv),
                    "vote_acc": sum(vote_correct) / len(vote_correct),
                    "mean_phi": (sum(phis) / len(phis)) if phis else None,
                }
            )
    return out


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

    # ------------------------------------------------------------------
    # 6. Direction #1 hypothesis test: least-correlated vs most-accurate
    #    3-model subset, via offline majority vote over cached stage 1
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("6. SUBSET VOTE TEST (direction #1: decorrelation beats accuracy?)")
    print("   Offline majority vote over cached stage-1 answers; tie = wrong.")
    print("-" * 70)
    for bench in benchmarks:
        cells = [
            graded for (b, _q), (graded, _r) in stage1_by_question.items()
            if b == bench
        ]
        if len(cells) < 5:
            continue
        models = sorted({m for g in cells for m in g})
        cells = [g for g in cells if all(m in g for m in models)]
        stats = subset_vote_stats(cells, models)
        triples = [s for s in stats if s["size"] == 3 and s["mean_phi"] is not None]
        print(f"\n  {bench} (n={len(cells)} questions):")
        for s in sorted(stats, key=lambda s: -s["vote_acc"]):
            names = "+".join(m.split("/")[-1][:12] for m in s["subset"])
            phi_str = f"{s['mean_phi']:+.3f}" if s["mean_phi"] is not None else "  n/a "
            print(
                f"    {names:42s} vote={s['vote_acc']:6.1%}  "
                f"indiv-mean={s['mean_individual_acc']:6.1%}  phi={phi_str}"
            )
        if len(triples) >= 2:
            by_acc = max(triples, key=lambda s: s["mean_individual_acc"])
            by_phi = min(triples, key=lambda s: s["mean_phi"])
            verdict = (
                "DECORRELATION WINS"
                if by_phi["vote_acc"] > by_acc["vote_acc"]
                else "accuracy wins" if by_acc["vote_acc"] > by_phi["vote_acc"]
                else "tied"
            )
            print(
                f"    -> least-phi triple {by_phi['vote_acc']:.1%} vs "
                f"highest-indiv triple {by_acc['vote_acc']:.1%}: {verdict}"
            )

    # ------------------------------------------------------------------
    # 7. Committee-size sweep (CJT diminishing returns)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("7. COMMITTEE-SIZE SWEEP (mean offline vote accuracy by N; tie = wrong)")
    print("-" * 70)
    for bench in benchmarks:
        cells = [
            graded for (b, _q), (graded, _r) in stage1_by_question.items()
            if b == bench
        ]
        if len(cells) < 5:
            continue
        models = sorted({m for g in cells for m in g})
        cells = [g for g in cells if all(m in g for m in models)]
        line = []
        for size in range(1, len(models) + 1):
            stats = subset_vote_stats(cells, models, sizes=[size])
            accs = [s["vote_acc"] for s in stats]
            line.append(f"N={size}: {sum(accs) / len(accs):6.1%}")
        print(f"  {bench:15s} " + "  ".join(line))

    # ------------------------------------------------------------------
    # 8. Recognition vs generation (direction #3)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("8. RECOGNITION vs GENERATION (direction #3)")
    print("   recognition = ranker's top-ranked peer answer was correct")
    print("   generation  = ranker's own stage-1 accuracy")
    print("-" * 70)
    rank_struct = "Independent → Rank → Synthesize"
    for bench in benchmarks:
        per_model = defaultdict(lambda: {"reco": [], "gen": [], "self_top": 0, "n": 0})
        for (b, q, s), (graded, r) in stage1_graded.items():
            if b != bench or s != rank_struct:
                continue
            s2 = r.get("stage2_data") or {}
            rankings = s2.get("rankings") or {}
            label_to_model = s2.get("label_to_model") or {}
            for ranker, order in rankings.items():
                if ranker not in graded or not order:
                    continue
                top_model = label_to_model.get(order[0])
                if top_model not in graded:
                    continue
                st = per_model[ranker]
                st["reco"].append(graded[top_model][0])
                st["gen"].append(graded[ranker][0])
                st["n"] += 1
                if top_model == ranker:
                    st["self_top"] += 1
        if not per_model:
            continue
        print(f"\n  {bench}:")
        for model, st in sorted(per_model.items()):
            reco = sum(st["reco"]) / len(st["reco"])
            gen = sum(st["gen"]) / len(st["gen"])
            print(
                f"    {model.split('/')[-1]:24s} recognition={reco:6.1%}  "
                f"generation={gen:6.1%}  gap={reco - gen:+6.1%}  "
                f"self-top={st['self_top']}/{st['n']}"
            )

    # Deliberation flips: stage-1 answer vs post-deliberation answer
    print("\n  DELIBERATION FLIPS (stage-1 -> post-deliberation, both Deliberate structures):")
    flip = defaultdict(lambda: defaultdict(int))
    for (b, q, s), (graded, r) in stage1_graded.items():
        if "Deliberate" not in s:
            continue
        s2 = r.get("stage2_data") or {}
        post = s2.get("extracted_answers") or {}
        for model, (was_correct, _pred) in graded.items():
            if model not in post:
                continue
            now_correct = grade_answer(b, q, r["expected"], post[model])
            flip[model][(was_correct, now_correct)] += 1
    for model, counts in sorted(flip.items()):
        total = sum(counts.values())
        ww = counts.get((False, True), 0)   # learned
        rw = counts.get((True, False), 0)   # herded away from correct
        print(
            f"    {model.split('/')[-1]:24s} wrong->right={ww:3d}  "
            f"right->wrong={rw:3d}  (n={total})"
        )

    # ------------------------------------------------------------------
    # 9. Coasean gate simulation (direction #8)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("9. COASEAN GATE (direction #8): route to solo answer when stage 1 agrees")
    print("   gate fires when >= k of 4 members give the same answer")
    print("-" * 70)
    for bench in benchmarks:
        q_cells = {
            q: (graded, r)
            for (b, q), (graded, r) in stage1_by_question.items()
            if b == bench
        }
        if len(q_cells) < 5:
            continue
        print(f"\n  {bench} (n={len(q_cells)} questions):")
        for k in (4, 3):
            routed_solo = 0
            gated_correct_by_struct = defaultdict(list)
            for q, (graded, _r) in q_cells.items():
                answers = [str(p).strip().upper() for _c, p in graded.values()]
                top_count = max(
                    sum(1 for a in answers if a == ans) for ans in set(answers)
                )
                consensus = top_count >= k
                if consensus:
                    routed_solo += 1
                    winner = majority_winner(
                        [p for _c, p in graded.values()]
                    )
                    correct_answers = {
                        str(p).strip().upper() for c, p in graded.values() if c
                    }
                    solo_ok = winner is not None and winner in correct_answers
                for struct in structures:
                    rec = stage1_graded.get((bench, q, struct))
                    if rec is None:
                        continue
                    final_ok = bool(rec[1].get("is_correct"))
                    gated_correct_by_struct[struct].append(
                        solo_ok if consensus else final_ok
                    )
            frac = routed_solo / len(q_cells)
            print(f"    gate k={k}: routes {frac:6.1%} of questions to solo")
            for struct, vals in sorted(gated_correct_by_struct.items()):
                if not vals:
                    continue
                always = [
                    bool(stage1_graded[(bench, q, struct)][1].get("is_correct"))
                    for q in q_cells
                    if (bench, q, struct) in stage1_graded
                ]
                print(
                    f"      {struct:43s} gated={sum(vals) / len(vals):6.1%}  "
                    f"always-council={sum(always) / len(always):6.1%}"
                )

    # ------------------------------------------------------------------
    # 10. Veto-threshold sweep (jury theory: unanimity worse than majority?)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("10. VETO-THRESHOLD SWEEP (Agenda Setter + Veto, re-scored offline)")
    print("    reject proposal iff vetoes >= k; fallback = member majority vote")
    print("-" * 70)
    veto_struct = "Agenda Setter + Veto"
    for bench in benchmarks:
        rows = [
            r for r in records
            if r["benchmark"] == bench and r["structure"] == veto_struct
            and r.get("stage3_data")
        ]
        if len(rows) < 5:
            continue
        line = []
        for k in range(1, 5):
            correct = []
            for r in rows:
                s2 = r.get("stage2_data") or {}
                s3 = r["stage3_data"]
                vetoes = int(s3.get("veto_count") or 0)
                if vetoes >= k:
                    members = s2.get("extracted_answers") or {}
                    answer = majority_winner(list(members.values()))
                else:
                    answer = s2.get("chair_proposal")
                correct.append(
                    grade_answer(bench, r["question_id"], r["expected"], answer)
                )
            line.append(f"k={k}: {sum(correct) / len(correct):6.1%}")
        print(f"  {bench:15s} " + "  ".join(line))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze frontier council results")
    parser.add_argument(
        "--results-dir",
        default="experiments/results_frontier_full",
        help="Directory containing pilot_results.json",
    )
    args = parser.parse_args()
    analyze(args.results_dir)
