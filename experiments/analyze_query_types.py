"""Query-type sweep (phase-2 direction 3): where do councils work, by category?

Entirely offline: re-slices the already-paid stage-1 solo baselines and
council-types results by HLE category. Answers, per category:
  - how often the council disagrees (= how often structure matters at all)
  - how each model does solo
  - how each council structure does on the disagreement questions
  - whether unanimity predicts correctness

Usage:
    python -m experiments.analyze_query_types
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from backend.config import CHAIRMAN_V2_MODEL, COUNCIL_V2_MODELS

STAGE1 = Path("experiments/results_phase2_stage1/stage1_results.json")
COUNCIL = Path("experiments/results_council_types_v2/council_types_results.json")

COUNCIL_TYPES = ("jury", "cabinet", "cabinet_opus", "court", "peer_review")


def load(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    stage1 = [r for r in load(STAGE1) if not r.get("error")]
    council = [r for r in load(COUNCIL) if not r.get("error")]

    category: Dict[str, str] = {r["question_id"]: r["category"] for r in stage1}

    # Clean questions: all 4 members answered correct|wrong.
    by_q: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in stage1:
        if r["model"] in COUNCIL_V2_MODELS and r["outcome"] in ("correct", "wrong"):
            by_q[r["question_id"]][r["model"]] = r
    clean = {q: ms for q, ms in by_q.items() if len(ms) == 4}
    unanimous = {q for q, ms in clean.items() if len({m["predicted"] for m in ms.values()}) == 1}
    disagreement = set(clean) - unanimous

    council_by_type: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in council:
        council_by_type[r["council_type"]][r["question_id"]] = r

    cats = sorted({category[q] for q in clean}, key=lambda c: -sum(1 for q in clean if category[q] == c))

    print("=" * 100)
    print("QUERY-TYPE SWEEP: per-category council behavior (clean questions only)")
    print("=" * 100)

    # Fable@dis is the honest baseline for the council columns: all council
    # types run on the disagreement subset, so compare against solo Fable on
    # that same subset, not on the whole category.
    header = (
        f"{'category':<18} {'n':>4} {'disagree':>9} {'unan.acc':>9} "
        f"{'Fable':>7} {'Fable@dis':>10} {'best-other':>10}"
        + "".join(f" {t[:9]:>10}" for t in COUNCIL_TYPES)
    )
    print(header)
    print("-" * len(header))

    for cat in cats + ["ALL"]:
        qs = [q for q in clean if cat == "ALL" or category[q] == cat]
        if not qs:
            continue
        dis = [q for q in qs if q in disagreement]
        una = [q for q in qs if q in unanimous]
        # unanimity as a correctness signal
        una_acc = (
            sum(1 for q in una if next(iter(clean[q].values()))["predicted"]
                == clean[q][CHAIRMAN_V2_MODEL]["ground_truth"]) / len(una)
            if una else float("nan")
        )
        fable_acc = sum(clean[q][CHAIRMAN_V2_MODEL]["is_correct"] for q in qs) / len(qs)
        fable_dis = (
            sum(clean[q][CHAIRMAN_V2_MODEL]["is_correct"] for q in dis) / len(dis)
            if dis else float("nan")
        )
        others = [m for m in COUNCIL_V2_MODELS if m != CHAIRMAN_V2_MODEL]
        best_other = max(
            sum(clean[q][m]["is_correct"] for q in qs) / len(qs) for m in others
        )

        row = (
            f"{cat:<18} {len(qs):>4} {len(dis)/len(qs):>8.0%} "
            f"{una_acc:>9.0%} {fable_acc:>7.0%} {fable_dis:>10.0%} {best_other:>10.0%}"
        )
        for ctype in COUNCIL_TYPES:
            recs = [council_by_type[ctype].get(q) for q in dis]
            recs = [r for r in recs if r]
            acc = sum(r["is_correct"] for r in recs) / len(recs) if recs else float("nan")
            row += f" {acc:>10.0%}" if recs else f" {'-':>10}"
        print(row)

    # Verifiable-vs-less-verifiable proxy: math/physics/engineering vs rest.
    print()
    print("VERIFIABLE-PROXY SPLIT (Math+Physics+Engineering vs Humanities+Other+CS)")
    hard_verif = {"Math", "Physics", "Engineering"}
    for label, group in (
        ("verifiable-ish", [q for q in disagreement if category[q] in hard_verif]),
        ("open-ended-ish", [q for q in disagreement if category[q] not in hard_verif]),
    ):
        fable = sum(clean[q][CHAIRMAN_V2_MODEL]["is_correct"] for q in group) / len(group)
        line = f"  {label:<15} n={len(group):>3} | solo-Fable {fable:.0%}"
        for ctype in COUNCIL_TYPES:
            recs = [council_by_type[ctype].get(q) for q in group]
            recs = [r for r in recs if r]
            if recs:
                line += f" | {ctype} {sum(r['is_correct'] for r in recs)/len(recs):.0%}"
        print(line)


if __name__ == "__main__":
    main()
