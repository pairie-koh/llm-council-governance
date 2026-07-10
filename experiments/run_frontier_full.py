"""Frontier council experiment: Hall's full benchmark suite + hard benchmarks.

Reruns the original llm-council-governance experiment (all 7 governance
structures) with 2026 frontier models, on:

  1. MMLU-Pro Math (150 Q)  - Hall's primary benchmark, apples-to-apples
  2. GSM8K (40 Q)           - Hall's pilot benchmark (expected to ceiling)
  3. TruthfulQA (40 Q)      - Hall's pilot benchmark (expected to ceiling)
  4. AIMO Level 5 (100 Q)   - hard olympiad math, integer answers
  5. GPQA-Diamond (198 Q)   - hard PhD science (gated on HF - needs HF_TOKEN)

Council models come from backend.config (USE_CHEAP_MODELS=false in .env).

Usage:
    python -m experiments.run_frontier_full            # full run
    python -m experiments.run_frontier_full --smoke    # 5 Q/benchmark gate
    python -m experiments.run_frontier_full --skip-gpqa  # if HF access not set up

Checkpointing: results are saved incrementally to a single
pilot_results.json; rerunning resumes from completed trials.
"""

import argparse
import asyncio
from datetime import datetime

from backend.config import CHAIRMAN_MODEL, COUNCIL_MODELS, WEIGHTS_FILE
from backend.evaluation.aimo import AIMOBenchmark
from backend.evaluation.gpqa import GPQABenchmark
from backend.evaluation.gsm8k import GSM8KBenchmark
from backend.evaluation.mmlu_pro import MMLUProBenchmark
from backend.evaluation.truthfulqa import TruthfulQABenchmark
from backend.governance import (
    AgendaSetterVetoStructure,
    DeliberateSynthesizeStructure,
    DeliberateVoteStructure,
    IndependentRankSynthesize,
    MajorityVoteStructure,
    SelfConsistencyVoteStructure,
    WeightedMajorityVote,
)
from experiments.run_pilot import run_experiment

# (benchmark factory, n_questions, include_self_consistency)
# Hall's suite runs all 7 structures (faithful replication, incl. the
# self-consistency control). The added hard benchmarks skip self-consistency:
# it is a known-worst control (Hall: 68.9%, dead last), has no original
# number to compare against on these benchmarks, and 9x chairman samples of
# olympiad-length reasoning is the single most expensive arm (~$150-190).
BENCHMARK_SPEC = [
    (lambda: MMLUProBenchmark(category="math"), 150, True),
    (lambda: GSM8KBenchmark(), 40, True),
    (lambda: TruthfulQABenchmark(), 40, True),
    (lambda: AIMOBenchmark(), 100, False),
    (lambda: GPQABenchmark(subset="gpqa_diamond"), 198, False),
]

SMOKE_N = 5  # questions per benchmark in smoke mode


def build_structures(include_self_consistency: bool = True):
    """The 7 governance structures from Hall's final MMLU-Pro experiment.

    Self-Consistency uses the chairman (best frontier model) as its base,
    mirroring Hall's final run where one model was both chairman and
    self-consistency base.
    """
    structures = [
        IndependentRankSynthesize(COUNCIL_MODELS, CHAIRMAN_MODEL),
        MajorityVoteStructure(COUNCIL_MODELS, CHAIRMAN_MODEL),
        DeliberateVoteStructure(COUNCIL_MODELS, CHAIRMAN_MODEL),
        DeliberateSynthesizeStructure(COUNCIL_MODELS, CHAIRMAN_MODEL),
        WeightedMajorityVote(COUNCIL_MODELS, CHAIRMAN_MODEL, weights_file=WEIGHTS_FILE),
        AgendaSetterVetoStructure(COUNCIL_MODELS, CHAIRMAN_MODEL),
    ]
    if include_self_consistency:
        structures.append(
            SelfConsistencyVoteStructure(
                base_model=CHAIRMAN_MODEL,
                n_samples=9,
                temperature=0.7,
            )
        )
    return structures


async def run_frontier_full(
    smoke: bool = False,
    skip_gpqa: bool = False,
    output_dir: str = None,
    max_concurrent: int = 3,
):
    """Run the full frontier experiment, one benchmark at a time.

    Sequential per-benchmark calls share one output_dir/results file, so
    the (benchmark, question_id, structure, replication) resume keys give
    suite-wide checkpointing.
    """
    if output_dir is None:
        output_dir = (
            "experiments/results_frontier_smoke"
            if smoke
            else "experiments/results_frontier_full"
        )

    print("=== Frontier Council Experiment ===")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Mode: {'SMOKE (' + str(SMOKE_N) + ' Q/benchmark)' if smoke else 'FULL'}")
    print(f"Output: {output_dir}")
    print(f"Council: {COUNCIL_MODELS}")
    print(f"Chairman: {CHAIRMAN_MODEL}")
    print()

    all_results = None
    for factory, n_questions, include_sc in BENCHMARK_SPEC:
        benchmark = factory()
        if skip_gpqa and benchmark.name.startswith("GPQA"):
            print(f"--- Skipping {benchmark.name} (--skip-gpqa) ---")
            continue
        n = SMOKE_N if smoke else n_questions
        structures = build_structures(include_self_consistency=include_sc)

        print(
            f"--- {benchmark.name}: {n} questions x {len(structures)} structures"
            f"{'' if include_sc else ' (self-consistency skipped)'} ---"
        )
        try:
            all_results = await run_experiment(
                structures=structures,
                benchmarks=[benchmark],
                n_questions=n,
                n_replications=1,
                output_dir=output_dir,
                max_concurrent=max_concurrent,
            )
        except RuntimeError as e:
            # A benchmark failing to load (e.g. GPQA without HF auth) should
            # not kill the rest of the suite.
            print(f"!!! {benchmark.name} failed: {e}")
            print("!!! Continuing with remaining benchmarks.")
            continue

    print(f"\nExperiment complete: {datetime.now().isoformat()}")
    if all_results is not None:
        print(f"Total trials in results file: {len(all_results)}")
    print(f"\nNext: python -m experiments.analyze_frontier --results-dir {output_dir}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frontier council experiment")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Smoke test: {SMOKE_N} questions per benchmark",
    )
    parser.add_argument(
        "--skip-gpqa",
        action="store_true",
        help="Skip GPQA-Diamond (if HF gated access is not set up)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Max concurrent trials (default 3, matching Hall's final run)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_frontier_full(
            smoke=args.smoke,
            skip_gpqa=args.skip_gpqa,
            output_dir=args.output_dir,
            max_concurrent=args.max_concurrent,
        )
    )
