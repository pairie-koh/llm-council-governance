"""Coding council: does verification let the council beat its best member?

Consumes the stage-1 coding baselines (experiments/results_coding_stage1/
stage1_results.json) and re-aggregates them with council governance
structures. The council's job on code is SELECTION: pick which member's
solution to submit; correctness = whether the selected solution passed the
hidden unit tests (looked up from stage-1 `passed`).

Working set = the DISAGREEMENT problems: those where members' solutions differ
in pass/fail (at least one passed AND at least one failed). This is the coding
analog of MCQ disagreement and the only place selection matters — on unanimous
problems every structure submits the same-verdict solution. By construction,
on the disagreement subset the union-of-members ceiling is 100% (some member
always passed); the best single member sits below it, and the council's
selection accuracy is what we measure against both.

Structures:
- cabinet:     one chairman reads the problem + N anonymized solutions and
               selects a councilor. (anthropic-direct chair => $0)
- court:       one advocate per DISTINCT solution (non-anthropic members,
               round-robin) argues its solution is correct; the judge selects.
- peer_review: each member ranks the solutions; Borda picks the winner.

THE KEY FEATURE — the --verify flag. With --verify, the chairman/judge/
reviewers are additionally shown, for each candidate solution, the
ground-truth unit-test result (PASSED/FAILED) as an oracle — i.e. they can
"run the tests". Without it (default, read-only), they only read the code and
reason, exactly like the judged HLE query type. The hypothesis: verify-mode
should let the council approach the union-of-members ceiling (100% selection),
where read-only pure judgment cannot. Each result records its mode; the
checkpoint key includes it, so verify and read-only runs coexist.

Jury is DELIBERATELY OMITTED. A plurality/majority vote needs discrete
matching options to form an equivalence class ("count identical answers").
Open-ended code solutions have no natural "same answer" relation — two correct
solutions are usually textually different — so majority-vote is ill-defined
for code. We do not fake it with a code-similarity heuristic.

NO fallbacks: a refused or unparseable chairman/judge/reviewer call is
recorded as outcome "no_answer", never substituted with another model.

Usage:
    python -m experiments.run_coding_council --dry-run
    python -m experiments.run_coding_council --types cabinet,court,peer_review
    python -m experiments.run_coding_council --verify --types cabinet
    python -m experiments.run_coding_council --analyze-only

Checkpointed like run_council_types: atomic writes, .bak fallback, resume from
completed (council_type, verify, task_id) triples; errored records retry.
"""

import argparse
import asyncio
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import httpx

from backend.config import CHAIRMAN_V2_MODEL, COUNCIL_V2_MODELS
from backend.evaluation.humaneval import HumanEvalBenchmark
from backend.model_router import is_anthropic_direct, query_model_routed
from experiments.run_phase2_scout import REFUSAL_MARKERS, load_results, save_results

ALL_TYPES = ("cabinet", "court", "peer_review")

OPUS = "anthropic/claude-opus-4.8"

# Advocates are the non-Anthropic members (the judge is anthropic-direct; an
# anthropic advocate before an anthropic judge would confound the court arm).
ADVOCATE_MODELS = [
    "openai/gpt-5.6-sol",
    "google/gemini-3.1-pro-preview",
    "x-ai/grok-4.5",
]

STAGE1_RESULTS = Path("experiments/results_coding_stage1/stage1_results.json")
DEFAULT_OUTPUT_DIR = "experiments/results_coding_council"
RESULTS_FILENAME = "coding_council_results.json"

SHUFFLE_SEED = 42
# Coding outputs (full solutions in prompts + argued briefs) are longer than
# MCQ, so the per-call dry-run estimate is higher than the MCQ runner's.
COST_PER_PAID_CALL = 0.10
COUNCILOR_LABELS = ("A", "B", "C", "D")

# Runtime-mutable council composition (main() may override via --council).
COUNCIL: List[str] = list(COUNCIL_V2_MODELS)
CHAIR: str = CHAIRMAN_V2_MODEL
ADVOCATES: List[str] = list(ADVOCATE_MODELS)

ANTHROPIC_MAX_CONCURRENT = 4
OPENROUTER_MAX_CONCURRENT = 6

_SELECTION_RE = re.compile(r"SELECTION:\s*\[?([A-D])\]?", re.IGNORECASE)
_RANKING_RE = re.compile(
    r"RANKING:\s*([A-D])\s*>\s*([A-D])\s*>\s*([A-D])\s*>\s*([A-D])",
    re.IGNORECASE,
)


def configure_council(models: List[str], chair: Optional[str] = None) -> None:
    """Set the active council. Chair defaults to Fable if present, else Opus
    (both anthropic-direct => free judge). Advocates are the non-anthropic
    members, since an anthropic advocate before an anthropic judge would
    confound the court arm."""
    global COUNCIL, CHAIR, ADVOCATES
    if len(models) != len(COUNCILOR_LABELS):
        raise SystemExit(
            f"--council needs exactly {len(COUNCILOR_LABELS)} models "
            f"(labels {list(COUNCILOR_LABELS)}); got {len(models)}"
        )
    COUNCIL = list(models)
    CHAIR = chair or (CHAIRMAN_V2_MODEL if CHAIRMAN_V2_MODEL in COUNCIL else OPUS)
    ADVOCATES = [m for m in COUNCIL if not is_anthropic_direct(m)]


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def result_key(record: dict) -> Tuple[str, bool, str]:
    """Checkpoint key: (council_type, verify_mode, task_id)."""
    return (record["council_type"], bool(record["verify"]), record["task_id"])


def load_stage1(path: Path) -> List[dict]:
    """Stage-1 records for the active council members only."""
    return [r for r in load_results(path) if r["model"] in COUNCIL]


def partition_problems(
    stage1: List[dict],
) -> Tuple[Dict[str, Dict[str, dict]], List[str], List[str]]:
    """Split clean problems into unanimous and disagreement sets.

    Clean = all members present with outcome passed|failed. Disagreement =
    members' solutions differ in pass/fail (>=1 passed and >=1 failed).
    Returns (clean_by_task, unanimous_ids, disagreement_ids).
    """
    by_t: Dict[str, Dict[str, dict]] = {}
    for r in stage1:
        by_t.setdefault(r["task_id"], {})[r["model"]] = r

    clean: Dict[str, Dict[str, dict]] = {}
    unanimous: List[str] = []
    disagreement: List[str] = []
    for tid in sorted(by_t):
        members = by_t[tid]
        if set(members) != set(COUNCIL):
            continue
        if any(m["outcome"] not in ("passed", "failed") for m in members.values()):
            continue
        clean[tid] = members
        passes = {bool(m["passed"]) for m in members.values()}
        (unanimous if len(passes) == 1 else disagreement).append(tid)
    return clean, unanimous, disagreement


def ordered_disagreement(disagreement_ids: List[str]) -> List[str]:
    """Deterministic run order: seeded shuffle over sorted task ids."""
    tids = sorted(disagreement_ids)
    random.Random(SHUFFLE_SEED).shuffle(tids)
    return tids


def councilor_mapping(tid: str, models: List[str]) -> Dict[str, str]:
    """Anonymization map label -> model, seeded on the task id (stable across
    resumes, varying across problems)."""
    shuffled = sorted(models)
    random.Random(f"{SHUFFLE_SEED}:{tid}").shuffle(shuffled)
    return dict(zip(COUNCILOR_LABELS, shuffled))


def solution_groups(mapping: Dict[str, str], members: Dict[str, dict]) -> List[List[str]]:
    """Group councilor labels by identical completion, ordered by min label.

    Each distinct completion is one candidate "solution" for the court arm.
    The representative label of a group is min(labels).
    """
    by_code: Dict[str, List[str]] = {}
    for label in COUNCILOR_LABELS:
        code = members[mapping[label]]["completion"]
        by_code.setdefault(code, []).append(label)
    return sorted(by_code.values(), key=min)


def load_problem_prompts(tids: Set[str]) -> Dict[str, str]:
    """Reload the original problem prompts (stage-1 records omit them)."""
    benchmark = HumanEvalBenchmark()
    prompts = {p.task_id: p.prompt for p in benchmark.load_problems()}
    missing = tids - set(prompts)
    if missing:
        raise RuntimeError(
            f"{len(missing)} stage-1 task ids not found in HumanEval reload: "
            f"{sorted(missing)[:5]}..."
        )
    return {tid: prompts[tid] for tid in tids}


# ---------------------------------------------------------------------------
# Aggregation primitives
# ---------------------------------------------------------------------------

def parse_selection(text: str) -> Optional[str]:
    """Extract a 'SELECTION: X' councilor letter (last occurrence wins)."""
    matches = _SELECTION_RE.findall(text or "")
    if matches:
        return matches[-1].upper()
    match = re.search(r"councilor\s+\[?([A-D])\]?", text or "", re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def parse_ranking(text: str) -> Optional[List[str]]:
    """Parse the last 'RANKING: A > B > C > D' into an ordered label list."""
    matches = _RANKING_RE.findall(text or "")
    if not matches:
        return None
    ranking = [label.upper() for label in matches[-1]]
    if len(set(ranking)) != len(COUNCILOR_LABELS):
        return None
    return ranking


def borda_scores(ballots: List[List[str]]) -> Dict[str, int]:
    """Borda count: best rank earns 3 points, worst 0 (4 candidates)."""
    scores: Dict[str, int] = {label: 0 for label in COUNCILOR_LABELS}
    for ballot in ballots:
        for position, label in enumerate(ballot):
            scores[label] += len(COUNCILOR_LABELS) - 1 - position
    return scores


def unusable(raw: dict) -> bool:
    """True when a call produced no usable content (refusal or empty)."""
    fr = (raw.get("native_finish_reason") or raw.get("finish_reason") or "").lower()
    content = (raw.get("content") or "").strip()
    if not content or "refusal" in fr:
        return True
    return any(m in content.lower()[:300] for m in REFUSAL_MARKERS)


def openrouter_cost(raws: List[dict]) -> float:
    """Sum only OpenRouter spend; anthropic-direct calls cost the study $0."""
    return sum(
        (raw.get("cost") or 0.0)
        for raw in raws
        if raw.get("provider") == "openrouter"
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def full_solution(problem_prompt: str, rec: dict) -> str:
    """The complete function = prompt (signature + docstring) + completion."""
    return problem_prompt + rec["completion"]


def _verify_tag(rec: dict) -> str:
    """The oracle line appended to a solution when --verify is on."""
    result = "PASSED" if rec["passed"] else "FAILED"
    return f"\n[VERIFIED TEST RESULT: {result}]"


def councilor_block(
    problem_prompt: str, mapping: Dict[str, str], members: Dict[str, dict], verify: bool
) -> str:
    """Anonymized solutions section shared by cabinet and peer-review prompts."""
    parts = []
    for label in COUNCILOR_LABELS:
        rec = members[mapping[label]]
        block = (
            f"Councilor {label}'s solution:\n"
            f"```python\n{full_solution(problem_prompt, rec)}\n```"
        )
        if verify:
            block += _verify_tag(rec)
        parts.append(block)
    return "\n\n".join(parts)


def _verify_preamble(verify: bool) -> str:
    return (
        "You have run each candidate solution against the hidden unit tests; "
        "the verified result (PASSED/FAILED) is shown after each solution.\n\n"
        if verify
        else ""
    )


def build_cabinet_prompt(
    problem_prompt: str, mapping: Dict[str, str], members: Dict[str, dict], verify: bool
) -> str:
    """Chairman selects one councilor's solution."""
    return f"""You are the chairman of a council of four AI coding models. They were each asked to solve this Python problem:

{problem_prompt}

Each councilor submitted a solution:

{councilor_block(problem_prompt, mapping, members, verify)}

{_verify_preamble(verify)}Select the single councilor whose solution correctly solves the problem (passes all its tests).
End with: SELECTION: [letter]"""


def build_advocate_prompt(problem_prompt: str, sol_label: str, code: str) -> str:
    """One advocate argues that one distinct solution is correct.

    Advocates never see the oracle, verify or not: their job is to argue.
    """
    return f"""You are an advocate before a court deciding which solution correctly solves this Python problem:

{problem_prompt}

Your assigned solution — Solution {sol_label}:
```python
{code}
```

Write the strongest possible case that Solution {sol_label} correctly solves the problem and passes all its tests. Reason about edge cases and correctness concretely. The judge will compare your brief against advocates for the other candidate solutions."""


def build_judge_prompt(
    problem_prompt: str,
    briefs: Dict[str, str],
    codes: Dict[str, str],
    verdicts: Dict[str, bool],
    verify: bool,
) -> str:
    """Judge selects the correct solution after reading one brief per solution."""
    blocks = []
    for label in sorted(briefs):
        block = (
            f"--- Solution {label} ---\n"
            f"```python\n{codes[label]}\n```\n"
            f"Advocate for Solution {label}:\n{briefs[label]}"
        )
        if verify:
            result = "PASSED" if verdicts[label] else "FAILED"
            block += f"\n[VERIFIED TEST RESULT for Solution {label}: {result}]"
        blocks.append(block)
    brief_block = "\n\n".join(blocks)
    return f"""You are the judge in a court deciding which solution correctly solves this Python problem:

{problem_prompt}

Advocates have each argued for a candidate solution:

{brief_block}

{_verify_preamble(verify)}Weigh the solutions and rule on which one correctly solves the problem.
End with: SELECTION: [letter]"""


def build_peer_prompt(
    problem_prompt: str, mapping: Dict[str, str], members: Dict[str, dict], verify: bool
) -> str:
    """A member ranks the four anonymized solutions."""
    return f"""Four councilors independently solved this Python problem:

{problem_prompt}

Their solutions:

{councilor_block(problem_prompt, mapping, members, verify)}

{_verify_preamble(verify)}Rank the four solutions from best to worst by correctness (does it solve the problem and pass all tests?). Refer to the councilors by their labels (A, B, C, D).
End with exactly one line of the form: RANKING: X > Y > Z > W"""


# ---------------------------------------------------------------------------
# Per-problem council runners
# ---------------------------------------------------------------------------

def base_record(council_type: str, tid: str, verify: bool, members: Dict[str, dict]) -> dict:
    """Skeleton record; runners fill selection/outcome/cost/responses."""
    return {
        "council_type": council_type,
        "task_id": tid,
        "verify": verify,
        "selected_label": None,
        "selected_model": None,
        "passed": None,
        "is_correct": False,
        "outcome": "no_answer",
        "member_passed": {m: bool(r["passed"]) for m, r in members.items()},
        "cost": 0.0,
        "error": None,
        "timestamp": None,
    }


def _finalize(rec: dict, label: str, mapping: Dict[str, str], members: Dict[str, dict]) -> dict:
    model = mapping[label]
    passed = bool(members[model]["passed"])
    rec["selected_label"] = label
    rec["selected_model"] = model
    rec["passed"] = passed
    rec["is_correct"] = passed
    rec["outcome"] = "passed" if passed else "failed"
    return rec


def _stamp(rec: dict) -> dict:
    rec["timestamp"] = datetime.now().isoformat()
    return rec


async def routed_call(
    client: httpx.AsyncClient, model: str, prompt: str, sems: Dict[str, asyncio.Semaphore]
) -> dict:
    """query_model_routed under the provider-appropriate semaphore."""
    sem = sems["anthropic"] if is_anthropic_direct(model) else sems["openrouter"]
    async with sem:
        return await query_model_routed(client, model, prompt)


async def cabinet_record(
    client: httpx.AsyncClient,
    sems: Dict[str, asyncio.Semaphore],
    tid: str,
    problem_prompt: str,
    members: Dict[str, dict],
    verify: bool,
) -> dict:
    """One chairman call selecting among the anonymized solutions."""
    rec = base_record("cabinet", tid, verify, members)
    mapping = councilor_mapping(tid, list(members))
    rec["councilors"] = mapping
    raw = await routed_call(
        client, CHAIR, build_cabinet_prompt(problem_prompt, mapping, members, verify), sems
    )
    rec["cost"] = openrouter_cost([raw])
    rec["responses"] = {"chairman": {"model": CHAIR, "content": raw["content"]}}
    if raw["error"]:
        rec["error"] = f"chairman: {raw['error']}"
        rec["outcome"] = "error"
        return _stamp(rec)
    label = None if unusable(raw) else parse_selection(raw["content"])
    if label is None:
        rec["outcome"] = "no_answer"  # refused/unparseable: no fallback
        return _stamp(rec)
    return _stamp(_finalize(rec, label, mapping, members))


def plan_advocates(
    ordered_tids: List[str], clean: Dict[str, Dict[str, dict]]
) -> Dict[Tuple[str, str], str]:
    """Round-robin advocate assignment keyed by (task_id, representative label).

    Computed over the full ordered plan so it is identical across resumes.
    """
    plan: Dict[Tuple[str, str], str] = {}
    counter = 0
    for tid in ordered_tids:
        mapping = councilor_mapping(tid, list(clean[tid]))
        for group in solution_groups(mapping, clean[tid]):
            plan[(tid, min(group))] = ADVOCATES[counter % len(ADVOCATES)]
            counter += 1
    return plan


async def court_record(
    client: httpx.AsyncClient,
    sems: Dict[str, asyncio.Semaphore],
    tid: str,
    problem_prompt: str,
    members: Dict[str, dict],
    verify: bool,
    advocate_plan: Dict[Tuple[str, str], str],
) -> dict:
    """One advocate brief per DISTINCT solution, then one judge ruling."""
    rec = base_record("court", tid, verify, members)
    mapping = councilor_mapping(tid, list(members))
    rec["councilors"] = mapping
    groups = solution_groups(mapping, members)
    rep_labels = [min(group) for group in groups]
    codes = {
        rep: full_solution(problem_prompt, members[mapping[rep]]) for rep in rep_labels
    }
    verdicts = {rep: bool(members[mapping[rep]]["passed"]) for rep in rep_labels}
    responses: Dict[str, dict] = {}
    raws: List[dict] = []

    async def one_advocate(rep: str) -> Tuple[str, str, dict]:
        model = advocate_plan[(tid, rep)]
        raw = await routed_call(
            client, model, build_advocate_prompt(problem_prompt, rep, codes[rep]), sems
        )
        return rep, model, raw

    briefs: Dict[str, Optional[str]] = {}
    for rep, model, raw in await asyncio.gather(*(one_advocate(r) for r in rep_labels)):
        raws.append(raw)
        responses[f"advocate_{rep}"] = {"model": model, "content": raw["content"]}
        if raw["error"]:
            rec["error"] = f"advocate {rep} ({model}): {raw['error']}"
        briefs[rep] = None if unusable(raw) else raw["content"]

    rec["cost"] = openrouter_cost(raws)
    rec["responses"] = responses
    if rec["error"]:
        rec["outcome"] = "error"  # transport failure: retried on rerun
        return _stamp(rec)
    if any(brief is None for brief in briefs.values()):
        rec["outcome"] = "no_answer"  # refused advocate: no substitution
        return _stamp(rec)

    raw = await routed_call(
        client,
        CHAIR,
        build_judge_prompt(problem_prompt, briefs, codes, verdicts, verify),
        sems,
    )
    raws.append(raw)
    rec["cost"] = openrouter_cost(raws)
    responses["judge"] = {"model": CHAIR, "content": raw["content"]}
    if raw["error"]:
        rec["error"] = f"judge: {raw['error']}"
        rec["outcome"] = "error"
        return _stamp(rec)
    label = None if unusable(raw) else parse_selection(raw["content"])
    if label is None or label not in rep_labels:
        rec["outcome"] = "no_answer"
        return _stamp(rec)
    return _stamp(_finalize(rec, label, mapping, members))


async def peer_review_record(
    client: httpx.AsyncClient,
    sems: Dict[str, asyncio.Semaphore],
    tid: str,
    problem_prompt: str,
    members: Dict[str, dict],
    verify: bool,
) -> dict:
    """Each member ranks the anonymized solutions; Borda picks the winner."""
    rec = base_record("peer_review", tid, verify, members)
    mapping = councilor_mapping(tid, list(members))
    rec["councilors"] = mapping
    prompt = build_peer_prompt(problem_prompt, mapping, members, verify)
    responses: Dict[str, dict] = {}
    raws: List[dict] = []

    async def one_reviewer(model: str) -> Tuple[str, dict]:
        return model, await routed_call(client, model, prompt, sems)

    ballots: List[List[str]] = []
    for model, raw in await asyncio.gather(*(one_reviewer(m) for m in COUNCIL)):
        raws.append(raw)
        responses[model] = {"content": raw["content"]}
        if raw["error"]:
            rec["error"] = f"reviewer {model}: {raw['error']}"
            continue
        ballot = None if unusable(raw) else parse_ranking(raw["content"])
        if ballot is not None:
            ballots.append(ballot)

    rec["cost"] = openrouter_cost(raws)
    rec["responses"] = responses
    rec["n_ballots"] = len(ballots)
    if rec["error"]:
        rec["outcome"] = "error"
        return _stamp(rec)
    if not ballots:
        rec["outcome"] = "no_answer"
        return _stamp(rec)

    scores = borda_scores(ballots)
    rec["borda_scores"] = scores
    top = max(scores.values())
    winners = [label for label in COUNCILOR_LABELS if scores[label] == top]
    # Tie only counts as unresolved when the tied labels disagree on pass/fail.
    outcomes = {bool(members[mapping[label]]["passed"]) for label in winners}
    if len(outcomes) > 1:
        rec["outcome"] = "tie"
        return _stamp(rec)
    return _stamp(_finalize(rec, winners[0], mapping, members))


# ---------------------------------------------------------------------------
# Cost estimation / dry run
# ---------------------------------------------------------------------------

def paid_calls_for(council_type: str, mapping: Dict[str, str], members: Dict[str, dict]) -> int:
    """OpenRouter calls a (type, problem) cell will buy. Anthropic-direct is $0."""
    judge_paid = 0 if is_anthropic_direct(CHAIR) else 1
    if council_type == "cabinet":
        return judge_paid
    if council_type == "court":
        n_solutions = len(solution_groups(mapping, members))
        return n_solutions + judge_paid  # advocates are non-anthropic members
    if council_type == "peer_review":
        return sum(1 for m in COUNCIL if not is_anthropic_direct(m))
    raise ValueError(f"unknown council type: {council_type}")


def dry_run(
    types: List[str],
    verify: bool,
    max_problems: Optional[int],
    stage1_path: Path,
    output_dir: str,
) -> dict:
    """Print the plan and cost estimate without making any calls."""
    stage1 = load_stage1(stage1_path)
    clean, unanimous, disagreement = partition_problems(stage1)
    ordered = ordered_disagreement(disagreement)
    capped = ordered[:max_problems] if max_problems is not None else ordered

    done = {
        result_key(r)
        for r in load_results(Path(output_dir) / RESULTS_FILENAME)
        if not r.get("error")
    }
    paid_calls = 0
    per_type: Dict[str, int] = {}
    for ctype in types:
        todo = [t for t in capped if (ctype, verify, t) not in done]
        per_type[ctype] = len(todo)
        for tid in todo:
            mapping = councilor_mapping(tid, list(clean[tid]))
            paid_calls += paid_calls_for(ctype, mapping, clean[tid])
    est_cost = paid_calls * COST_PER_PAID_CALL

    print("DRY RUN — no calls made")
    print(f"  verify mode:            {verify}")
    print(f"  clean problems:         {len(clean)}")
    print(f"  unanimous problems:     {len(unanimous)}")
    print(f"  disagreement problems:  {len(disagreement)}")
    if max_problems is not None:
        print(f"  capped to first {len(capped)} (seeded-shuffle order)")
    for ctype, n in per_type.items():
        print(f"  {ctype:12s} problems to run: {n}")
    print(f"  paid OpenRouter calls:  {paid_calls}")
    print(f"  estimated cost:         ${est_cost:.2f} (@ ${COST_PER_PAID_CALL}/call)")
    return {
        "n_disagreement": len(disagreement),
        "paid_calls": paid_calls,
        "est_cost": est_cost,
        "per_type": per_type,
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

async def run_coding_council(
    types: List[str],
    verify: bool,
    max_problems: Optional[int],
    output_dir: str,
    stage1_path: Path = STAGE1_RESULTS,
) -> List[dict]:
    """Run the selected council types over the disagreement problems."""
    stage1 = load_stage1(stage1_path)
    clean, unanimous, disagreement = partition_problems(stage1)
    ordered = ordered_disagreement(disagreement)
    capped = ordered[:max_problems] if max_problems is not None else ordered
    print(
        f"Stage-1: {len(clean)} clean | {len(unanimous)} unanimous, "
        f"{len(disagreement)} disagreement ({len(capped)} in scope) | verify={verify}"
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / RESULTS_FILENAME

    results = load_results(filepath)
    done = {result_key(r) for r in results if not r.get("error")}
    print(f"Resuming with {len(results)} records ({len(done)} clean)")
    results = [r for r in results if result_key(r) in done]  # errored retry

    todo = [(t, tid) for t in types for tid in capped if (t, verify, tid) not in done]
    print(f"To run: {len(todo)} (type, problem) cells across {types}")
    if not todo:
        save_results(results, filepath)
        return results

    prompts = load_problem_prompts({tid for _, tid in todo})
    advocate_plan = plan_advocates(capped, clean) if "court" in types else {}

    sems = {
        "anthropic": asyncio.Semaphore(ANTHROPIC_MAX_CONCURRENT),
        "openrouter": asyncio.Semaphore(OPENROUTER_MAX_CONCURRENT),
    }
    lock = asyncio.Lock()
    completed = 0

    async with httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=OPENROUTER_MAX_CONCURRENT + ANTHROPIC_MAX_CONCURRENT + 4
        )
    ) as client:

        async def worker(ctype: str, tid: str) -> None:
            nonlocal completed
            if ctype == "cabinet":
                rec = await cabinet_record(
                    client, sems, tid, prompts[tid], clean[tid], verify
                )
            elif ctype == "court":
                rec = await court_record(
                    client, sems, tid, prompts[tid], clean[tid], verify, advocate_plan
                )
            else:
                rec = await peer_review_record(
                    client, sems, tid, prompts[tid], clean[tid], verify
                )
            async with lock:
                results.append(rec)
                completed += 1
                if completed % 5 == 0 or completed == len(todo):
                    save_results(results, filepath)
                    spent = sum(r.get("cost") or 0 for r in results)
                    print(
                        f"  {completed}/{len(todo)} done | OpenRouter spend ${spent:.2f}",
                        flush=True,
                    )

        await asyncio.gather(*(worker(t, tid) for t, tid in todo))

    save_results(results, filepath)
    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _best_member(clean: Dict[str, Dict[str, dict]], tids: List[str]) -> Tuple[str, float]:
    """(best member slug, its pass rate) over a problem subset."""
    if not tids:
        return "n/a", float("nan")
    best_model, best_rate = "n/a", -1.0
    for model in COUNCIL:
        rate = sum(1 for t in tids if clean[t][model]["passed"]) / len(tids)
        if rate > best_rate:
            best_model, best_rate = model, rate
    return best_model, best_rate


def _union_rate(clean: Dict[str, Dict[str, dict]], tids: List[str]) -> float:
    """Fraction of problems where at least one member passed (the ceiling)."""
    if not tids:
        return float("nan")
    return sum(
        1 for t in tids if any(clean[t][m]["passed"] for m in COUNCIL)
    ) / len(tids)


def analyze(results: List[dict], stage1: List[dict]) -> None:
    """Per (structure, verify): selection accuracy vs best member vs union."""
    print("\n" + "=" * 70)
    print("CODING COUNCIL ANALYSIS")
    print("=" * 70)

    clean, unanimous, disagreement = partition_problems(stage1)
    disagreement_set = set(disagreement)
    unanimous_passed = sum(
        1 for tid in unanimous if next(iter(clean[tid].values()))["passed"]
    )
    pool_best_model, pool_best = _best_member(clean, sorted(clean))
    pool_union = _union_rate(clean, sorted(clean))
    print(
        f"Pool: {len(clean)} clean | {len(unanimous)} unanimous "
        f"({unanimous_passed} passed) | {len(disagreement)} disagreement"
    )
    print(
        f"  best member overall: {pool_best_model.split('/')[-1]} {pool_best:.1%} "
        f"| union-of-members ceiling: {pool_union:.1%}"
    )

    # Latest record per (type, verify, task) on disagreement problems.
    latest: Dict[Tuple[str, bool, str], dict] = {}
    for r in results:
        if r["task_id"] in disagreement_set:
            latest[result_key(r)] = r

    modes = sorted({k[1] for k in latest})
    for verify in modes:
        print(f"\n--- verify={verify} ---")
        for ctype in ALL_TYPES:
            recs = [r for k, r in latest.items() if k[0] == ctype and k[1] == verify]
            if not recs:
                continue
            n = len(recs)
            passed = sum(1 for r in recs if r["is_correct"])
            ties = sum(1 for r in recs if r["outcome"] == "tie")
            no_answer = sum(1 for r in recs if r["outcome"] == "no_answer")
            errors = sum(1 for r in recs if r["outcome"] == "error")
            cost = sum(r.get("cost") or 0 for r in recs)
            subset = [r["task_id"] for r in recs]
            best_model, best_rate = _best_member(clean, subset)
            union = _union_rate(clean, subset)
            pool_acc = (unanimous_passed + passed) / (len(unanimous) + n)

            print(f"\n{ctype.upper()} (n={n} disagreement problems, verify={verify})")
            print(f"  selection accuracy:   {passed}/{n} = {passed / n:.1%}")
            print(f"  ties: {ties} | no_answer: {no_answer} | errors: {errors}")
            print(
                f"  best member on subset: {best_model.split('/')[-1]} {best_rate:.1%} "
                f"| union ceiling: {union:.1%}"
            )
            print(f"  OpenRouter cost: ${cost:.2f}")
            print(
                f"  pool-level accuracy (unanimous + selected): "
                f"{unanimous_passed + passed}/{len(unanimous) + n} = {pool_acc:.1%} "
                f"(pool best member {pool_best:.1%})"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_types(spec: str) -> List[str]:
    types = [t.strip() for t in spec.split(",") if t.strip()]
    unknown = [t for t in types if t not in ALL_TYPES]
    if unknown:
        raise SystemExit(f"Unknown council types: {unknown}. Valid: {list(ALL_TYPES)}")
    return types


def main() -> None:
    parser = argparse.ArgumentParser(description="Coding council experiment")
    parser.add_argument(
        "--types",
        default=",".join(ALL_TYPES),
        help="comma-separated subset of cabinet,court,peer_review",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="give the chairman/judge/reviewers the ground-truth unit-test "
        "result per solution (the oracle). Default: read-only judgment.",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="cap to the first N disagreement problems (seeded-shuffle order)",
    )
    parser.add_argument(
        "--council",
        default=None,
        help="comma-separated council model slugs (exactly 4). Default: the v2 "
        "council. Fable-free example: 'openai/gpt-5.6-sol,"
        "google/gemini-3.1-pro-preview,x-ai/grok-4.5,anthropic/claude-opus-4.8'.",
    )
    parser.add_argument("--chair", default=None, help="chair/judge slug override")
    parser.add_argument("--dry-run", action="store_true", help="print plan + cost")
    parser.add_argument("--stage1", default=str(STAGE1_RESULTS))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if args.council:
        configure_council(
            [m.strip() for m in args.council.split(",") if m.strip()], args.chair
        )
    elif args.chair:
        configure_council(list(COUNCIL_V2_MODELS), args.chair)

    types = parse_types(args.types)
    stage1_path = Path(args.stage1)

    if args.analyze_only:
        analyze(
            load_results(Path(args.output_dir) / RESULTS_FILENAME),
            load_stage1(stage1_path),
        )
        return
    if args.dry_run:
        dry_run(types, args.verify, args.max_problems, stage1_path, args.output_dir)
        return

    results = asyncio.run(
        run_coding_council(
            types, args.verify, args.max_problems, args.output_dir, stage1_path
        )
    )
    analyze(results, load_stage1(stage1_path))


if __name__ == "__main__":
    main()
