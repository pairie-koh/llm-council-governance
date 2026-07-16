"""Phase-2 council types: jury vs cabinet vs court vs peer review.

Consumes the stage-1 solo baselines (experiments/results_phase2_stage1/
stage1_results.json) and re-aggregates them with four governance structures.
Only "clean" questions count (all 4 council members answered with outcome
correct|wrong), and only the DISAGREEMENT questions are actually run —
unanimous questions are decided identically by every council type, so their
contribution to pool-level accuracy is computed analytically.

The four types:
- jury:        offline plurality vote over the 4 stage-1 letters ($0).
- cabinet:     one chairman (Fable 5, anthropic-direct, $0) call per question
               with the four answers anonymized as Councilor A-D.
- court:       one advocate brief per distinct proposed letter (non-Anthropic
               members round-robin, OpenRouter, paid) + one judge ruling
               (Fable 5, $0).
- peer_review: each member ranks the four anonymized answers; Borda count
               picks the winner (3 of 4 calls paid via OpenRouter).

NO refusal fallbacks: a refused or unparseable chairman/judge/advocate call
is recorded as outcome "no_answer" — never substituted with another model.
That is a hard experimental-design constraint.

Usage:
    python -m experiments.run_council_types --dry-run
    python -m experiments.run_council_types --types jury,cabinet
    python -m experiments.run_council_types --types court,peer_review --max-questions 20
    python -m experiments.run_council_types --analyze-only

Checkpointed exactly like the scout: atomic writes, .bak fallback, rerunning
resumes from completed (council_type, question_id) pairs; errored records
retry.
"""

import argparse
import asyncio
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import httpx

from backend.config import CHAIRMAN_V2_MODEL, COUNCIL_V2_MODELS
from backend.evaluation.hle import HLEBenchmark
from backend.model_router import is_anthropic_direct, query_model_routed
from experiments.run_phase2_scout import REFUSAL_MARKERS, load_results, save_results

ALL_TYPES = ("jury", "cabinet", "cabinet_opus", "court", "peer_review")

# cabinet_opus: identical mechanism to cabinet but chaired by Opus 4.8 — a
# chairman WEAKER than the council's best member. Tests whether council
# exposure transfers accuracy downhill (lifting a weak chairman) the way it
# drags a strong one down. Also anthropic-direct, so $0.
CABINET_CHAIRMEN = {
    "cabinet": CHAIRMAN_V2_MODEL,
    "cabinet_opus": "anthropic/claude-opus-4.8",
}

# Advocates are the non-Anthropic council members: the judge is Fable 5, and
# a Fable advocate arguing before a Fable judge would confound the court arm.
ADVOCATE_MODELS = [
    "openai/gpt-5.6-sol",
    "google/gemini-3.1-pro-preview",
    "x-ai/grok-4.5",
]

STAGE1_RESULTS = Path("experiments/results_phase2_stage1/stage1_results.json")
DEFAULT_OUTPUT_DIR = "experiments/results_council_types_v2"
RESULTS_FILENAME = "council_types_results.json"

SHUFFLE_SEED = 42
EXCERPT_CHARS = 1500
COST_PER_PAID_CALL = 0.223  # dry-run estimate per OpenRouter call
COUNCILOR_LABELS = ("A", "B", "C", "D")

# Mirror stage-1 concurrency: Anthropic direct is rate-limited more tightly.
ANTHROPIC_MAX_CONCURRENT = 4
OPENROUTER_MAX_CONCURRENT = 6

# Instantiated only for _extract_letter_from_response (dataset load is lazy).
_LETTER_BENCH = HLEBenchmark()

_RANKING_RE = re.compile(
    r"RANKING:\s*([A-D])\s*>\s*([A-D])\s*>\s*([A-D])\s*>\s*([A-D])",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def result_key(record: dict) -> Tuple[str, str]:
    """Checkpoint key for this experiment: (council_type, question_id)."""
    return (record["council_type"], record["question_id"])


def load_stage1(path: Path) -> List[dict]:
    """Stage-1 records for the 4 council members only (Opus side-arm dropped)."""
    return [r for r in load_results(path) if r["model"] in COUNCIL_V2_MODELS]


def partition_questions(
    stage1: List[dict],
) -> Tuple[Dict[str, Dict[str, dict]], List[str], List[str]]:
    """Split clean questions into unanimous and disagreement sets.

    Clean = all 4 members present with outcome correct|wrong. Returns
    (clean_by_question, unanimous_qids, disagreement_qids) where
    clean_by_question maps qid -> {model: stage1_record}.
    """
    by_q: Dict[str, Dict[str, dict]] = {}
    for r in stage1:
        by_q.setdefault(r["question_id"], {})[r["model"]] = r

    clean: Dict[str, Dict[str, dict]] = {}
    unanimous: List[str] = []
    disagreement: List[str] = []
    for qid in sorted(by_q):
        members = by_q[qid]
        if set(members) != set(COUNCIL_V2_MODELS):
            continue
        if any(m["outcome"] not in ("correct", "wrong") for m in members.values()):
            continue
        clean[qid] = members
        letters = {m["predicted"] for m in members.values()}
        (unanimous if len(letters) == 1 else disagreement).append(qid)
    return clean, unanimous, disagreement


def ordered_disagreement(disagreement_qids: List[str]) -> List[str]:
    """Deterministic run order: seeded shuffle over sorted qids.

    --max-questions takes a prefix of this order, so capped runs are a fixed,
    reproducible subset rather than whatever finished first.
    """
    qids = sorted(disagreement_qids)
    random.Random(SHUFFLE_SEED).shuffle(qids)
    return qids


def councilor_mapping(qid: str, models: List[str]) -> Dict[str, str]:
    """Anonymization map label -> model, shuffled per question.

    Seeded on the question id so the mapping is stable across resumes but
    varies across questions — the chairman can't learn positional habits.
    """
    shuffled = sorted(models)
    random.Random(f"{SHUFFLE_SEED}:{qid}").shuffle(shuffled)
    return dict(zip(COUNCILOR_LABELS, shuffled))


def load_question_texts(qids: Set[str]) -> Dict[str, str]:
    """Reload original question texts from HLE (stage-1 records omit them)."""
    benchmark = HLEBenchmark(exclude_categories=HLEBenchmark.PHASE2_EXCLUDED_CATEGORIES)
    texts = {q.id: q.text for q in benchmark.load_questions()}
    missing = qids - set(texts)
    if missing:
        raise RuntimeError(
            f"{len(missing)} stage-1 question ids not found in HLE reload: "
            f"{sorted(missing)[:5]}..."
        )
    return {qid: texts[qid] for qid in qids}


# ---------------------------------------------------------------------------
# Aggregation primitives
# ---------------------------------------------------------------------------

def majority_letter(letters: List[str]) -> Optional[str]:
    """Plurality winner over predicted letters; None on a tie for first place."""
    counts = Counter(letters).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


def parse_ranking(text: str) -> Optional[List[str]]:
    """Parse 'RANKING: A > B > C > D' into an ordered label list.

    Takes the LAST ranking line (models often restate the format mid-
    reasoning). Returns None if absent or if labels repeat — the ballot is
    then dropped, per design.
    """
    matches = _RANKING_RE.findall(text or "")
    if not matches:
        return None
    ranking = [label.upper() for label in matches[-1]]
    if len(set(ranking)) != len(COUNCILOR_LABELS):
        return None
    return ranking


def borda_scores(ballots: List[List[str]]) -> Dict[str, int]:
    """Borda count: best rank earns 3 points, worst earns 0 (4 candidates)."""
    scores: Dict[str, int] = {label: 0 for label in COUNCILOR_LABELS}
    for ballot in ballots:
        for position, label in enumerate(ballot):
            scores[label] += len(COUNCILOR_LABELS) - 1 - position
    return scores


def extract_letter(text: str) -> Optional[str]:
    """Extract the FINAL ANSWER letter using the benchmark's own parser."""
    return _LETTER_BENCH._extract_letter_from_response(text or "")


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

def _excerpt(response: str) -> str:
    return (response or "")[-EXCERPT_CHARS:]


def councilor_block(mapping: Dict[str, str], members: Dict[str, dict]) -> str:
    """Anonymized answers section shared by the cabinet and peer-review prompts."""
    parts = []
    for label in COUNCILOR_LABELS:
        rec = members[mapping[label]]
        parts.append(
            f"Councilor {label} answered: {rec['predicted']}\n"
            f"Councilor {label}'s reasoning (final excerpt):\n{_excerpt(rec['response'])}"
        )
    return "\n\n".join(parts)


def build_cabinet_prompt(question_text: str, mapping: Dict[str, str], members: Dict[str, dict]) -> str:
    """Chairman decides alone after reading the four anonymized answers."""
    return f"""You are the chairman of a council of four AI models. The council was asked the following question:

{question_text}

The four councilors answered independently. Their answers and reasoning excerpts:

{councilor_block(mapping, members)}

Weigh the councilors' answers and reasoning, then decide the council's final answer. Your answer must be a single letter.
End with: FINAL ANSWER: [letter]"""


def build_advocate_prompt(
    question_text: str, letter: str, supporters: List[dict], label_of: Dict[str, str]
) -> str:
    """One advocate defends one proposed letter using its supporters' reasoning."""
    support = "\n\n".join(
        f"Councilor {label_of[rec['model']]}'s reasoning (final excerpt):\n{_excerpt(rec['response'])}"
        for rec in supporters
    )
    return f"""You are an advocate before a court deciding the answer to a difficult question:

{question_text}

Your assigned position: the correct answer is {letter}.

The councilors who proposed {letter} reasoned as follows:

{support}

Write the strongest possible case that {letter} is the correct answer. Address the substance of the question directly; use the supporting reasoning where it helps, and strengthen it where it is weak. Be rigorous — the judge will compare your brief against advocates for the other proposed answers."""


def build_judge_prompt(question_text: str, briefs: Dict[str, str]) -> str:
    """Judge rules after reading one brief per proposed letter."""
    brief_block = "\n\n".join(
        f"--- Advocate for answer {letter} ---\n{brief}"
        for letter, brief in sorted(briefs.items())
    )
    return f"""You are the judge in a court deciding the answer to a difficult question:

{question_text}

Advocates have each argued for a different candidate answer:

{brief_block}

Weigh the briefs on their merits and rule on the correct answer. Your ruling must be a single letter.
End with: FINAL ANSWER: [letter]"""


def build_peer_prompt(question_text: str, mapping: Dict[str, str], members: Dict[str, dict]) -> str:
    """A member ranks the four anonymized answers (its own included, unmarked)."""
    return f"""Four councilors independently answered the following question:

{question_text}

Their answers and reasoning excerpts:

{councilor_block(mapping, members)}

Rank the four councilors' answers from best to worst, judging correctness of the final answer and soundness of the reasoning. Refer to the councilors by their labels (A, B, C, D).
End with exactly one line of the form: RANKING: X > Y > Z > W"""


# ---------------------------------------------------------------------------
# Per-question council runners
# ---------------------------------------------------------------------------

def base_record(council_type: str, qid: str, members: Dict[str, dict]) -> dict:
    """Skeleton record; runners fill council_answer/outcome/cost/responses."""
    return {
        "council_type": council_type,
        "question_id": qid,
        "ground_truth": next(iter(members.values()))["ground_truth"],
        "council_answer": None,
        "is_correct": False,
        "outcome": "no_answer",
        "member_letters": {m: r["predicted"] for m, r in members.items()},
        "cost": 0.0,
        "error": None,
        "timestamp": None,
    }


def _finalize(rec: dict, letter: str) -> dict:
    rec["council_answer"] = letter
    rec["is_correct"] = letter == rec["ground_truth"]
    rec["outcome"] = "correct" if rec["is_correct"] else "wrong"
    return rec


def _stamp(rec: dict) -> dict:
    rec["timestamp"] = datetime.now().isoformat()
    return rec


def jury_record(qid: str, members: Dict[str, dict]) -> dict:
    """Offline plurality vote; ties recorded as their own outcome."""
    rec = base_record("jury", qid, members)
    letter = majority_letter([r["predicted"] for r in members.values()])
    if letter is None:
        rec["outcome"] = "tie"  # counts as incorrect in accuracy
    else:
        _finalize(rec, letter)
    return _stamp(rec)


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
    qid: str,
    question_text: str,
    members: Dict[str, dict],
    council_type: str = "cabinet",
) -> dict:
    """One chairman call over the anonymized answers."""
    chairman = CABINET_CHAIRMEN[council_type]
    rec = base_record(council_type, qid, members)
    mapping = councilor_mapping(qid, list(members))
    rec["councilors"] = mapping
    raw = await routed_call(client, chairman, build_cabinet_prompt(question_text, mapping, members), sems)
    rec["cost"] = openrouter_cost([raw])
    rec["responses"] = {"chairman": {"model": chairman, "content": raw["content"]}}
    if raw["error"]:
        rec["error"] = f"chairman: {raw['error']}"
        rec["outcome"] = "error"
        return _stamp(rec)
    letter = None if unusable(raw) else extract_letter(raw["content"])
    if letter is None:
        rec["outcome"] = "no_answer"  # refused/unparseable chairman: no fallback
        return _stamp(rec)
    return _stamp(_finalize(rec, letter))


def plan_advocates(
    ordered_qids: List[str], clean: Dict[str, Dict[str, dict]]
) -> Dict[Tuple[str, str], str]:
    """Round-robin advocate assignment over the full ordered plan.

    Computed over ALL capped questions (not just the todo remainder) so the
    assignment is identical across resumes.
    """
    plan: Dict[Tuple[str, str], str] = {}
    counter = 0
    for qid in ordered_qids:
        for letter in sorted({r["predicted"] for r in clean[qid].values()}):
            plan[(qid, letter)] = ADVOCATE_MODELS[counter % len(ADVOCATE_MODELS)]
            counter += 1
    return plan


async def court_record(
    client: httpx.AsyncClient,
    sems: Dict[str, asyncio.Semaphore],
    qid: str,
    question_text: str,
    members: Dict[str, dict],
    advocate_plan: Dict[Tuple[str, str], str],
) -> dict:
    """One advocate brief per distinct proposed letter, then one judge ruling."""
    rec = base_record("court", qid, members)
    mapping = councilor_mapping(qid, list(members))
    rec["councilors"] = mapping
    label_of = {model: label for label, model in mapping.items()}
    letters = sorted({r["predicted"] for r in members.values()})
    responses: Dict[str, dict] = {}
    raws: List[dict] = []

    async def one_advocate(letter: str) -> Tuple[str, str, dict]:
        supporters = [members[m] for m in sorted(members) if members[m]["predicted"] == letter]
        model = advocate_plan[(qid, letter)]
        raw = await routed_call(
            client, model, build_advocate_prompt(question_text, letter, supporters, label_of), sems
        )
        return letter, model, raw

    briefs: Dict[str, Optional[str]] = {}
    for letter, model, raw in await asyncio.gather(*(one_advocate(le) for le in letters)):
        raws.append(raw)
        responses[f"advocate_{letter}"] = {"model": model, "content": raw["content"]}
        if raw["error"]:
            rec["error"] = f"advocate {letter} ({model}): {raw['error']}"
        briefs[letter] = None if unusable(raw) else raw["content"]

    rec["cost"] = openrouter_cost(raws)
    rec["responses"] = responses
    if rec["error"]:
        rec["outcome"] = "error"  # transport failure: retried on rerun
        return _stamp(rec)
    if any(brief is None for brief in briefs.values()):
        rec["outcome"] = "no_answer"  # refused advocate: no substitution
        return _stamp(rec)

    raw = await routed_call(client, CHAIRMAN_V2_MODEL, build_judge_prompt(question_text, briefs), sems)
    raws.append(raw)
    rec["cost"] = openrouter_cost(raws)
    responses["judge"] = {"model": CHAIRMAN_V2_MODEL, "content": raw["content"]}
    if raw["error"]:
        rec["error"] = f"judge: {raw['error']}"
        rec["outcome"] = "error"
        return _stamp(rec)
    letter = None if unusable(raw) else extract_letter(raw["content"])
    if letter is None:
        rec["outcome"] = "no_answer"
        return _stamp(rec)
    return _stamp(_finalize(rec, letter))


async def peer_review_record(
    client: httpx.AsyncClient,
    sems: Dict[str, asyncio.Semaphore],
    qid: str,
    question_text: str,
    members: Dict[str, dict],
) -> dict:
    """Each member ranks the anonymized answers; Borda count picks the winner."""
    rec = base_record("peer_review", qid, members)
    mapping = councilor_mapping(qid, list(members))
    rec["councilors"] = mapping
    prompt = build_peer_prompt(question_text, mapping, members)
    responses: Dict[str, dict] = {}
    raws: List[dict] = []

    async def one_reviewer(model: str) -> Tuple[str, dict]:
        return model, await routed_call(client, model, prompt, sems)

    ballots: List[List[str]] = []
    for model, raw in await asyncio.gather(*(one_reviewer(m) for m in COUNCIL_V2_MODELS)):
        raws.append(raw)
        responses[model] = {"content": raw["content"]}
        if raw["error"]:
            rec["error"] = f"reviewer {model}: {raw['error']}"
            continue
        ballot = None if unusable(raw) else parse_ranking(raw["content"])
        if ballot is not None:
            ballots.append(ballot)
        # unparseable/refused ballots are dropped, per design

    rec["cost"] = openrouter_cost(raws)
    rec["responses"] = responses
    rec["n_ballots"] = len(ballots)
    if rec["error"]:
        rec["outcome"] = "error"  # transport failure: retried on rerun
        return _stamp(rec)
    if not ballots:
        rec["outcome"] = "no_answer"
        return _stamp(rec)

    scores = borda_scores(ballots)
    rec["borda_scores"] = scores
    top = max(scores.values())
    top_letters = {
        members[mapping[label]]["predicted"]
        for label in COUNCILOR_LABELS
        if scores[label] == top
    }
    if len(top_letters) > 1:
        rec["outcome"] = "tie"  # Borda tie across different letters
        return _stamp(rec)
    return _stamp(_finalize(rec, top_letters.pop()))


# ---------------------------------------------------------------------------
# Cost estimation / dry run
# ---------------------------------------------------------------------------

def paid_calls_for(council_type: str, members: Dict[str, dict]) -> int:
    """OpenRouter calls a (type, question) cell will buy. Anthropic-direct is $0."""
    chairman_paid = 0 if is_anthropic_direct(CHAIRMAN_V2_MODEL) else 1
    if council_type == "jury":
        return 0
    if council_type in CABINET_CHAIRMEN:
        return 0 if is_anthropic_direct(CABINET_CHAIRMEN[council_type]) else 1
    if council_type == "court":
        n_letters = len({r["predicted"] for r in members.values()})
        return n_letters + chairman_paid  # advocates are all non-Anthropic
    if council_type == "peer_review":
        return sum(1 for m in COUNCIL_V2_MODELS if not is_anthropic_direct(m))
    raise ValueError(f"unknown council type: {council_type}")


def dry_run(
    types: List[str], max_questions: Optional[int], stage1_path: Path, output_dir: str
) -> dict:
    """Print the plan and cost estimate without making any calls."""
    stage1 = load_stage1(stage1_path)
    clean, unanimous, disagreement = partition_questions(stage1)
    ordered = ordered_disagreement(disagreement)
    capped = ordered[:max_questions] if max_questions is not None else ordered

    done = {
        result_key(r)
        for r in load_results(Path(output_dir) / RESULTS_FILENAME)
        if not r.get("error")
    }
    paid_calls = 0
    per_type: Dict[str, int] = {}
    for ctype in types:
        qids = ordered if ctype == "jury" else capped
        todo = [q for q in qids if (ctype, q) not in done]
        per_type[ctype] = len(todo)
        paid_calls += sum(paid_calls_for(ctype, clean[q]) for q in todo)
    est_cost = paid_calls * COST_PER_PAID_CALL

    print("DRY RUN — no calls made")
    print(f"  clean questions:        {len(clean)}")
    print(f"  unanimous questions:    {len(unanimous)}")
    print(f"  disagreement questions: {len(disagreement)}")
    if max_questions is not None:
        print(f"  capped to first {len(capped)} (seeded-shuffle order)")
    for ctype, n in per_type.items():
        print(f"  {ctype:12s} questions to run: {n}")
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

async def run_council_types(
    types: List[str],
    max_questions: Optional[int],
    output_dir: str,
    stage1_path: Path = STAGE1_RESULTS,
) -> List[dict]:
    """Run the selected council types over the disagreement questions."""
    stage1 = load_stage1(stage1_path)
    clean, unanimous, disagreement = partition_questions(stage1)
    ordered = ordered_disagreement(disagreement)
    capped = ordered[:max_questions] if max_questions is not None else ordered
    print(
        f"Stage-1: {len(clean)} clean questions | {len(unanimous)} unanimous, "
        f"{len(disagreement)} disagreement ({len(capped)} in scope for API types)"
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    filepath = out / RESULTS_FILENAME

    results = load_results(filepath)
    done = {result_key(r) for r in results if not r.get("error")}
    print(f"Resuming with {len(results)} records ({len(done)} clean)")
    results = [r for r in results if result_key(r) in done]  # errored retry

    # Jury is offline and free: always run it over ALL disagreement questions.
    if "jury" in types:
        new = [jury_record(qid, clean[qid]) for qid in ordered if ("jury", qid) not in done]
        if new:
            results.extend(new)
            save_results(results, filepath)
        print(f"jury: {len(new)} new records (offline)")

    api_types = [t for t in types if t != "jury"]
    todo = [(t, qid) for t in api_types for qid in capped if (t, qid) not in done]
    print(f"To run: {len(todo)} (type, question) cells across {api_types}")
    if not todo:
        save_results(results, filepath)
        return results

    qtexts = load_question_texts({qid for _, qid in todo})
    advocate_plan = plan_advocates(capped, clean) if "court" in api_types else {}

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

        async def worker(ctype: str, qid: str) -> None:
            nonlocal completed
            if ctype in CABINET_CHAIRMEN:
                rec = await cabinet_record(client, sems, qid, qtexts[qid], clean[qid], ctype)
            elif ctype == "court":
                rec = await court_record(
                    client, sems, qid, qtexts[qid], clean[qid], advocate_plan
                )
            else:
                rec = await peer_review_record(client, sems, qid, qtexts[qid], clean[qid])
            async with lock:
                results.append(rec)
                completed += 1
                save_results(results, filepath)
                if completed % 5 == 0 or completed == len(todo):
                    spent = sum(
                        r.get("cost") or 0
                        for r in results
                        if r["council_type"] != "jury"
                    )
                    print(
                        f"  {completed}/{len(todo)} done | OpenRouter spend ${spent:.2f}",
                        flush=True,
                    )

        await asyncio.gather(*(worker(t, q) for t, q in todo))

    save_results(results, filepath)
    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _subset_baselines(
    clean: Dict[str, Dict[str, dict]], qids: List[str]
) -> Tuple[float, float]:
    """(solo-Fable accuracy, jury accuracy) over a question subset."""
    if not qids:
        return float("nan"), float("nan")
    fable_correct = sum(1 for q in qids if clean[q][CHAIRMAN_V2_MODEL]["is_correct"])
    jury_correct = 0
    for q in qids:
        gt = next(iter(clean[q].values()))["ground_truth"]
        letter = majority_letter([r["predicted"] for r in clean[q].values()])
        if letter is not None and letter == gt:
            jury_correct += 1
    return fable_correct / len(qids), jury_correct / len(qids)


def analyze(results: List[dict], stage1: List[dict]) -> None:
    """Per-type accuracy, ties, rescue rate, cost, baselines, pool-level accuracy."""
    print("\n" + "=" * 70)
    print("COUNCIL TYPES ANALYSIS")
    print("=" * 70)

    clean, unanimous, disagreement = partition_questions(stage1)
    disagreement_set = set(disagreement)
    unanimous_correct = sum(
        1
        for qid in unanimous
        if next(iter(clean[qid].values()))["predicted"]
        == next(iter(clean[qid].values()))["ground_truth"]
    )
    minority_qids = {
        qid
        for qid in disagreement
        if sum(1 for r in clean[qid].values() if r["is_correct"]) == 1
    }
    print(
        f"Pool: {len(clean)} clean | {len(unanimous)} unanimous "
        f"({unanimous_correct} correct) | {len(disagreement)} disagreement "
        f"| {len(minority_qids)} minority-correct cells"
    )

    # Keep the latest record per (type, question); drop stale qids.
    latest: Dict[Tuple[str, str], dict] = {}
    for r in results:
        if r["question_id"] in disagreement_set:
            latest[result_key(r)] = r

    for ctype in ALL_TYPES:
        recs = [r for k, r in latest.items() if k[0] == ctype]
        if not recs:
            continue
        n = len(recs)
        correct = sum(1 for r in recs if r["is_correct"])
        ties = sum(1 for r in recs if r["outcome"] == "tie")
        no_answer = sum(1 for r in recs if r["outcome"] == "no_answer")
        errors = sum(1 for r in recs if r["outcome"] == "error")
        cost = sum(r.get("cost") or 0 for r in recs)
        subset = [r["question_id"] for r in recs]
        fable_acc, jury_acc = _subset_baselines(clean, subset)
        attempted_minority = [r for r in recs if r["question_id"] in minority_qids]
        rescued = sum(1 for r in attempted_minority if r["is_correct"])
        pool_acc = (unanimous_correct + correct) / (len(unanimous) + n)

        print(f"\n{ctype.upper()} (n={n} disagreement questions)")
        print(f"  disagreement accuracy: {correct}/{n} = {correct / n:.1%}")
        print(f"  ties: {ties} | no_answer: {no_answer} | errors: {errors}")
        if attempted_minority:
            print(
                f"  minority-correct rescue: {rescued}/{len(attempted_minority)} "
                f"= {rescued / len(attempted_minority):.1%}"
            )
        else:
            print("  minority-correct rescue: n/a (no minority cells attempted)")
        print(f"  OpenRouter cost: ${cost:.2f}")
        print(
            f"  baselines on same subset: solo-Fable {fable_acc:.1%} | jury {jury_acc:.1%}"
        )
        print(
            f"  pool-level accuracy (unanimous answered with unanimous letter): "
            f"{unanimous_correct + correct}/{len(unanimous) + n} = {pool_acc:.1%}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_types(spec: str) -> List[str]:
    """Validate the --types CSV against the known council types."""
    types = [t.strip() for t in spec.split(",") if t.strip()]
    unknown = [t for t in types if t not in ALL_TYPES]
    if unknown:
        raise SystemExit(f"Unknown council types: {unknown}. Valid: {list(ALL_TYPES)}")
    return types


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2 council types experiment")
    parser.add_argument(
        "--types",
        default=",".join(ALL_TYPES),
        help="comma-separated subset of jury,cabinet,court,peer_review",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="cap API-calling types to the first N disagreement questions "
        "(seeded-shuffle order); jury always runs on all",
    )
    parser.add_argument("--dry-run", action="store_true", help="print plan + cost, no calls")
    parser.add_argument("--stage1", default=str(STAGE1_RESULTS), help="stage-1 results path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    types = parse_types(args.types)
    stage1_path = Path(args.stage1)

    if args.analyze_only:
        analyze(
            load_results(Path(args.output_dir) / RESULTS_FILENAME),
            load_stage1(stage1_path),
        )
        return
    if args.dry_run:
        dry_run(types, args.max_questions, stage1_path, args.output_dir)
        return

    results = asyncio.run(
        run_council_types(types, args.max_questions, args.output_dir, stage1_path)
    )
    analyze(results, load_stage1(stage1_path))


if __name__ == "__main__":
    main()
