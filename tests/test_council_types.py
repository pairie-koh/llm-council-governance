"""Tests for experiments/run_council_types.py — all network calls mocked."""

import asyncio
import json
from pathlib import Path
from typing import List, Optional

import pytest

import experiments.run_council_types as rct
from backend.config import CHAIRMAN_V2_MODEL, COUNCIL_V2_MODELS

MODELS = COUNCIL_V2_MODELS  # gpt-5.6-sol, gemini-3.1-pro, fable-5, grok-4.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def s1(model: str, qid: str, gt: str, pred: str, outcome: Optional[str] = None) -> dict:
    """One synthetic stage-1 record."""
    correct = pred == gt
    return {
        "model": model,
        "question_id": qid,
        "ground_truth": gt,
        "predicted": pred,
        "is_correct": correct,
        "outcome": outcome or ("correct" if correct else "wrong"),
        # No model name in the text: excerpts are quoted verbatim in prompts,
        # and the anonymization tests assert model names never appear there.
        "response": f"Some reasoning about {qid}. FINAL ANSWER: {pred}",
        "category": "Math",
    }


def question_records(qid: str, gt: str, preds: List[str]) -> List[dict]:
    """Stage-1 records for all 4 members on one question."""
    return [s1(m, qid, gt, p) for m, p in zip(MODELS, preds)]


def raw_response(
    content: str,
    provider: str = "openrouter",
    cost: float = 0.01,
    error: Optional[str] = None,
) -> dict:
    """A query_model_routed-shaped return value."""
    return {
        "content": content,
        "finish_reason": "stop",
        "native_finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 10,
        "reasoning_tokens": None,
        "cost": cost,
        "provider": provider,
        "error": error,
    }


def make_sems() -> dict:
    return {"anthropic": asyncio.Semaphore(4), "openrouter": asyncio.Semaphore(6)}


@pytest.fixture(autouse=True)
def deterministic_routing(monkeypatch):
    """Provider routing must not depend on whether ANTHROPIC_API_KEY is set."""
    monkeypatch.setattr(
        rct, "is_anthropic_direct", lambda m: m.startswith("anthropic/")
    )


@pytest.fixture
def mock_router(monkeypatch):
    """Patch query_model_routed where run_council_types imported it.

    Returns a call log; tests set log.responder to control replies.
    """

    class Log(list):
        responder = staticmethod(lambda model, prompt: raw_response("FINAL ANSWER: A"))

    log = Log()

    async def fake_query(client, model, prompt, **kwargs):
        log.append({"model": model, "prompt": prompt})
        return log.responder(model, prompt)

    monkeypatch.setattr(rct, "query_model_routed", fake_query)
    return log


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------

class TestPartition:
    def test_unanimous_vs_disagreement(self):
        stage1 = (
            question_records("q_unan", "A", ["A", "A", "A", "A"])
            + question_records("q_dis", "B", ["A", "A", "B", "C"])
        )
        clean, unanimous, disagreement = rct.partition_questions(stage1)
        assert set(clean) == {"q_unan", "q_dis"}
        assert unanimous == ["q_unan"]
        assert disagreement == ["q_dis"]

    def test_dirty_question_excluded(self):
        # One member refused -> question is not clean.
        recs = question_records("q_ref", "A", ["A", "A", "B", "A"])
        recs[2]["outcome"] = "refused"
        clean, unanimous, disagreement = rct.partition_questions(recs)
        assert clean == {} and unanimous == [] and disagreement == []

    def test_missing_member_excluded(self):
        recs = question_records("q_part", "A", ["A", "A", "B", "A"])[:3]
        clean, unanimous, disagreement = rct.partition_questions(recs)
        assert clean == {}

    def test_side_arm_dropped_by_load_stage1(self, tmp_path):
        recs = question_records("q1", "A", ["A", "A", "A", "A"]) + [
            s1("anthropic/claude-opus-4.8", "q1", "A", "B")
        ]
        path = tmp_path / "stage1_results.json"
        path.write_text(json.dumps(recs), encoding="utf-8")
        loaded = rct.load_stage1(path)
        assert {r["model"] for r in loaded} == set(MODELS)

    def test_ordered_disagreement_deterministic(self):
        qids = [f"q{i}" for i in range(10)]
        assert rct.ordered_disagreement(qids) == rct.ordered_disagreement(list(reversed(qids)))


# ---------------------------------------------------------------------------
# Jury
# ---------------------------------------------------------------------------

class TestJury:
    def test_majority_three_one(self):
        members = {r["model"]: r for r in question_records("q1", "A", ["A", "A", "A", "B"])}
        rec = rct.jury_record("q1", members)
        assert rec["council_answer"] == "A"
        assert rec["is_correct"] is True
        assert rec["outcome"] == "correct"
        assert rec["member_letters"] == dict(zip(MODELS, ["A", "A", "A", "B"]))

    def test_plurality_two_one_one(self):
        members = {r["model"]: r for r in question_records("q1", "C", ["B", "B", "A", "C"])}
        rec = rct.jury_record("q1", members)
        assert rec["council_answer"] == "B"
        assert rec["outcome"] == "wrong"

    def test_two_two_tie(self):
        members = {r["model"]: r for r in question_records("q1", "A", ["A", "A", "B", "B"])}
        rec = rct.jury_record("q1", members)
        assert rec["outcome"] == "tie"
        assert rec["council_answer"] is None
        assert rec["is_correct"] is False

    def test_all_different_tie(self):
        members = {r["model"]: r for r in question_records("q1", "A", ["A", "B", "C", "D"])}
        assert rct.jury_record("q1", members)["outcome"] == "tie"


# ---------------------------------------------------------------------------
# Ranking parse + Borda
# ---------------------------------------------------------------------------

class TestBorda:
    def test_parse_ranking_valid(self):
        assert rct.parse_ranking("blah\nRANKING: B > a > D > C") == ["B", "A", "D", "C"]

    def test_parse_ranking_takes_last(self):
        text = "RANKING: A > B > C > D is the format.\nRANKING: D > C > B > A"
        assert rct.parse_ranking(text) == ["D", "C", "B", "A"]

    def test_parse_ranking_unparseable(self):
        assert rct.parse_ranking("I think Councilor A is best.") is None
        assert rct.parse_ranking("") is None
        assert rct.parse_ranking("RANKING: A > B > B > D") is None  # repeat label

    def test_borda_scores(self):
        ballots = [["A", "B", "C", "D"], ["A", "C", "B", "D"], ["B", "A", "C", "D"]]
        scores = rct.borda_scores(ballots)
        assert scores == {"A": 8, "B": 6, "C": 4, "D": 0}

    async def test_peer_review_aggregation(self, mock_router):
        members = {r["model"]: r for r in question_records("q1", "B", ["A", "A", "B", "C"])}
        mapping = rct.councilor_mapping("q1", list(members))
        # Label whose model predicted B must win the Borda count.
        winner_label = next(
            lb for lb, m in mapping.items() if members[m]["predicted"] == "B"
        )
        others = [lb for lb in rct.COUNCILOR_LABELS if lb != winner_label]
        ranking = f"RANKING: {winner_label} > {others[0]} > {others[1]} > {others[2]}"
        mock_router.responder = lambda model, prompt: raw_response(ranking)

        rec = await rct.peer_review_record(None, make_sems(), "q1", "Question?", members)
        assert len(mock_router) == 4  # one ranking call per member
        assert {c["model"] for c in mock_router} == set(MODELS)
        assert rec["n_ballots"] == 4
        assert rec["council_answer"] == "B"
        assert rec["is_correct"] is True

    async def test_peer_review_drops_unparseable_ballot(self, mock_router):
        members = {r["model"]: r for r in question_records("q1", "B", ["A", "A", "B", "C"])}
        mapping = rct.councilor_mapping("q1", list(members))
        winner_label = next(
            lb for lb, m in mapping.items() if members[m]["predicted"] == "B"
        )
        others = [lb for lb in rct.COUNCILOR_LABELS if lb != winner_label]
        ranking = f"RANKING: {winner_label} > {others[0]} > {others[1]} > {others[2]}"

        def responder(model, prompt):
            if model == MODELS[0]:
                return raw_response("no ranking here, sorry")  # dropped ballot
            return raw_response(ranking)

        mock_router.responder = responder
        rec = await rct.peer_review_record(None, make_sems(), "q1", "Question?", members)
        assert rec["n_ballots"] == 3
        assert rec["council_answer"] == "B"

    async def test_peer_review_all_ballots_dropped(self, mock_router):
        members = {r["model"]: r for r in question_records("q1", "B", ["A", "A", "B", "C"])}
        mock_router.responder = lambda model, prompt: raw_response("garbage")
        rec = await rct.peer_review_record(None, make_sems(), "q1", "Question?", members)
        assert rec["outcome"] == "no_answer"
        assert rec["council_answer"] is None


# ---------------------------------------------------------------------------
# Cabinet
# ---------------------------------------------------------------------------

class TestCabinet:
    async def test_extracts_answer(self, mock_router):
        members = {r["model"]: r for r in question_records("q1", "B", ["A", "A", "B", "C"])}
        mock_router.responder = lambda model, prompt: raw_response(
            "Councilor C is right. FINAL ANSWER: B",
            provider="anthropic-direct",
            cost=0.0,
        )
        rec = await rct.cabinet_record(None, make_sems(), "q1", "The question?", members)

        assert len(mock_router) == 1
        call = mock_router[0]
        assert call["model"] == CHAIRMAN_V2_MODEL
        assert "The question?" in call["prompt"]
        for label in rct.COUNCILOR_LABELS:
            assert f"Councilor {label} answered:" in call["prompt"]
        # Anonymized: real model names must not leak into the prompt.
        for model in MODELS:
            assert model not in call["prompt"]

        assert rec["council_answer"] == "B"
        assert rec["is_correct"] is True
        assert rec["outcome"] == "correct"
        assert rec["cost"] == 0.0  # anthropic-direct never bills OpenRouter

    async def test_refusal_is_no_answer_no_fallback(self, mock_router):
        members = {r["model"]: r for r in question_records("q1", "B", ["A", "A", "B", "C"])}
        mock_router.responder = lambda model, prompt: raw_response(
            "I can't help with that.", provider="anthropic-direct", cost=0.0
        )
        rec = await rct.cabinet_record(None, make_sems(), "q1", "Q?", members)
        assert rec["outcome"] == "no_answer"
        assert rec["council_answer"] is None
        assert len(mock_router) == 1  # no substitute model was called

    async def test_transport_error_marked_for_retry(self, mock_router):
        members = {r["model"]: r for r in question_records("q1", "B", ["A", "A", "B", "C"])}
        mock_router.responder = lambda model, prompt: raw_response(
            "", provider="anthropic-direct", cost=0.0, error="HTTP 529"
        )
        rec = await rct.cabinet_record(None, make_sems(), "q1", "Q?", members)
        assert rec["outcome"] == "error"
        assert "529" in rec["error"]

    def test_mapping_deterministic_and_question_varying(self):
        m1 = rct.councilor_mapping("q1", list(MODELS))
        assert m1 == rct.councilor_mapping("q1", list(reversed(MODELS)))
        assert sorted(m1.values()) == sorted(MODELS)
        # Across many questions the mapping must not be constant.
        mappings = {tuple(rct.councilor_mapping(f"q{i}", list(MODELS)).values()) for i in range(20)}
        assert len(mappings) > 1


# ---------------------------------------------------------------------------
# Court
# ---------------------------------------------------------------------------

class TestCourt:
    async def test_one_advocate_per_distinct_letter(self, mock_router):
        members = {r["model"]: r for r in question_records("q1", "B", ["A", "A", "B", "C"])}
        clean = {"q1": members}
        plan = rct.plan_advocates(["q1"], clean)
        assert set(plan) == {("q1", "A"), ("q1", "B"), ("q1", "C")}
        assert all(m in rct.ADVOCATE_MODELS for m in plan.values())

        def responder(model, prompt):
            if model == CHAIRMAN_V2_MODEL:
                return raw_response(
                    "Ruling. FINAL ANSWER: B", provider="anthropic-direct", cost=0.0
                )
            return raw_response("A strong brief.", cost=0.05)

        mock_router.responder = responder
        rec = await rct.court_record(None, make_sems(), "q1", "Q?", members, plan)

        # 3 distinct letters -> 3 advocate calls + 1 judge call.
        assert len(mock_router) == 4
        advocate_calls = [c for c in mock_router if c["model"] != CHAIRMAN_V2_MODEL]
        judge_calls = [c for c in mock_router if c["model"] == CHAIRMAN_V2_MODEL]
        assert len(advocate_calls) == 3 and len(judge_calls) == 1
        defended = {
            letter
            for letter in "ABC"
            for c in advocate_calls
            if f"the correct answer is {letter}" in c["prompt"]
        }
        assert defended == {"A", "B", "C"}
        assert rec["council_answer"] == "B"
        assert rec["is_correct"] is True
        assert rec["cost"] == pytest.approx(0.15)  # 3 paid advocates, free judge

    def test_advocate_round_robin_deterministic(self):
        clean = {
            "q1": {r["model"]: r for r in question_records("q1", "B", ["A", "A", "B", "C"])},
            "q2": {r["model"]: r for r in question_records("q2", "A", ["A", "B", "A", "B"])},
        }
        plan = rct.plan_advocates(["q1", "q2"], clean)
        # 3 letters for q1 then 2 for q2, round-robin over 3 advocates.
        assert plan[("q1", "A")] == rct.ADVOCATE_MODELS[0]
        assert plan[("q1", "B")] == rct.ADVOCATE_MODELS[1]
        assert plan[("q1", "C")] == rct.ADVOCATE_MODELS[2]
        assert plan[("q2", "A")] == rct.ADVOCATE_MODELS[0]
        assert plan[("q2", "B")] == rct.ADVOCATE_MODELS[1]
        assert plan == rct.plan_advocates(["q1", "q2"], clean)

    async def test_refused_advocate_is_no_answer(self, mock_router):
        members = {r["model"]: r for r in question_records("q1", "B", ["A", "A", "B", "C"])}
        plan = rct.plan_advocates(["q1"], {"q1": members})

        def responder(model, prompt):
            if "the correct answer is B" in prompt:
                return raw_response("I can't help with that.")
            return raw_response("Brief.")

        mock_router.responder = responder
        rec = await rct.court_record(None, make_sems(), "q1", "Q?", members, plan)
        assert rec["outcome"] == "no_answer"
        # Judge must never be called when a brief is missing; no substitution.
        assert all(c["model"] != CHAIRMAN_V2_MODEL for c in mock_router)


# ---------------------------------------------------------------------------
# Checkpointing + dry run
# ---------------------------------------------------------------------------

def write_stage1(tmp_path: Path, records: List[dict]) -> Path:
    path = tmp_path / "stage1_results.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


class TestCheckpointAndDryRun:
    def _stage1_two_disagreements(self, tmp_path: Path) -> Path:
        recs = (
            question_records("q_dis1", "B", ["A", "A", "B", "C"])
            + question_records("q_dis2", "A", ["A", "B", "A", "B"])
            + question_records("q_unan", "A", ["A", "A", "A", "A"])
        )
        return write_stage1(tmp_path, recs)

    async def test_resume_skips_done_pairs(self, tmp_path, monkeypatch, mock_router):
        stage1_path = self._stage1_two_disagreements(tmp_path)
        outdir = tmp_path / "out"
        outdir.mkdir()
        done = [
            {
                "council_type": "cabinet",
                "question_id": "q_dis1",
                "ground_truth": "B",
                "council_answer": "B",
                "is_correct": True,
                "outcome": "correct",
                "member_letters": {},
                "cost": 0.0,
                "error": None,
            },
            {
                "council_type": "jury",
                "question_id": "q_dis1",
                "ground_truth": "B",
                "council_answer": "A",
                "is_correct": False,
                "outcome": "wrong",
                "member_letters": {},
                "cost": 0.0,
                "error": None,
            },
        ]
        (outdir / rct.RESULTS_FILENAME).write_text(json.dumps(done), encoding="utf-8")
        monkeypatch.setattr(
            rct, "load_question_texts", lambda qids: {q: f"text {q}" for q in qids}
        )
        mock_router.responder = lambda model, prompt: raw_response(
            "FINAL ANSWER: A", provider="anthropic-direct", cost=0.0
        )

        results = await rct.run_council_types(
            ["jury", "cabinet"], None, str(outdir), stage1_path
        )

        # Cabinet only ran on the not-done question.
        assert len(mock_router) == 1
        cabinet = [r for r in results if r["council_type"] == "cabinet"]
        assert {r["question_id"] for r in cabinet} == {"q_dis1", "q_dis2"}
        # Jury not duplicated: exactly one record per disagreement question.
        jury = [r for r in results if r["council_type"] == "jury"]
        assert sorted(r["question_id"] for r in jury) == ["q_dis1", "q_dis2"]

    async def test_errored_record_retries(self, tmp_path, monkeypatch, mock_router):
        stage1_path = self._stage1_two_disagreements(tmp_path)
        outdir = tmp_path / "out"
        outdir.mkdir()
        errored = [
            {
                "council_type": "cabinet",
                "question_id": "q_dis1",
                "ground_truth": "B",
                "council_answer": None,
                "is_correct": False,
                "outcome": "error",
                "member_letters": {},
                "cost": 0.0,
                "error": "HTTP 529",
            }
        ]
        (outdir / rct.RESULTS_FILENAME).write_text(json.dumps(errored), encoding="utf-8")
        monkeypatch.setattr(
            rct, "load_question_texts", lambda qids: {q: f"text {q}" for q in qids}
        )
        mock_router.responder = lambda model, prompt: raw_response(
            "FINAL ANSWER: B", provider="anthropic-direct", cost=0.0
        )

        results = await rct.run_council_types(["cabinet"], None, str(outdir), stage1_path)

        assert len(mock_router) == 2  # q_dis1 retried + q_dis2
        cabinet = [r for r in results if r["council_type"] == "cabinet"]
        assert all(r["error"] is None for r in cabinet)

    async def test_max_questions_caps_api_types(self, tmp_path, monkeypatch, mock_router):
        stage1_path = self._stage1_two_disagreements(tmp_path)
        outdir = tmp_path / "out"
        outdir.mkdir()
        monkeypatch.setattr(
            rct, "load_question_texts", lambda qids: {q: f"text {q}" for q in qids}
        )
        mock_router.responder = lambda model, prompt: raw_response(
            "FINAL ANSWER: A", provider="anthropic-direct", cost=0.0
        )

        results = await rct.run_council_types(
            ["jury", "cabinet"], 1, str(outdir), stage1_path
        )

        assert len(mock_router) == 1  # cabinet capped to first question in order
        expected_first = rct.ordered_disagreement(["q_dis1", "q_dis2"])[0]
        cabinet = [r for r in results if r["council_type"] == "cabinet"]
        assert [r["question_id"] for r in cabinet] == [expected_first]
        # Jury is offline and always covers the full disagreement set.
        jury = [r for r in results if r["council_type"] == "jury"]
        assert len(jury) == 2

    def test_dry_run_makes_zero_calls(self, tmp_path, monkeypatch):
        stage1_path = self._stage1_two_disagreements(tmp_path)

        async def explode(*args, **kwargs):
            raise AssertionError("dry run must not call the router")

        monkeypatch.setattr(rct, "query_model_routed", explode)
        monkeypatch.setattr(
            rct,
            "load_question_texts",
            lambda qids: (_ for _ in ()).throw(AssertionError("no HLE load in dry run")),
        )

        est = rct.dry_run(
            list(rct.ALL_TYPES), None, stage1_path, str(tmp_path / "out")
        )

        assert est["n_disagreement"] == 2
        # q_dis1: court 3 letters, q_dis2: 2 letters -> 5 advocate calls.
        # peer_review: 3 paid per question -> 6. cabinet + jury + judges: 0.
        assert est["paid_calls"] == 11
        assert est["est_cost"] == pytest.approx(11 * rct.COST_PER_PAID_CALL)

    def test_parse_types_rejects_unknown(self):
        with pytest.raises(SystemExit):
            rct.parse_types("jury,senate")
        assert rct.parse_types("jury, cabinet") == ["jury", "cabinet"]
