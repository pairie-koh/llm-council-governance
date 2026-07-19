"""Tests for experiments/run_coding_council.py — all network calls mocked."""

import asyncio
from typing import List, Optional

import pytest

import experiments.run_coding_council as rcc
from backend.config import CHAIRMAN_V2_MODEL, COUNCIL_V2_MODELS

MODELS = COUNCIL_V2_MODELS  # gpt-5.6-sol, gemini-3.1-pro, fable-5, grok-4.5


def problem_records(tid: str, passes: List[bool], completions: Optional[List[str]] = None):
    """Stage-1 coding records for all 4 members on one problem."""
    recs = []
    for i, (m, p) in enumerate(zip(MODELS, passes)):
        comp = completions[i] if completions else f"    return {i}"
        recs.append(
            {
                "model": m,
                "task_id": tid,
                "passed": p,
                "outcome": "passed" if p else "failed",
                "completion": comp,
            }
        )
    return recs


def members_of(tid: str, passes, completions=None) -> dict:
    return {r["model"]: r for r in problem_records(tid, passes, completions)}


def raw_response(
    content: str,
    provider: str = "openrouter",
    cost: float = 0.05,
    error: Optional[str] = None,
) -> dict:
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
    monkeypatch.setattr(rcc, "is_anthropic_direct", lambda m: m.startswith("anthropic/"))


@pytest.fixture
def mock_router(monkeypatch):
    class Log(list):
        responder = staticmethod(lambda model, prompt: raw_response("SELECTION: A"))

    log = Log()

    async def fake_query(client, model, prompt, **kwargs):
        log.append({"model": model, "prompt": prompt})
        return log.responder(model, prompt)

    monkeypatch.setattr(rcc, "query_model_routed", fake_query)
    return log


def passing_label(tid: str, members: dict) -> str:
    mapping = rcc.councilor_mapping(tid, list(members))
    return next(lb for lb, m in mapping.items() if members[m]["passed"])


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------

class TestPartition:
    def test_mixed_is_disagreement(self):
        stage1 = (
            problem_records("t_unan_pass", [True, True, True, True])
            + problem_records("t_unan_fail", [False, False, False, False])
            + problem_records("t_dis", [True, False, False, True])
        )
        clean, unanimous, disagreement = rcc.partition_problems(stage1)
        assert set(clean) == {"t_unan_pass", "t_unan_fail", "t_dis"}
        assert set(unanimous) == {"t_unan_pass", "t_unan_fail"}
        assert disagreement == ["t_dis"]

    def test_dirty_problem_excluded(self):
        recs = problem_records("t", [True, False, False, True])
        recs[1]["outcome"] = "refused"  # one member did not answer cleanly
        clean, unanimous, disagreement = rcc.partition_problems(recs)
        assert clean == {} and disagreement == []

    def test_missing_member_excluded(self):
        recs = problem_records("t", [True, False, False, True])[:3]
        clean, _, _ = rcc.partition_problems(recs)
        assert clean == {}

    def test_side_arm_dropped_by_load_stage1(self, tmp_path):
        import json

        recs = problem_records("t", [True, False, False, True]) + [
            {
                "model": "anthropic/claude-opus-4.8",
                "task_id": "t",
                "passed": True,
                "outcome": "passed",
                "completion": "x",
            }
        ]
        path = tmp_path / "stage1.json"
        path.write_text(json.dumps(recs), encoding="utf-8")
        assert {r["model"] for r in rcc.load_stage1(path)} == set(MODELS)


# ---------------------------------------------------------------------------
# Cabinet
# ---------------------------------------------------------------------------

class TestCabinet:
    async def test_selection_extraction(self, mock_router):
        members = members_of("t1", [True, False, False, True])
        win = passing_label("t1", members)
        mock_router.responder = lambda model, prompt: raw_response(
            f"Reasoning. SELECTION: {win}", provider="anthropic-direct", cost=0.0
        )
        rec = await rcc.cabinet_record(None, make_sems(), "t1", "def f():\n", members, verify=False)

        assert len(mock_router) == 1
        assert mock_router[0]["model"] == CHAIRMAN_V2_MODEL
        assert rec["selected_label"] == win
        assert rec["is_correct"] is True
        assert rec["outcome"] == "passed"
        assert rec["cost"] == 0.0
        # Anonymized: real model names must not leak into the prompt.
        for model in MODELS:
            assert model not in mock_router[0]["prompt"]

    async def test_selects_failing_member(self, mock_router):
        members = members_of("t1", [True, False, False, True])
        mapping = rcc.councilor_mapping("t1", list(members))
        lose = next(lb for lb, m in mapping.items() if not members[m]["passed"])
        mock_router.responder = lambda model, prompt: raw_response(
            f"SELECTION: {lose}", provider="anthropic-direct", cost=0.0
        )
        rec = await rcc.cabinet_record(None, make_sems(), "t1", "p", members, verify=False)
        assert rec["outcome"] == "failed"
        assert rec["is_correct"] is False

    async def test_unparseable_is_no_answer(self, mock_router):
        members = members_of("t1", [True, False, False, True])
        mock_router.responder = lambda model, prompt: raw_response(
            "I have no idea.", provider="anthropic-direct", cost=0.0
        )
        rec = await rcc.cabinet_record(None, make_sems(), "t1", "p", members, verify=False)
        assert rec["outcome"] == "no_answer"
        assert rec["selected_label"] is None
        assert len(mock_router) == 1  # no substitute call

    async def test_transport_error_retryable(self, mock_router):
        members = members_of("t1", [True, False, False, True])
        mock_router.responder = lambda model, prompt: raw_response(
            "", provider="anthropic-direct", cost=0.0, error="HTTP 529"
        )
        rec = await rcc.cabinet_record(None, make_sems(), "t1", "p", members, verify=False)
        assert rec["outcome"] == "error"
        assert "529" in rec["error"]

    async def test_verify_injects_oracle(self, mock_router):
        members = members_of("t1", [True, False, False, True])
        mock_router.responder = lambda model, prompt: raw_response(
            "SELECTION: A", provider="anthropic-direct", cost=0.0
        )
        await rcc.cabinet_record(None, make_sems(), "t1", "p", members, verify=True)
        prompt = mock_router[0]["prompt"]
        assert "VERIFIED TEST RESULT" in prompt
        assert "PASSED" in prompt and "FAILED" in prompt

    async def test_readonly_no_oracle(self, mock_router):
        members = members_of("t1", [True, False, False, True])
        mock_router.responder = lambda model, prompt: raw_response(
            "SELECTION: A", provider="anthropic-direct", cost=0.0
        )
        await rcc.cabinet_record(None, make_sems(), "t1", "p", members, verify=False)
        assert "VERIFIED TEST RESULT" not in mock_router[0]["prompt"]


# ---------------------------------------------------------------------------
# Court
# ---------------------------------------------------------------------------

class TestCourt:
    def _mixed_members(self):
        # Two members share a solution ("return 1"), so 3 distinct solutions.
        completions = ["    return 1", "    return 1", "    return 2", "    return 3"]
        passes = [True, True, False, False]
        return members_of("t1", passes, completions)

    async def test_one_advocate_per_distinct_solution(self, mock_router):
        members = self._mixed_members()
        clean = {"t1": members}
        plan = rcc.plan_advocates(["t1"], clean)
        assert len(plan) == 3  # 3 distinct solutions
        assert all(m in rcc.ADVOCATE_MODELS for m in plan.values())

        mapping = rcc.councilor_mapping("t1", list(members))
        groups = rcc.solution_groups(mapping, members)
        # The passing solution is the "return 1" group.
        passing_rep = next(
            min(g) for g in groups if members[mapping[min(g)]]["passed"]
        )

        def responder(model, prompt):
            if model == CHAIRMAN_V2_MODEL:
                return raw_response(
                    f"SELECTION: {passing_rep}", provider="anthropic-direct", cost=0.0
                )
            return raw_response("A strong brief.", cost=0.05)

        mock_router.responder = responder
        rec = await rcc.court_record(
            None, make_sems(), "t1", "def f():\n", members, False, plan
        )

        advocate_calls = [c for c in mock_router if c["model"] != CHAIRMAN_V2_MODEL]
        judge_calls = [c for c in mock_router if c["model"] == CHAIRMAN_V2_MODEL]
        assert len(advocate_calls) == 3 and len(judge_calls) == 1
        assert rec["is_correct"] is True
        assert rec["outcome"] == "passed"
        assert rec["cost"] == pytest.approx(0.15)  # 3 paid advocates, free judge

    async def test_refused_advocate_is_no_answer(self, mock_router):
        members = self._mixed_members()
        plan = rcc.plan_advocates(["t1"], {"t1": members})

        def responder(model, prompt):
            if "Solution" in prompt and model != CHAIRMAN_V2_MODEL:
                return raw_response("I can't help with that.")
            return raw_response("SELECTION: A", provider="anthropic-direct", cost=0.0)

        mock_router.responder = responder
        rec = await rcc.court_record(
            None, make_sems(), "t1", "p", members, False, plan
        )
        assert rec["outcome"] == "no_answer"
        # Judge must never be called when a brief is missing.
        assert all(c["model"] != CHAIRMAN_V2_MODEL for c in mock_router)

    async def test_judge_verify_oracle(self, mock_router):
        members = self._mixed_members()
        plan = rcc.plan_advocates(["t1"], {"t1": members})
        judge_prompts = []

        def responder(model, prompt):
            if model == CHAIRMAN_V2_MODEL:
                judge_prompts.append(prompt)
                return raw_response("SELECTION: A", provider="anthropic-direct", cost=0.0)
            return raw_response("brief")

        mock_router.responder = responder
        await rcc.court_record(None, make_sems(), "t1", "p", members, True, plan)
        assert judge_prompts and "VERIFIED TEST RESULT" in judge_prompts[0]

    def test_advocate_never_sees_oracle(self):
        # Advocate prompt has no oracle field regardless of verify.
        prompt = rcc.build_advocate_prompt("p", "A", "    return 1")
        assert "VERIFIED TEST RESULT" not in prompt


# ---------------------------------------------------------------------------
# Peer review (Borda)
# ---------------------------------------------------------------------------

class TestPeerReview:
    async def test_borda_picks_passing(self, mock_router):
        members = members_of("t1", [True, False, False, True])
        win = passing_label("t1", members)
        others = [lb for lb in rcc.COUNCILOR_LABELS if lb != win]
        ranking = f"RANKING: {win} > {others[0]} > {others[1]} > {others[2]}"
        mock_router.responder = lambda model, prompt: raw_response(ranking)

        rec = await rcc.peer_review_record(None, make_sems(), "t1", "p", members, verify=False)
        assert len(mock_router) == 4  # one ballot per member
        assert {c["model"] for c in mock_router} == set(MODELS)
        assert rec["n_ballots"] == 4
        assert rec["selected_label"] == win
        assert rec["is_correct"] is True

    async def test_drops_unparseable_ballot(self, mock_router):
        members = members_of("t1", [True, False, False, True])
        win = passing_label("t1", members)
        others = [lb for lb in rcc.COUNCILOR_LABELS if lb != win]
        ranking = f"RANKING: {win} > {others[0]} > {others[1]} > {others[2]}"

        def responder(model, prompt):
            if model == MODELS[0]:
                return raw_response("no ranking here")
            return raw_response(ranking)

        mock_router.responder = responder
        rec = await rcc.peer_review_record(None, make_sems(), "t1", "p", members, verify=False)
        assert rec["n_ballots"] == 3
        assert rec["selected_label"] == win

    async def test_all_ballots_dropped_is_no_answer(self, mock_router):
        members = members_of("t1", [True, False, False, True])
        mock_router.responder = lambda model, prompt: raw_response("garbage")
        rec = await rcc.peer_review_record(None, make_sems(), "t1", "p", members, verify=False)
        assert rec["outcome"] == "no_answer"

    async def test_verify_injects_oracle(self, mock_router):
        members = members_of("t1", [True, False, False, True])
        mock_router.responder = lambda model, prompt: raw_response("garbage")
        await rcc.peer_review_record(None, make_sems(), "t1", "p", members, verify=True)
        assert all("VERIFIED TEST RESULT" in c["prompt"] for c in mock_router)


# ---------------------------------------------------------------------------
# configure_council + result_key + dry run
# ---------------------------------------------------------------------------

class TestConfigure:
    def _restore(self):
        rcc.COUNCIL = list(rcc.COUNCIL_V2_MODELS)
        rcc.CHAIR = rcc.CHAIRMAN_V2_MODEL
        rcc.ADVOCATES = list(rcc.ADVOCATE_MODELS)

    def test_fable_free_sets_opus_chair(self):
        try:
            rcc.configure_council(
                [
                    "openai/gpt-5.6-sol",
                    "google/gemini-3.1-pro-preview",
                    "x-ai/grok-4.5",
                    "anthropic/claude-opus-4.8",
                ]
            )
            assert rcc.CHAIR == rcc.OPUS
            assert "anthropic/claude-opus-4.8" not in rcc.ADVOCATES
            assert set(rcc.ADVOCATES) == {
                "openai/gpt-5.6-sol",
                "google/gemini-3.1-pro-preview",
                "x-ai/grok-4.5",
            }
        finally:
            self._restore()

    def test_wrong_size_rejected(self):
        try:
            with pytest.raises(SystemExit):
                rcc.configure_council(["a", "b"])
        finally:
            self._restore()

    def test_result_key_includes_verify(self):
        base = {"council_type": "cabinet", "task_id": "t1"}
        assert rcc.result_key({**base, "verify": True}) == ("cabinet", True, "t1")
        assert rcc.result_key({**base, "verify": False}) == ("cabinet", False, "t1")


class TestDryRun:
    def test_counts_and_cost(self, tmp_path, monkeypatch):
        import json

        recs = (
            problem_records("t_dis1", [True, False, False, True])
            + problem_records("t_dis2", [True, True, False, False])
            + problem_records("t_unan", [True, True, True, True])
        )
        path = tmp_path / "stage1.json"
        path.write_text(json.dumps(recs), encoding="utf-8")

        async def explode(*a, **k):
            raise AssertionError("dry run must not call the router")

        monkeypatch.setattr(rcc, "query_model_routed", explode)
        est = rcc.dry_run(
            ["cabinet", "court", "peer_review"], False, None, path, str(tmp_path / "out")
        )
        assert est["n_disagreement"] == 2
        # cabinet: 0 paid (fable chair). peer_review: 3 paid/problem x 2 = 6.
        # court: distinct solutions per problem (each has 4 unique "return i")
        # = 4 advocates x 2 problems = 8, judge free -> 8.
        assert est["per_type"] == {"cabinet": 2, "court": 2, "peer_review": 2}
        assert est["paid_calls"] == 0 + 8 + 6
