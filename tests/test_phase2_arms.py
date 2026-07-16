"""Tests for experiments/run_phase2_arms.py — all network calls mocked."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from experiments import run_phase2_arms as arms

MEMBERS = [
    "openai/gpt-5.6-sol",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-fable-5",
    "x-ai/grok-4.5",
]


def make_members(letters=("A", "B", "B", "C"), truth="A"):
    """Stage-1-shaped member dict for one question."""
    return {
        m: {
            "model": m,
            "question_id": "hle_q1",
            "ground_truth": truth,
            "predicted": letters[i],
            "is_correct": letters[i] == truth,
            "outcome": "correct" if letters[i] == truth else "wrong",
            "response": f"reasoning body for {m} " * 50,
        }
        for i, m in enumerate(MEMBERS)
    }


def raw(content, provider="openrouter", cost=0.01, error=None):
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


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

class TestPrompts:
    def test_arm_a_includes_peer_reasoning(self):
        members = make_members()
        p = arms.build_prompt("A", "What is X?", MEMBERS[0], members, "hle_q1")
        assert "Their reasoning" in p
        assert "You previously answered this question with: A" in p

    def test_arm_b_positions_only(self):
        members = make_members()
        p = arms.build_prompt("B", "What is X?", MEMBERS[0], members, "hle_q1")
        assert "reasoning" not in p.lower().replace("not their reasoning", "")
        assert "Councilor A answered:" in p

    def test_arm_d1_no_peers(self):
        members = make_members()
        p = arms.build_prompt("D1", "What is X?", MEMBERS[0], members, "hle_q1")
        assert "Councilor" not in p
        assert "reviewer has flagged" in p

    def test_arm_d2_challenge_plus_peers(self):
        members = make_members()
        p = arms.build_prompt("D2", "What is X?", MEMBERS[0], members, "hle_q1")
        assert "Councilor" in p
        assert "reviewer has flagged" in p

    def test_no_model_names_leak(self):
        members = make_members()
        for arm in arms.ALL_ARMS:
            p = arms.build_prompt(arm, "What is X?", MEMBERS[0], members, "hle_q1")
            body = p.replace(f"reasoning body for", "")  # excerpts quote raw text
            for m in MEMBERS:
                # model slugs must never appear as labels (excerpts aside)
                assert f"{m} answered" not in body

    def test_own_answer_excluded_from_peers(self):
        members = make_members(letters=("A", "B", "C", "D"))
        p = arms.build_prompt("B", "What is X?", MEMBERS[0], members, "hle_q1")
        # member 0 answered A; only B, C, D should appear as councilor positions
        assert "answered: A" not in p

    def test_anonymization_deterministic(self):
        members = make_members()
        p1 = arms.build_prompt("A", "Q", MEMBERS[0], members, "hle_q1")
        p2 = arms.build_prompt("A", "Q", MEMBERS[0], members, "hle_q1")
        assert p1 == p2


# ---------------------------------------------------------------------------
# Record semantics
# ---------------------------------------------------------------------------

def run_record(arm, member, members, mock_raw):
    sems = {"anthropic": asyncio.Semaphore(1), "openrouter": asyncio.Semaphore(1)}
    with patch.object(arms, "query_model_routed", new=AsyncMock(return_value=mock_raw)):
        return asyncio.run(
            arms.arm_record(None, sems, arm, member, "hle_q1", "What is X?", members)
        )


class TestArmRecord:
    def test_kept_answer(self):
        members = make_members()
        rec = run_record("A", MEMBERS[0], members, raw("thinking... FINAL ANSWER: A"))
        assert rec["outcome"] == "revised"
        assert rec["after"] == "A" and not rec["flipped"]
        assert rec["before_correct"] and rec["after_correct"]

    def test_capitulation_flip(self):
        members = make_members()  # member 0 correct with A
        rec = run_record("A", MEMBERS[0], members, raw("FINAL ANSWER: B"))
        assert rec["flipped"] and rec["before_correct"] and not rec["after_correct"]

    def test_correction_flip(self):
        members = make_members()  # member 1 wrong with B
        rec = run_record("A", MEMBERS[1], members, raw("FINAL ANSWER: A"))
        assert rec["flipped"] and not rec["before_correct"] and rec["after_correct"]

    def test_refusal_is_no_answer_not_flip(self):
        members = make_members()
        r = raw("")
        r["finish_reason"] = "refusal"
        r["native_finish_reason"] = "refusal"
        rec = run_record("A", MEMBERS[0], members, r)
        assert rec["outcome"] == "no_answer" and rec["after"] is None and not rec["flipped"]

    def test_transport_error_retryable(self):
        members = make_members()
        rec = run_record("A", MEMBERS[0], members, raw("", error="boom"))
        assert rec["outcome"] == "error" and rec["error"] == "boom"

    def test_anthropic_cost_zero(self):
        members = make_members()
        rec = run_record("A", MEMBERS[2], members, raw("FINAL ANSWER: B", provider="anthropic-direct", cost=1.23))
        assert rec["cost"] == 0.0


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

class TestParseArms:
    def test_d_expands(self):
        assert arms.parse_arms("A,D") == ["A", "D1", "D2"]

    def test_dedup(self):
        assert arms.parse_arms("D,D1") == ["D1", "D2"]

    def test_unknown_rejected(self):
        with pytest.raises(SystemExit):
            arms.parse_arms("A,Z")


class TestCheckpointResume:
    def test_done_cells_skipped(self, tmp_path, monkeypatch):
        members = make_members()
        stage1 = [dict(r, category="Math") for r in members.values()]
        # second question so disagreement partition is non-trivial
        for m, letters in zip(MEMBERS, ("A", "A", "A", "A")):
            stage1.append(dict(members[m], question_id="hle_q2", predicted=letters,
                               is_correct=letters == "A", outcome="correct"))
        s1path = tmp_path / "stage1.json"
        s1path.write_text(json.dumps(stage1))
        monkeypatch.setattr(arms, "STAGE1_RESULTS", s1path)
        monkeypatch.setattr(arms, "load_question_texts", lambda qids: {q: "Q?" for q in qids})

        outdir = tmp_path / "out"
        outdir.mkdir()
        # pre-seed one completed cell
        pre = [{
            "arm": "B", "model": MEMBERS[0], "question_id": "hle_q1",
            "ground_truth": "A", "before": "A", "before_correct": True,
            "after": "A", "after_correct": True, "flipped": False,
            "outcome": "revised", "cost": 0.0, "error": None, "timestamp": "t",
        }]
        (outdir / arms.RESULTS_FILENAME).write_text(json.dumps(pre))

        calls = []

        async def fake(client, model, prompt, **kw):
            calls.append(model)
            return raw("FINAL ANSWER: A")

        with patch.object(arms, "query_model_routed", new=fake):
            results = asyncio.run(arms.run_arms(["B"], None, str(outdir)))

        # 4 members x 1 disagreement question, minus 1 pre-seeded = 3 calls
        assert len(calls) == 3
        assert len([r for r in results if r["arm"] == "B"]) == 4


class TestDryRun:
    def test_no_calls(self, tmp_path, monkeypatch, capsys):
        members = make_members()
        stage1 = [dict(r, category="Math") for r in members.values()]
        s1path = tmp_path / "stage1.json"
        s1path.write_text(json.dumps(stage1))
        monkeypatch.setattr(arms, "STAGE1_RESULTS", s1path)
        called = AsyncMock()
        with patch.object(arms, "query_model_routed", new=called):
            arms.dry_run(["A", "B", "D1", "D2"], None, str(tmp_path / "none"))
        called.assert_not_called()
        out = capsys.readouterr().out
        # 1 disagreement Q x 3 paid members x 4 arms = 12 paid calls
        assert "paid OpenRouter calls: 12" in out
