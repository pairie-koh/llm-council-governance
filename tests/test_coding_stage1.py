"""Tests for experiments/run_coding_stage1.py — network mocked, exec is real.

The generation call (query_model_routed) is always mocked. Execution against
unit tests is real subprocess execution of trivial, test-authored code.
"""

import json
from pathlib import Path
from typing import Optional

import pytest

import experiments.run_coding_stage1 as rcs
from backend.evaluation.humaneval import CodingProblem
from backend.execution import extract_code


def add_problem(tid: str = "HumanEval/0") -> CodingProblem:
    return CodingProblem(
        task_id=tid,
        prompt='def add(a, b):\n    """add."""\n',
        test="def check(candidate):\n    assert candidate(1, 2) == 3\n",
        entry_point="add",
        canonical_solution="    return a + b\n",
    )


def raw_response(
    content: str,
    provider: str = "openrouter",
    cost: float = 0.01,
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


@pytest.fixture(autouse=True)
def deterministic_routing(monkeypatch):
    monkeypatch.setattr(rcs, "is_anthropic_direct", lambda m: m.startswith("anthropic/"))


class TestBuildRecord:
    def _code(self, content):
        return extract_code(content, "add")

    def test_passed(self):
        raw = raw_response("```python\n    return a + b\n```")
        rec = rcs.build_record("m", add_problem(), raw, self._code(raw["content"]))
        assert rec["outcome"] == "passed"
        assert rec["passed"] is True
        assert rec["error"] is None
        assert rec["exec_error"] is None
        # Record shape.
        for key in (
            "model", "task_id", "completion", "passed", "outcome",
            "cost", "provider", "error", "timestamp",
        ):
            assert key in rec

    def test_failed_keeps_error_none(self):
        raw = raw_response("```python\n    return a - b\n```")
        rec = rcs.build_record("m", add_problem(), raw, self._code(raw["content"]))
        assert rec["outcome"] == "failed"
        assert rec["passed"] is False
        # error must stay None so a failed test is NOT retried on resume.
        assert rec["error"] is None
        assert rec["exec_error"]

    def test_refusal_is_refused_no_execution(self):
        raw = raw_response("I can't help with that.")
        rec = rcs.build_record("m", add_problem(), raw, self._code(raw["content"]))
        assert rec["outcome"] == "refused"
        assert rec["passed"] is False

    def test_empty_is_no_answer(self):
        raw = raw_response("")
        rec = rcs.build_record("m", add_problem(), raw, "")
        assert rec["outcome"] == "no_answer"

    def test_transport_error(self):
        raw = raw_response("", cost=None, error="HTTP 529")
        rec = rcs.build_record("m", add_problem(), raw, "")
        assert rec["outcome"] == "error"
        assert rec["error"] == "HTTP 529"


class FakeBenchmark:
    def __init__(self, *a, **k):
        pass

    def load_problems(self, n=None):
        probs = [add_problem("HumanEval/0"), add_problem("HumanEval/1")]
        return probs[:n] if n is not None else probs

    def build_generation_prompt(self, problem):
        return problem.prompt

    def extract_code(self, response, entry_point):
        return extract_code(response, entry_point)


class TestRunStage1:
    @pytest.fixture(autouse=True)
    def _one_model_council(self, monkeypatch):
        monkeypatch.setattr(rcs, "COUNCIL", ["openai/gpt-5.6-sol"])
        monkeypatch.setattr(rcs, "HumanEvalBenchmark", FakeBenchmark)

    async def test_generates_and_executes(self, tmp_path, monkeypatch):
        calls = []

        async def fake_query(client, model, prompt, **kwargs):
            calls.append(model)
            return raw_response("```python\n    return a + b\n```")

        monkeypatch.setattr(rcs, "query_model_routed", fake_query)
        results = await rcs.run_stage1(None, str(tmp_path), max_concurrent=2)

        assert len(results) == 2  # 1 model x 2 problems
        assert all(r["outcome"] == "passed" for r in results)
        assert {r["task_id"] for r in results} == {"HumanEval/0", "HumanEval/1"}
        assert len(calls) == 2

    async def test_resume_skips_done(self, tmp_path, monkeypatch):
        done = [
            {
                "model": "openai/gpt-5.6-sol",
                "task_id": "HumanEval/0",
                "completion": "    return a + b",
                "passed": True,
                "outcome": "passed",
                "error": None,
                "exec_error": None,
                "cost": 0.01,
                "provider": "openrouter",
            }
        ]
        (tmp_path / rcs.RESULTS_FILENAME).write_text(json.dumps(done), encoding="utf-8")

        calls = []

        async def fake_query(client, model, prompt, **kwargs):
            calls.append(model)
            return raw_response("```python\n    return a + b\n```")

        monkeypatch.setattr(rcs, "query_model_routed", fake_query)
        results = await rcs.run_stage1(None, str(tmp_path), max_concurrent=2)

        # Only HumanEval/1 remained to run.
        assert len(calls) == 1
        assert {r["task_id"] for r in results} == {"HumanEval/0", "HumanEval/1"}

    async def test_errored_record_retries(self, tmp_path, monkeypatch):
        errored = [
            {
                "model": "openai/gpt-5.6-sol",
                "task_id": "HumanEval/0",
                "completion": "",
                "passed": False,
                "outcome": "error",
                "error": "HTTP 529",
                "exec_error": None,
                "cost": None,
                "provider": "openrouter",
            }
        ]
        (tmp_path / rcs.RESULTS_FILENAME).write_text(json.dumps(errored), encoding="utf-8")

        async def fake_query(client, model, prompt, **kwargs):
            return raw_response("```python\n    return a + b\n```")

        monkeypatch.setattr(rcs, "query_model_routed", fake_query)
        results = await rcs.run_stage1(None, str(tmp_path), max_concurrent=2)

        assert all(r["error"] is None for r in results)
        assert all(r["outcome"] == "passed" for r in results)

    def test_dry_run_no_calls(self, tmp_path, monkeypatch):
        async def explode(*a, **k):
            raise AssertionError("dry run must not call the router")

        monkeypatch.setattr(rcs, "query_model_routed", explode)
        est = rcs.dry_run(None, str(tmp_path))
        assert est["n_problems"] == 2
        assert est["paid_calls"] == 2  # 1 non-anthropic model x 2 problems
