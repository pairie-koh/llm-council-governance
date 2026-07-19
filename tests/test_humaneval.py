"""Tests for backend/evaluation/humaneval.py — dataset load is mocked."""

import datasets
import pytest

from backend.evaluation.humaneval import CodingProblem, HumanEvalBenchmark


class FakeDataset:
    """Minimal stand-in for a HuggingFace Dataset (len + integer indexing)."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _row(i: int) -> dict:
    return {
        "task_id": f"HumanEval/{i}",
        "prompt": f'def f{i}(x):\n    """Problem {i}."""\n',
        "test": f"def check(candidate):\n    assert candidate(0) == {i}\n",
        "entry_point": f"f{i}",
        "canonical_solution": f"    return {i}\n",
    }


@pytest.fixture
def mock_dataset(monkeypatch):
    rows = [_row(i) for i in range(8)]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: FakeDataset(rows))
    return rows


class TestLoad:
    def test_parses_rows(self, mock_dataset):
        problems = HumanEvalBenchmark().load_problems()
        assert len(problems) == 8
        assert all(isinstance(p, CodingProblem) for p in problems)
        by_id = {p.task_id: p for p in problems}
        p3 = by_id["HumanEval/3"]
        assert p3.entry_point == "f3"
        assert p3.canonical_solution == "    return 3\n"
        assert "Problem 3" in p3.prompt

    def test_subset_n(self, mock_dataset):
        problems = HumanEvalBenchmark().load_problems(n=3)
        assert len(problems) == 3

    def test_deterministic_seeded_subset(self, mock_dataset):
        a = HumanEvalBenchmark(sample_seed=42).load_problems(n=4)
        b = HumanEvalBenchmark(sample_seed=42).load_problems(n=4)
        assert [p.task_id for p in a] == [p.task_id for p in b]

    def test_seed_shuffles_order(self, mock_dataset):
        # A seeded shuffle should generally not return raw dataset order.
        ids = [p.task_id for p in HumanEvalBenchmark(sample_seed=42).load_problems()]
        raw = [f"HumanEval/{i}" for i in range(8)]
        assert set(ids) == set(raw)
        assert ids != raw

    def test_different_seed_different_subset(self, mock_dataset):
        a = [p.task_id for p in HumanEvalBenchmark(sample_seed=1).load_problems(n=3)]
        b = [p.task_id for p in HumanEvalBenchmark(sample_seed=2).load_problems(n=3)]
        assert a != b


class TestPromptAndExtract:
    def test_generation_prompt_includes_problem_and_format(self):
        prob = CodingProblem("HumanEval/0", "def f(x):\n    pass\n", "t", "f", "s")
        prompt = HumanEvalBenchmark().build_generation_prompt(prob)
        assert "def f(x):" in prompt
        assert "code block" in prompt

    def test_extract_code_delegates(self):
        bench = HumanEvalBenchmark()
        response = "```python\ndef add(a, b):\n    return a + b\n```"
        assert bench.extract_code(response, "add") == "    return a + b"

    def test_evaluate_runs_tests(self):
        prob = CodingProblem(
            task_id="HumanEval/x",
            prompt='def add(a, b):\n    """add."""\n',
            test="def check(candidate):\n    assert candidate(1, 1) == 2\n",
            entry_point="add",
            canonical_solution="    return a + b\n",
        )
        assert HumanEvalBenchmark().evaluate(prob, "    return a + b\n")["passed"] is True
        assert HumanEvalBenchmark().evaluate(prob, "    return a - b\n")["passed"] is False
