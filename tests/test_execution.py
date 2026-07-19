"""Tests for backend/execution.py.

These run REAL Python in a subprocess (that is the point of the executor).
Every snippet here is trivial, known-safe code written by the test itself —
no model calls, no network.
"""

from backend.execution import check_solution, extract_code

ADD_PROBLEM = {
    "prompt": 'def add(a, b):\n    """Return a + b."""\n',
    "test": (
        "def check(candidate):\n"
        "    assert candidate(1, 2) == 3\n"
        "    assert candidate(0, 0) == 0\n"
        "    assert candidate(-1, 1) == 0\n"
    ),
    "entry_point": "add",
}


class TestCheckSolution:
    def test_passing_solution(self):
        result = check_solution(ADD_PROBLEM, "    return a + b\n")
        assert result == {"passed": True, "error": None, "timed_out": False}

    def test_wrong_solution(self):
        result = check_solution(ADD_PROBLEM, "    return a - b\n")
        assert result["passed"] is False
        assert result["timed_out"] is False
        assert result["error"]  # assertion traceback captured

    def test_infinite_loop_times_out(self):
        result = check_solution(
            ADD_PROBLEM, "    while True:\n        pass\n", timeout=3.0
        )
        assert result["timed_out"] is True
        assert result["passed"] is False

    def test_syntax_error(self):
        result = check_solution(ADD_PROBLEM, "    return a +\n")
        assert result["passed"] is False
        assert result["timed_out"] is False
        assert result["error"]

    def test_accepts_attribute_object(self):
        class P:
            prompt = ADD_PROBLEM["prompt"]
            test = ADD_PROBLEM["test"]
            entry_point = ADD_PROBLEM["entry_point"]

        assert check_solution(P(), "    return a + b\n")["passed"] is True


class TestExtractCode:
    def test_strips_python_fences(self):
        response = "Here you go:\n```python\n    return a + b\n```\nDone."
        assert extract_code(response, "add") == "    return a + b"

    def test_strips_bare_fences(self):
        response = "```\n    return a + b\n```"
        assert extract_code(response, "add") == "    return a + b"

    def test_strips_reemitted_signature(self):
        response = "```python\ndef add(a, b):\n    return a + b\n```"
        code = extract_code(response, "add")
        assert code == "    return a + b"
        # And the recovered body must actually pass when concatenated.
        assert check_solution(ADD_PROBLEM, code)["passed"] is True

    def test_multiline_signature_reemit(self):
        response = (
            "```python\n"
            "def add(\n    a,\n    b,\n):\n"
            "    return a + b\n"
            "```"
        )
        code = extract_code(response, "add")
        assert check_solution(ADD_PROBLEM, code)["passed"] is True

    def test_body_only_passthrough(self):
        assert extract_code("    return a + b", "add") == "    return a + b"

    def test_truncated_fence_takes_remainder(self):
        response = "```python\n    return a + b\n"  # no closing fence
        assert extract_code(response, "add").strip() == "return a + b"

    def test_empty_response(self):
        assert extract_code("", "add") == ""
        assert extract_code(None, "add") == ""
