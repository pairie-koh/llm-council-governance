"""Tests for the offline analysis passes in experiments.analyze_frontier."""

from experiments.analyze_frontier import (
    grade_answer,
    majority_winner,
    phi_coefficient,
    subset_vote_stats,
)


class TestMajorityWinner:
    def test_strict_majority(self):
        assert majority_winner(["A", "A", "B"]) == "A"

    def test_tie_returns_none(self):
        assert majority_winner(["A", "B"]) is None

    def test_three_way_tie_returns_none(self):
        assert majority_winner(["A", "B", "C"]) is None

    def test_normalizes_case_and_whitespace(self):
        assert majority_winner([" a", "A ", "b"]) == "A"

    def test_empty(self):
        assert majority_winner([]) is None


class TestPhiCoefficient:
    def test_perfect_correlation(self):
        assert phi_coefficient([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0

    def test_perfect_anticorrelation(self):
        assert phi_coefficient([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0

    def test_degenerate_constant_vector(self):
        assert phi_coefficient([1, 1, 1], [1, 0, 1]) is None

    def test_empty(self):
        assert phi_coefficient([], []) is None


class TestSubsetVoteStats:
    def _cells(self):
        # 4 questions, 3 models. m1 always right; m2 and m3 correlated:
        # both wrong on q3/q4 with the SAME wrong answer.
        return [
            {"m1": (True, "A"), "m2": (True, "A"), "m3": (True, "A")},
            {"m1": (True, "B"), "m2": (True, "B"), "m3": (True, "B")},
            {"m1": (True, "C"), "m2": (False, "X"), "m3": (False, "X")},
            {"m1": (True, "D"), "m2": (False, "Y"), "m3": (False, "Y")},
        ]

    def test_full_council_vote_loses_to_correlated_majority(self):
        # m2+m3 outvote the lone correct m1 on q3/q4 -> vote acc 50%
        stats = subset_vote_stats(self._cells(), ["m1", "m2", "m3"], sizes=[3])
        assert len(stats) == 1
        assert stats[0]["vote_acc"] == 0.5
        assert stats[0]["mean_individual_acc"] == (1.0 + 0.5 + 0.5) / 3

    def test_tie_scored_wrong(self):
        cells = [{"m1": (True, "A"), "m2": (False, "B")}]
        stats = subset_vote_stats(cells, ["m1", "m2"], sizes=[2])
        assert stats[0]["vote_acc"] == 0.0

    def test_subset_count(self):
        stats = subset_vote_stats(self._cells(), ["m1", "m2", "m3"], sizes=[2])
        assert len(stats) == 3  # 3 choose 2

    def test_default_sizes_are_triples_plus_full(self):
        cells = [
            {m: (True, "A") for m in ("m1", "m2", "m3", "m4")}
            for _ in range(3)
        ]
        stats = subset_vote_stats(cells, ["m1", "m2", "m3", "m4"])
        sizes = sorted({s["size"] for s in stats})
        assert sizes == [3, 4]
        assert len([s for s in stats if s["size"] == 3]) == 4


class TestGradeAnswer:
    def test_correct_mcq_answer(self):
        assert grade_answer("MMLU-Pro-Math", "q1", "H", "H") is True

    def test_wrong_mcq_answer(self):
        assert grade_answer("MMLU-Pro-Math", "q1", "H", "A") is False

    def test_numeric_answer(self):
        assert grade_answer("AIMO", "q1", "-15", "-15") is True

    def test_none_answer_is_wrong(self):
        assert grade_answer("AIMO", "q1", "-15", None) is False

    def test_empty_answer_is_wrong(self):
        assert grade_answer("GSM8K", "q1", "42", "  ") is False

    def test_unknown_benchmark_is_wrong(self):
        assert grade_answer("NoSuchBench", "q1", "42", "42") is False
