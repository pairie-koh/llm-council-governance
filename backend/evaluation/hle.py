"""HLE benchmark - Humanity's Last Exam, text-only multiple-choice subset."""

import random
import re
from typing import List, Optional

from backend.evaluation.base import Benchmark, EvalResult, Question


class HLEBenchmark(Benchmark):
    """
    Humanity's Last Exam (Phan et al. 2025), text-only multiple-choice subset.

    2,500 expert-written questions across math, sciences, and humanities,
    designed to be the hardest broad academic benchmark. Frontier models
    score ~45-53% as of mid-2026, which makes it the phase-2 replacement
    for the saturated MMLU/GSM8K/AIMO suite: disagreement cells exist.

    Loads the ungated text-only mirror ``macabdul9/hle_text_only`` (the
    official ``cais/hle`` is auto-gated on HuggingFace) and filters to
    ``answer_type == "multipleChoice"`` (560 questions). Questions embed
    their own answer choices; ground truth is a single letter.

    A deterministic seeded shuffle selects the n-question subset so runs
    are reproducible and resumable.

    ``exclude_categories`` drops questions whose HLE category is in the
    set. Phase-2 pre-registered exclusion: Biology/Medicine and Chemistry,
    where Claude Fable 5's bio/chem safety classifiers refuse 67-71% of
    questions (scout, 2026-07-16), versus 0-11% everywhere else. The
    filter is applied AFTER the seeded shuffle so the remaining questions
    keep their original sample order and existing checkpoint records
    stay inside the sample.
    """

    DATASET = "macabdul9/hle_text_only"

    #: Categories where the chairman model (Fable 5) refuses most questions.
    PHASE2_EXCLUDED_CATEGORIES = frozenset({"Biology/Medicine", "Chemistry"})

    def __init__(self, sample_seed: int = 42, exclude_categories: Optional[frozenset] = None):
        self.sample_seed = sample_seed
        self.exclude_categories = exclude_categories or frozenset()
        self._dataset = None

    @property
    def name(self) -> str:
        return "HLE"

    def _load_dataset(self):
        if self._dataset is None:
            from datasets import load_dataset

            ds = load_dataset(self.DATASET, split="test")
            ds = ds.filter(
                lambda x: x["answer_type"] == "multipleChoice"
                and not (x["image"] or "").strip()
            )
            self._dataset = ds
        return self._dataset

    def load_questions(self, n: Optional[int] = None) -> List[Question]:
        dataset = self._load_dataset()

        indices = list(range(len(dataset)))
        random.Random(self.sample_seed).shuffle(indices)

        questions = []
        for idx in indices:
            if n is not None and len(questions) >= n:
                break
            item = dataset[idx]
            if item.get("category", "unknown") in self.exclude_categories:
                continue
            text = f"""{item["question"].strip()}

This is an extremely difficult expert-level question. Think through it step by step.
Choose the best answer from the options above. Your answer should be a single letter.
End with: FINAL ANSWER: [letter]"""

            questions.append(
                Question(
                    id=f"hle_{item['id']}",
                    text=text,
                    ground_truth=item["answer"].strip().upper(),
                    metadata={
                        "category": item.get("category", "unknown"),
                        "raw_subject": item.get("raw_subject", "unknown"),
                    },
                )
            )

        return questions

    def _extract_letter_from_response(self, response: str) -> Optional[str]:
        """Extract a letter answer (A-J) from a model response."""
        match = re.search(r"FINAL ANSWER:\s*\[?([A-Ja-j])\]?\b", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        match = re.search(
            r"(?:the answer is|answer is|answer:)\s*\[?([A-Ja-j])\]?\b",
            response,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()

        match = re.search(r"\b([A-Ja-j])\s*[.)]?\s*$", response.strip())
        if match:
            return match.group(1).upper()

        return None

    def evaluate(self, question: Question, response: str) -> EvalResult:
        predicted = self._extract_letter_from_response(response)

        if predicted is None:
            return EvalResult(
                question_id=question.id,
                is_correct=False,
                predicted="[no letter found]",
                expected=question.ground_truth,
            )

        return EvalResult(
            question_id=question.id,
            is_correct=predicted.upper() == question.ground_truth.upper(),
            predicted=predicted,
            expected=question.ground_truth,
        )
