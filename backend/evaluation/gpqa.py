"""GPQA benchmark - Graduate-level Google-proof Q&A (PhD science)."""

import random
import re
from typing import List, Optional

from backend.evaluation.base import Benchmark, EvalResult, Question


class GPQABenchmark(Benchmark):
    """
    GPQA: A Graduate-Level Google-Proof Q&A Benchmark.

    PhD-level physics, chemistry, and biology questions written by domain
    experts, designed so that skilled non-experts cannot solve them even
    with web access. 4-option multiple choice (A-D).

    Default subset is gpqa_diamond (198 questions) - the hardest split,
    where both experts agreed and non-experts failed.

    NOTE: The HuggingFace dataset (Idavidrein/gpqa) is gated. You must:
    1. Accept the terms at https://huggingface.co/datasets/Idavidrein/gpqa
    2. Authenticate: `huggingface-cli login` or set HF_TOKEN in the
       environment.

    Answer choices are shuffled deterministically per question (seeded on
    question index) so the correct answer is not always in position A.
    """

    VALID_SUBSETS = ["gpqa_diamond", "gpqa_main", "gpqa_extended"]

    def __init__(self, subset: str = "gpqa_diamond", shuffle_seed: int = 42):
        """
        Initialize GPQA benchmark.

        Args:
            subset: Dataset config - gpqa_diamond (default), gpqa_main,
                    or gpqa_extended.
            shuffle_seed: Base seed for deterministic per-question option
                          shuffling.
        """
        if subset not in self.VALID_SUBSETS:
            raise ValueError(
                f"Invalid subset '{subset}'. Must be one of {self.VALID_SUBSETS}"
            )
        self.subset = subset
        self.shuffle_seed = shuffle_seed
        self._dataset = None

    @property
    def name(self) -> str:
        if self.subset == "gpqa_diamond":
            return "GPQA-Diamond"
        readable = self.subset.replace("gpqa_", "").title()
        return f"GPQA-{readable}"

    def _load_dataset(self):
        """Lazy load the dataset from HuggingFace (gated - requires HF_TOKEN)."""
        if self._dataset is None:
            try:
                from datasets import load_dataset

                self._dataset = load_dataset(
                    "Idavidrein/gpqa", self.subset, split="train"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load GPQA dataset: {e}\n"
                    "GPQA is gated on HuggingFace. Accept the terms at "
                    "https://huggingface.co/datasets/Idavidrein/gpqa and "
                    "authenticate (huggingface-cli login or HF_TOKEN env var)."
                )
        return self._dataset

    def load_questions(self, n: Optional[int] = None) -> List[Question]:
        """
        Load questions from GPQA.

        Args:
            n: Number of questions to load. If None, load all.

        Returns:
            List of Question objects formatted as 4-option multiple choice
        """
        dataset = self._load_dataset()

        if n is not None:
            dataset = dataset.select(range(min(n, len(dataset))))

        questions = []
        for i, item in enumerate(dataset):
            formatted_text, correct_letter = self._format_multiple_choice(item, i)

            questions.append(
                Question(
                    id=f"gpqa_{i}",
                    text=formatted_text,
                    ground_truth=correct_letter,
                    metadata={
                        "subset": self.subset,
                        "domain": item.get("High-level domain", "unknown"),
                    },
                )
            )

        return questions

    def _format_multiple_choice(self, item: dict, index: int) -> tuple[str, str]:
        """
        Format a GPQA item as a 4-option multiple choice question.

        Options are shuffled with a per-question deterministic seed so the
        correct answer position is uniform across the benchmark but stable
        across runs.

        Args:
            item: Dataset item with Question, Correct Answer, and three
                  Incorrect Answer fields
            index: Question index (used for the shuffle seed)

        Returns:
            Tuple of (formatted question text, correct answer letter)
        """
        question = item["Question"].strip()
        correct = item["Correct Answer"].strip()
        options = [
            correct,
            item["Incorrect Answer 1"].strip(),
            item["Incorrect Answer 2"].strip(),
            item["Incorrect Answer 3"].strip(),
        ]

        rng = random.Random(self.shuffle_seed + index)
        rng.shuffle(options)

        letters = "ABCD"
        correct_letter = letters[options.index(correct)]

        options_text = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]

        formatted_question = f"""{question}

{chr(10).join(options_text)}

This is a challenging question requiring careful reasoning. Think through it step by step.
Choose the best answer from the options above. Your answer should be a single letter (A through D).
End with: FINAL ANSWER: [letter]"""

        return formatted_question, correct_letter

    def _extract_letter_from_response(self, response: str) -> Optional[str]:
        """
        Extract a letter answer (A-D) from a model response.
        """
        # Try FINAL ANSWER pattern first (most reliable)
        match = re.search(r"FINAL ANSWER:\s*([A-Da-d])\b", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Try "answer is [letter]" pattern
        match = re.search(
            r"(?:the answer is|answer is|answer:)\s*([A-Da-d])\b",
            response,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()

        # Try "option [letter]" or "choice [letter]" pattern
        match = re.search(
            r"(?:correct option is|option|choice)\s*([A-Da-d])\b",
            response,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()

        # Try a standalone letter at end of response
        match = re.search(r"\b([A-Da-d])\s*[.)]?\s*$", response.strip())
        if match:
            return match.group(1).upper()

        # Last resort: the last standalone letter A-D
        matches = re.findall(r"(?<![a-zA-Z])([A-Da-d])(?![a-zA-Z])", response)
        if matches:
            return matches[-1].upper()

        return None

    def evaluate(self, question: Question, response: str) -> EvalResult:
        """
        Evaluate a response against the ground truth.

        Args:
            question: The question that was answered
            response: The model's response

        Returns:
            EvalResult with correctness assessment
        """
        predicted = self._extract_letter_from_response(response)

        if predicted is None:
            return EvalResult(
                question_id=question.id,
                is_correct=False,
                predicted="[no letter found]",
                expected=question.ground_truth,
            )

        is_correct = predicted.upper() == question.ground_truth.upper()

        return EvalResult(
            question_id=question.id,
            is_correct=is_correct,
            predicted=predicted,
            expected=question.ground_truth,
        )
