"""HumanEval benchmark - verifiable-executable coding query type.

Second query type in the council study. Where HLE is judged multiple-choice
(the chairman can only *reason* about answers), HumanEval is verifiable: each
problem ships hidden unit tests, so the council's judge/chairman can actually
*run* candidate solutions. That verification asymmetry is the whole point of
the experiment (see EXPERIMENT-6-CODING.md).

Loads the ``openai_humaneval`` dataset (HuggingFace ``datasets``, split
"test", 164 problems). Each problem exposes:

- ``task_id``          e.g. "HumanEval/0"
- ``prompt``           function signature + docstring the model completes
- ``test``             the unit-test code (defines ``check(candidate)``)
- ``entry_point``      the function name under test
- ``canonical_solution`` the reference completion (body only)

A deterministic seeded shuffle (seed 42, mirroring HLE) selects the
n-problem subset so runs are reproducible and resumable.
"""

import random
from dataclasses import dataclass
from typing import List, Optional

from backend.execution import check_solution, extract_code


@dataclass
class CodingProblem:
    """A single verifiable coding problem from HumanEval."""

    task_id: str
    prompt: str
    test: str
    entry_point: str
    canonical_solution: str
    metadata: Optional[dict] = None


class HumanEvalBenchmark:
    """OpenAI HumanEval loader, mirroring the HLEBenchmark structure.

    Unlike ``Benchmark`` subclasses whose ``evaluate`` compares a string, the
    coding query type is graded by executing unit tests (see
    ``backend.execution.check_solution``), so ``evaluate`` here runs the tests
    rather than string-matching.
    """

    DATASET = "openai_humaneval"

    def __init__(self, sample_seed: int = 42):
        self.sample_seed = sample_seed
        self._dataset = None

    @property
    def name(self) -> str:
        return "HumanEval"

    def _load_dataset(self):
        if self._dataset is None:
            from datasets import load_dataset

            self._dataset = load_dataset(self.DATASET, split="test")
        return self._dataset

    def load_problems(self, n: Optional[int] = None) -> List[CodingProblem]:
        """Load an n-problem seeded-shuffle subset (all 164 if n is None)."""
        dataset = self._load_dataset()

        indices = list(range(len(dataset)))
        random.Random(self.sample_seed).shuffle(indices)

        problems: List[CodingProblem] = []
        for idx in indices:
            if n is not None and len(problems) >= n:
                break
            item = dataset[idx]
            problems.append(
                CodingProblem(
                    task_id=item["task_id"],
                    prompt=item["prompt"],
                    test=item["test"],
                    entry_point=item["entry_point"],
                    canonical_solution=item["canonical_solution"],
                    metadata={"task_id": item["task_id"]},
                )
            )
        return problems

    def build_generation_prompt(self, problem: CodingProblem) -> str:
        """Prompt asking the model to complete the function.

        Instructs a single fenced code block containing the full function so
        ``extract_code`` can recover a clean completion.
        """
        return f"""Complete the following Python function.

{problem.prompt}

Return ONLY the complete function implementation (including the signature) \
inside a single ```python code block. Do not include any explanation, usage \
examples, or test code."""

    def extract_code(self, response: Optional[str], entry_point: str) -> str:
        """Delegate to the shared extractor (strips fences / re-emitted sig)."""
        return extract_code(response, entry_point)

    def evaluate(self, problem: CodingProblem, completion: str, timeout: float = 10.0) -> dict:
        """Run the problem's unit tests against a completion.

        Returns ``{"passed", "error", "timed_out"}`` from the executor.
        """
        return check_solution(problem, completion, timeout=timeout)
