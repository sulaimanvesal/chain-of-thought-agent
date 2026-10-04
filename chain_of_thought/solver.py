"""The CoT solver: prompt -> generate -> extract.

``CoTSolver`` is the whole paper in one class:

* ``mode="few_shot"``  (Wei et al. 2022): exemplars + question, one
  generation, extract the answer from the trace.
* ``mode="zero_shot"``  (Kojima et al. 2022): question + "Let's think step
  by step." for stage 1 (the rationale), then the rationale plus the
  answer-trigger prompt for stage 2 (the answer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .backends import LLMBackend
from .exemplars import ExemplarBank, PAPER_EXEMPLARS
from .extractor import ExtractionResult, extract_answer
from .prompts import (
    build_few_shot_prompt,
    build_zero_shot_answer_prompt,
    build_zero_shot_prompt,
)


@dataclass(frozen=True)
class SolveResult:
    question: str
    mode: str
    prompt: str  # the stage-1 prompt (few-shot) / reasoning prompt (zero-shot)
    trace: str  # the model's generated chain of thought
    extraction: ExtractionResult
    second_prompt: Optional[str] = None  # only for zero-shot stage 2
    second_output: Optional[str] = None  # only for zero-shot stage 2

    @property
    def answer(self) -> Optional[str]:
        return self.extraction.answer


class CoTSolver:
    """Chain-of-thought reasoning over any ``LLMBackend``."""

    FEW_SHOT = "few_shot"
    ZERO_SHOT = "zero_shot"

    def __init__(
        self,
        backend: LLMBackend,
        mode: str = FEW_SHOT,
        exemplars: Optional[ExemplarBank] = None,
        n_exemplars: int = 8,
        exemplar_strategy: str = "first",
        max_tokens: int = 512,
    ) -> None:
        if mode not in (self.FEW_SHOT, self.ZERO_SHOT):
            raise ValueError(f"mode must be {self.FEW_SHOT!r} or {self.ZERO_SHOT!r}")
        self.backend = backend
        self.mode = mode
        self.bank = exemplars if exemplars is not None else ExemplarBank(list(PAPER_EXEMPLARS))
        self.n_exemplars = n_exemplars
        self.exemplar_strategy = exemplar_strategy
        self.max_tokens = max_tokens

    def solve(self, question: str) -> SolveResult:
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")
        if self.mode == self.FEW_SHOT:
            return self._solve_few_shot(question)
        return self._solve_zero_shot(question)

    def solve_batch(self, questions: List[str]) -> List[SolveResult]:
        return [self.solve(q) for q in questions]

    # -- internals ---------------------------------------------------------

    def _solve_few_shot(self, question: str) -> SolveResult:
        exemplars = self.bank.select(self.n_exemplars, self.exemplar_strategy)
        prompt = build_few_shot_prompt(question, exemplars)
        trace = self.backend.generate(prompt, self.max_tokens)
        return SolveResult(
            question=question,
            mode=self.FEW_SHOT,
            prompt=prompt,
            trace=trace,
            extraction=extract_answer(trace),
        )

    def _solve_zero_shot(self, question: str) -> SolveResult:
        prompt = build_zero_shot_prompt(question)
        rationale = self.backend.generate(prompt, self.max_tokens)
        second_prompt = build_zero_shot_answer_prompt(question, rationale)
        answer_text = self.backend.generate(second_prompt, self.max_tokens)
        # Kojima et al. stage 2 yields the answer directly; still run the
        # extractor so both modes expose identical result fields.
        extraction = extract_answer(answer_text)
        return SolveResult(
            question=question,
            mode=self.ZERO_SHOT,
            prompt=prompt,
            trace=rationale,
            extraction=extraction,
            second_prompt=second_prompt,
            second_output=answer_text,
        )
