"""Chain-of-Thought (CoT) prompting, after Wei et al., NeurIPS 2022.

Public API:
    backends   -- LLMBackend protocol, MockLLM (offline), OpenAILLM (optional)
    exemplars  -- the paper's 8 GSM8K few-shot CoT exemplars + ExemplarBank
    prompts    -- few-shot CoT prompt builder + zero-shot CoT template
    extractor  -- answer extraction from a generated rationale
    solver     -- CoTSolver: prompt -> generate -> extract
"""

from .backends import LLMBackend, MockLLM, OpenAILLM
from .exemplars import CoTExemplar, ExemplarBank, PAPER_EXEMPLARS
from .extractor import ExtractionResult, extract_answer
from .prompts import (
    build_few_shot_prompt,
    build_zero_shot_answer_prompt,
    build_zero_shot_prompt,
)
from .solver import CoTSolver, SolveResult

__all__ = [
    "LLMBackend",
    "MockLLM",
    "OpenAILLM",
    "CoTExemplar",
    "ExemplarBank",
    "PAPER_EXEMPLARS",
    "ExtractionResult",
    "extract_answer",
    "build_few_shot_prompt",
    "build_zero_shot_answer_prompt",
    "build_zero_shot_prompt",
    "CoTSolver",
    "SolveResult",
]

__version__ = "0.1.0"
