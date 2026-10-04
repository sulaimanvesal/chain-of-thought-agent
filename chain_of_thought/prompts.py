"""Prompt builders for the two CoT variants.

* Few-shot CoT (Wei et al. 2022): the exemplars precede the target question,
  formatted "Q: ...\\nA: <chain>".
* Zero-shot CoT (Kojima et al. 2022, arXiv:2205.11916): no exemplars at all.
  The reasoning prompt appends "Let's think step by step." to the question,
  and the answer is then elicited with a second-stage answer-trigger prompt
  ending in "Therefore, the answer is".
"""

from __future__ import annotations

from typing import List

from .exemplars import CoTExemplar


ZERO_SHOT_TRIGGER = "Let's think step by step."
ANSWER_TRIGGER = "Therefore, the answer is"


def format_exemplar(exemplar: CoTExemplar) -> str:
    """Render one exemplar as a Q/A block."""
    return f"Q: {exemplar.question}\nA: {exemplar.rationale}"


def build_few_shot_prompt(
    question: str,
    exemplars: List[CoTExemplar],
    question_prefix: str = "Q: ",
    answer_prefix: str = "A: ",
) -> str:
    """Build a few-shot CoT prompt: exemplars followed by the target question.

    The trailing answer prefix ("A: ") cues the model to continue with its
    own chain of thought, mirroring the exemplars.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    blocks = [format_exemplar(e) for e in exemplars]
    blocks.append(f"{question_prefix}{question}\n{answer_prefix}")
    return "\n\n".join(blocks)


def build_zero_shot_prompt(question: str) -> str:
    """Build the stage-1 zero-shot CoT prompt (question + trigger)."""
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    return f"{question.strip()} {ZERO_SHOT_TRIGGER}"


def build_zero_shot_answer_prompt(question: str, rationale: str) -> str:
    """Build the stage-2 prompt that elicits the final answer from a rationale.

    Follows Kojima et al.: the model-generated rationale is fed back in and
    the answer-trigger phrase prompts the final answer in the next tokens.
    """
    if not question or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not rationale or not rationale.strip():
        raise ValueError("rationale must be a non-empty string")
    return (
        f"{question.strip()}\n\n"
        f"{rationale.strip()}\n\n"
        f"{ANSWER_TRIGGER}"
    )
