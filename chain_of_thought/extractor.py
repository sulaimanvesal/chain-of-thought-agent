"""Answer extraction from a generated chain of thought.

A model thinking out loud does not stop cleanly at the answer, so CoT
pipelines extract it from the trace. This module tries, in order:

1. ``\\boxed{...}`` (common in later CoT-based math work)
2. ``Therefore, the answer is <answer>`` (Kojima et al.'s answer trigger)
3. ``The answer is <answer>`` (the paper's own exemplar convention)
4. ``answer: <answer>`` / ``Answer: <answer>`` (loose convention)
5. last standalone number in the trace (numeric fallback)

The answer is normalised: commas stripped, floats that are integral become
ints, everything else returned as a string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExtractionResult:
    answer: Optional[str]
    strategy: str  # which rule fired; "none" when nothing matched
    matched_text: Optional[str]  # the raw span that produced the answer


_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_THEREFORE_RE = re.compile(
    r"[Tt]herefore[,\s]+the answer is[:\s]*([^\.\n]+?)[\.\s]*$", re.IGNORECASE
)
_ANSWER_IS_RE = re.compile(r"[Tt]he answer is[:\s]*([^\.\n]+?)[\.\s]*$")
_ANSWER_COLON_RE = re.compile(r"[Aa]nswer\s*[:=]\s*([^\n]+)")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalise(raw: str) -> str:
    """Clean a raw answer span into a canonical answer string."""
    cleaned = raw.strip().rstrip(".").strip()
    cleaned = cleaned.replace(",", "")
    # numbers: keep ints as ints, floats that are integral as ints
    num = re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned)
    if num:
        value = float(cleaned)
        return str(int(value)) if value.is_integer() else str(value)
    return cleaned


def extract_answer(trace: str) -> ExtractionResult:
    """Extract the final answer from a chain-of-thought trace."""
    if not trace or not trace.strip():
        return ExtractionResult(answer=None, strategy="none", matched_text=None)
    text = trace.strip()

    m = _BOXED_RE.search(text)
    if m:
        raw = m.group(1)
        return ExtractionResult(_normalise(raw), "boxed", m.group(0))

    m = _THEREFORE_RE.search(text)
    if m:
        raw = m.group(1)
        return ExtractionResult(_normalise(raw), "therefore", m.group(0))

    m = _ANSWER_IS_RE.search(text)
    if m:
        raw = m.group(1)
        return ExtractionResult(_normalise(raw), "answer_is", m.group(0))

    m = _ANSWER_COLON_RE.search(text)
    if m:
        raw = m.group(1)
        return ExtractionResult(_normalise(raw), "answer_colon", m.group(0))

    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return ExtractionResult(_normalise(numbers[-1]), "numeric_fallback", numbers[-1])

    return ExtractionResult(answer=None, strategy="none", matched_text=None)
