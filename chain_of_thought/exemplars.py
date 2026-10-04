"""Few-shot CoT exemplars.

``PAPER_EXEMPLARS`` is the eight arithmetic (GSM8K-style) exemplars from the
paper's appendix (Wei et al., NeurIPS 2022, Table A1). The question text is
reproduced verbatim; the rationales follow the paper's own few-shot chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class CoTExemplar:
    question: str
    rationale: str  # the step-by-step chain of thought (ends with "The answer is N.")


PAPER_EXEMPLARS: List[CoTExemplar] = [
    CoTExemplar(
        question="There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        rationale="There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    CoTExemplar(
        question="If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        rationale="There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    CoTExemplar(
        question="Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        rationale="Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. They ate 35, so they had 74 - 35 = 39. The answer is 39.",
    ),
    CoTExemplar(
        question="Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        rationale="Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    CoTExemplar(
        question="Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        rationale="Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
    CoTExemplar(
        question="There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        rationale="There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    CoTExemplar(
        question="Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        rationale="Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33. The answer is 33.",
    ),
    CoTExemplar(
        question="Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        rationale="Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
]


class ExemplarBank:
    """Selectable set of exemplars for few-shot CoT prompts.

    By default the first ``n`` exemplars are used, in the paper's original
    order (the paper uses all eight). ``strategy="reverse"`` lets callers
    check robustness to exemplar ordering.
    """

    def __init__(self, exemplars: List[CoTExemplar] | None = None) -> None:
        self.exemplars = list(exemplars) if exemplars is not None else list(PAPER_EXEMPLARS)

    def select(self, n: int, strategy: str = "first") -> List[CoTExemplar]:
        if n < 0:
            raise ValueError("n must be >= 0")
        if n > len(self.exemplars):
            raise ValueError(f"bank only holds {len(self.exemplars)} exemplars, asked for {n}")
        if strategy == "first":
            return self.exemplars[:n]
        if strategy == "reverse":
            return list(reversed(self.exemplars[:n]))
        raise ValueError(f"unknown strategy {strategy!r}; expected 'first' or 'reverse'")
