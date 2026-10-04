#!/usr/bin/env python3
"""Offline end-to-end demo of Chain-of-Thought prompting.

Runs five arithmetic word problems through the MockLLM with both few-shot
CoT (Wei et al., 2022) and zero-shot CoT (Kojima et al., 2022), printing the
generated reasoning trace and the extracted answer for each.

No API key, no network.
"""

import sys

sys.path.insert(0, ".")

from chain_of_thought import CoTSolver, MockLLM

DEMO_PROBLEMS = [
    (
        "A grocer has 3 baskets of apples. Each basket holds 4 apples. "
        "She eats 5 apples. How many apples does she have left?",
        "7",
    ),
    (
        "Tom had 24 stickers. He gave 6 away on Monday and 3 more on Tuesday. "
        "How many stickers does he have now?",
        "15",
    ),
    (
        "Dan started with 7 marbles. He won 9 more marbles in a game, then "
        "gave 4 to his sister. How many marbles does Dan have now?",
        "12",
    ),
    (
        "A baker made 36 cookies. Each bag holds 9 cookies, and she gave "
        "away 1 bag. How many bags of cookies does she still have?",
        "3",
    ),
    (
        "There were 50 balloons at a party. 13 popped and the clown gave "
        "12 away. How many balloons are left?",
        "25",
    ),
]


def run(mode: str, n_exemplars: int = 8) -> bool:
    backend = MockLLM()
    solver = CoTSolver(backend, mode=mode, n_exemplars=n_exemplars)
    all_ok = True
    for question, expected in DEMO_PROBLEMS:
        result = solver.solve(question)
        ok = result.answer == expected
        all_ok = all_ok and ok
        status = "OK " if ok else "FAIL"
        print(f"[{status}] ({mode}) {question}")
        print(f"  trace: {result.trace}")
        print(f"  extracted answer: {result.answer} (expected {expected})")
        print()
    return all_ok


def main() -> None:
    print("=== Chain-of-Thought offline demo (MockLLM) ===\n")
    print("--- Few-shot CoT (Wei et al., 8 exemplars) ---")
    few_ok = run(CoTSolver.FEW_SHOT)
    print("--- Zero-shot CoT (Kojima et al., 'Let's think step by step.') ---")
    zero_ok = run(CoTSolver.ZERO_SHOT)
    if few_ok and zero_ok:
        print("All demo problems solved with correct answers.")
    else:
        print("Some problems were answered incorrectly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
