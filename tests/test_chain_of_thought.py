"""Pytest suite for the chain_of_thought package."""

import pytest

from chain_of_thought import (
    CoTSolver,
    CoTExemplar,
    ExemplarBank,
    MockLLM,
    PAPER_EXEMPLARS,
    SolveResult,
    build_few_shot_prompt,
    build_zero_shot_answer_prompt,
    build_zero_shot_prompt,
    extract_answer,
)
from chain_of_thought.backends import MOCK_TRACES, OpenAILLM
from chain_of_thought.prompts import ANSWER_TRIGGER, ZERO_SHOT_TRIGGER


# ---------------------------------------------------------------- exemplar bank

def test_paper_exemplars_count_and_shape():
    assert len(PAPER_EXEMPLARS) == 8
    for e in PAPER_EXEMPLARS:
        assert isinstance(e, CoTExemplar)
        assert e.question and e.rationale
        assert "answer" in e.rationale.lower()


def test_paper_exemplars_known_questions():
    questions = [e.question for e in PAPER_EXEMPLARS]
    assert any("tennis balls" not in q for q in questions)  # sanity: real list
    assert any("lollipops" in q for q in questions)
    assert any("golf balls" in q for q in questions)


def test_bank_select_first_n():
    bank = ExemplarBank()
    assert bank.select(0) == []
    assert len(bank.select(3)) == 3
    assert bank.select(3)[0].question == PAPER_EXEMPLARS[0].question


def test_bank_select_too_many_raises():
    with pytest.raises(ValueError):
        ExemplarBank().select(9)


def test_bank_select_negative_raises():
    with pytest.raises(ValueError):
        ExemplarBank().select(-1)


def test_bank_select_reverse_strategy():
    bank = ExemplarBank()
    selected = bank.select(4, strategy="reverse")
    assert [e.question for e in selected] == [e.question for e in reversed(PAPER_EXEMPLARS[:4])]


def test_bank_select_unknown_strategy_raises():
    with pytest.raises(ValueError):
        ExemplarBank().select(2, strategy="random")


def test_bank_custom_exemplars():
    custom = [CoTExemplar(question="Q1", rationale="R1 The answer is 1.")]
    assert ExemplarBank(custom).select(1)[0].question == "Q1"


# ---------------------------------------------------------------- prompt builders

def test_few_shot_prompt_contains_exemplars_and_question():
    prompt = build_few_shot_prompt("What is 2+2?", PAPER_EXEMPLARS[:2])
    assert "Q:" in prompt and "A:" in prompt
    assert PAPER_EXEMPLARS[0].question in prompt
    assert PAPER_EXEMPLARS[0].rationale in prompt
    assert prompt.rstrip().endswith("A:")
    assert "What is 2+2?" in prompt


def test_few_shot_prompt_exemplar_order_preserved():
    prompt = build_few_shot_prompt("q?", PAPER_EXEMPLARS[:3])
    positions = [prompt.index(e.question) for e in PAPER_EXEMPLARS[:3]]
    assert positions == sorted(positions)


def test_few_shot_prompt_empty_question_raises():
    with pytest.raises(ValueError):
        build_few_shot_prompt("   ", PAPER_EXEMPLARS[:1])


def test_few_shot_prompt_zero_exemplars():
    prompt = build_few_shot_prompt("q?", [])
    assert "q?" in prompt and prompt.rstrip().endswith("A:")


def test_zero_shot_prompt_has_trigger():
    prompt = build_zero_shot_prompt("How many?")
    assert ZERO_SHOT_TRIGGER in prompt
    assert prompt.startswith("How many?")


def test_zero_shot_prompt_strips_whitespace():
    assert build_zero_shot_prompt("  How many? \n") == f"How many? {ZERO_SHOT_TRIGGER}"


def test_zero_shot_prompt_empty_raises():
    with pytest.raises(ValueError):
        build_zero_shot_prompt("")


def test_zero_shot_answer_prompt_has_trigger():
    p = build_zero_shot_answer_prompt("How many?", "Some reasoning steps.")
    assert p.rstrip().endswith(ANSWER_TRIGGER)
    assert "Some reasoning steps." in p
    assert "How many?" in p


def test_zero_shot_answer_prompt_empty_rationale_raises():
    with pytest.raises(ValueError):
        build_zero_shot_answer_prompt("q?", "  ")


# ---------------------------------------------------------------- answer extraction

def test_extract_the_answer_is():
    r = extract_answer("First do this. Then do that. The answer is 42.")
    assert r.answer == "42" and r.strategy == "answer_is"


def test_extract_therefore_trigger():
    r = extract_answer("Long chain of steps. Therefore, the answer is 7.")
    assert r.answer == "7" and r.strategy == "therefore"


def test_extract_boxed():
    r = extract_answer("Reasoning steps... \\boxed{123}")
    assert r.answer == "123" and r.strategy == "boxed"


def test_extract_answer_colon():
    r = extract_answer("Steps here. Answer: 9 apples")
    assert r.answer == "9 apples" and r.strategy == "answer_colon"


def test_extract_strips_commas():
    r = extract_answer("The answer is 1,234.")
    assert r.answer == "1234"


def test_extract_integral_float_becomes_int():
    r = extract_answer("Therefore, the answer is 8.0.")
    assert r.answer == "8"


def test_extract_nonintegral_float_kept():
    r = extract_answer("The answer is 2.5.")
    assert r.answer == "2.5"


def test_extract_numeric_fallback_last_number():
    r = extract_answer("He had 58 and lost 23, ending with 33.")
    assert r.answer == "33" and r.strategy == "numeric_fallback"


def test_extract_empty_trace():
    r = extract_answer("   ")
    assert r.answer is None and r.strategy == "none" and r.matched_text is None


def test_extract_no_answer_found():
    r = extract_answer("Just some prose with no digits or answer phrase at all.")
    assert r.answer is None and r.strategy == "none"


def test_extract_matched_text_captured():
    r = extract_answer("Reasoning done. The answer is 6.")
    assert r.matched_text is not None and "6" in r.matched_text


# ---------------------------------------------------------------- mock backend

def test_mock_llm_known_problem_returns_canned_trace():
    llm = MockLLM()
    trace = llm.generate("Q: A baker made 36 cookies. Each bag holds 9 cookies ... A:")
    assert trace == MOCK_TRACES["36 cookies"]
    assert "The answer is 3." in trace


def test_mock_llm_unknown_problem_returns_generic_trace():
    llm = MockLLM()
    trace = llm.generate("Q: What is the capital of France? A:")
    assert extract_answer(trace).answer == "unknown"


def test_mock_llm_custom_traces_override():
    llm = MockLLM({"moon": "The answer is 42."})
    assert llm.generate("the moon") == "The answer is 42."


def test_openai_llm_reads_env(monkeypatch):
    monkeypatch.setenv("COT_MODEL", "test-model")
    monkeypatch.setenv("COT_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("COT_API_KEY", "sk-test")
    llm = OpenAILLM()
    assert llm.model == "test-model"
    assert llm.base_url == "http://localhost:9999/v1"
    assert llm.api_key == "sk-test"


# ---------------------------------------------------------------- solver end-to-end

def _demo_question(key: str) -> str:
    # minimal question whose wording contains the mock's matching key
    return {
        "3 baskets": "A grocer has 3 baskets of apples, 4 apples each, ate 5. Left?",
        "24 stickers": "Tom had 24 stickers, gave 6 Monday and 3 Tuesday. Now?",
        "7 marbles": "Dan had 7 marbles, won 9, gave 4 away. Now?",
        "36 cookies": "Baker made 36 cookies, 9 per bag, gave 1 bag away. Left?",
        "50 balloons": "Party had 50 balloons, 13 popped, 12 given away. Left?",
    }[key]


@pytest.mark.parametrize("key,expected", [
    ("3 baskets", "7"),
    ("24 stickers", "15"),
    ("7 marbles", "12"),
    ("36 cookies", "3"),
    ("50 balloons", "25"),
])
def test_few_shot_solve_end_to_end(key, expected):
    solver = CoTSolver(MockLLM(), mode=CoTSolver.FEW_SHOT, n_exemplars=8)
    result = solver.solve(_demo_question(key))
    assert isinstance(result, SolveResult)
    assert result.mode == CoTSolver.FEW_SHOT
    assert result.answer == expected
    assert "A:" in result.prompt  # exemplars present


@pytest.mark.parametrize("key,expected", [
    ("3 baskets", "7"),
    ("50 balloons", "25"),
])
def test_zero_shot_solve_end_to_end(key, expected):
    solver = CoTSolver(MockLLM(), mode=CoTSolver.ZERO_SHOT)
    result = solver.solve(_demo_question(key))
    assert result.mode == CoTSolver.ZERO_SHOT
    assert ZERO_SHOT_TRIGGER in result.prompt
    assert ANSWER_TRIGGER in result.second_prompt
    assert result.second_output is not None
    assert result.answer == expected


def test_zero_shot_stage1_prompt_has_no_exemplars():
    solver = CoTSolver(MockLLM(), mode=CoTSolver.ZERO_SHOT)
    result = solver.solve(_demo_question("7 marbles"))
    assert "lollipops" not in result.prompt  # no paper exemplars leaked in


def test_solver_empty_question_raises():
    with pytest.raises(ValueError):
        CoTSolver(MockLLM()).solve(" ")


def test_solver_invalid_mode_raises():
    with pytest.raises(ValueError):
        CoTSolver(MockLLM(), mode="self_consistency")


def test_solver_batch():
    solver = CoTSolver(MockLLM(), mode=CoTSolver.FEW_SHOT, n_exemplars=2)
    results = solver.solve_batch([_demo_question("7 marbles"), _demo_question("3 baskets")])
    assert [r.answer for r in results] == ["12", "7"]


def test_solver_few_exemplars_still_works():
    solver = CoTSolver(MockLLM(), mode=CoTSolver.FEW_SHOT, n_exemplars=1)
    result = solver.solve(_demo_question("24 stickers"))
    assert result.answer == "15"


def test_mock_records_calls():
    backend = MockLLM()
    solver = CoTSolver(backend, mode=CoTSolver.FEW_SHOT, n_exemplars=2)
    solver.solve(_demo_question("7 marbles"))
    assert len(backend.calls) == 1
    solver2 = CoTSolver(backend, mode=CoTSolver.ZERO_SHOT)
    solver2.solve(_demo_question("7 marbles"))
    assert len(backend.calls) == 3  # two stages
