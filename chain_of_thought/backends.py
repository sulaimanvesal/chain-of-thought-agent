"""Pluggable LLM backends.

The solver only ever calls ``LLMBackend.generate(prompt)``. Two backends ship:

* ``MockLLM`` -- deterministic, offline. Returns canned chain-of-thought
  traces keyed to the demo word problems. No network, no API key.
* ``OpenAILLM`` -- thin wrapper over any OpenAI-compatible ``/v1/chat/completions``
  endpoint (OpenAI, Ollama, vLLM, ...), activated by environment variables.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Dict, Optional, Protocol


class LLMBackend(Protocol):
    """Any text-in / text-out model. Only ``generate`` is required."""

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Return the model's continuation for ``prompt``."""
        ...


# ----------------------------------------------------------------------------
# Canned traces for the offline demo (arithmetic word problems).
# Each key is a substring matched against the *question* so the mock works
# with either few-shot or zero-shot prompts. Keys and answers must stay in
# sync with ``examples/run_demo.py``'s DEMO_PROBLEMS.
# ----------------------------------------------------------------------------

MOCK_TRACES: Dict[str, str] = {
    "3 baskets": (
        "There were 3 baskets of apples originally. Each basket has 4 apples, "
        "so that's 3 * 4 = 12 apples. She ate 5, so 12 - 5 = 7. "
        "The answer is 7."
    ),
    "24 stickers": (
        "There were 24 stickers originally. He gave away 6 on Monday, so "
        "24 - 6 = 18 left. On Tuesday he gave away 3 more, so 18 - 3 = 15. "
        "The answer is 15."
    ),
    "7 marbles": (
        "He started with 7 marbles. He won 9 more, so 7 + 9 = 16. "
        "He gave his sister 4, so 16 - 4 = 12. The answer is 12."
    ),
    "36 cookies": (
        "There are 36 cookies in total. Each bag holds 9 cookies, so the "
        "number of bags is 36 / 9 = 4. She gave away 1 bag, so 4 - 1 = 3. "
        "The answer is 3."
    ),
    "50 balloons": (
        "There were 50 balloons originally. 13 popped, so 50 - 13 = 37 left. "
        "He gave 12 away, so 37 - 12 = 25. The answer is 25."
    ),
}


class MockLLM:
    """Deterministic offline stand-in for a real LLM.

    Matches the question against canned word problems and returns a
    step-by-step CoT trace. Unknown questions get a generic trace so the
    solver pipeline is still exercisable.
    """

    def __init__(self, traces: Optional[Dict[str, str]] = None) -> None:
        self.traces = traces if traces is not None else dict(MOCK_TRACES)
        self.calls: list[str] = []  # prompts seen (handy for tests/debugging)

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        self.calls.append(prompt)
        for key, trace in self.traces.items():
            if key in prompt:
                return trace
        return (
            "I will think step by step. The problem asks for a number. "
            "I cannot compute it exactly. The answer is unknown."
        )


class OpenAILLM:
    """Real backend over any OpenAI-compatible chat-completions endpoint.

    Activation (all optional, sensible defaults):
        COT_BASE_URL  -- e.g. "https://api.openai.com/v1" or "http://localhost:11434/v1"
        COT_API_KEY   -- bearer token (empty string for local servers)
        COT_MODEL     -- model name, default "gpt-4o-mini"
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model or os.environ.get("COT_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("COT_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("COT_API_KEY", "")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]["content"]
