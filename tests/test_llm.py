from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.core.exceptions import LLMError
from app.services.llm import LLMClient


class Answer(BaseModel):
    intent: str
    score: int


class FakeCompletions:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[dict] = []  # type: ignore[type-arg]

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        content = self.outputs.pop(0)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def _client(outputs: list[str]) -> tuple[LLMClient, FakeCompletions]:
    llm = LLMClient(api_key="test", model="test-model")
    fake = FakeCompletions(outputs)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))  # type: ignore[assignment]
    return llm, fake


def test_structured_output_parses_json() -> None:
    llm, fake = _client(['{"intent": "question", "score": 3}'])

    result = llm.generate_structured("sys", [{"role": "user", "content": "hi"}], Answer)

    assert result == Answer(intent="question", score=3)
    call = fake.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["temperature"] == 0.0
    # schema is included in the system prompt
    assert '"score"' in call["messages"][0]["content"]


def test_structured_output_retries_with_feedback() -> None:
    llm, fake = _client(['{"intent": "question"}', '{"intent": "question", "score": 1}'])

    result = llm.generate_structured("sys", [{"role": "user", "content": "hi"}], Answer)

    assert result.score == 1
    retry_messages = fake.calls[1]["messages"]
    assert retry_messages[-1]["role"] == "user"
    assert "invalid" in retry_messages[-1]["content"]


def test_structured_output_gives_up() -> None:
    llm, _ = _client(["not json", "still not json"])

    with pytest.raises(LLMError):
        llm.generate_structured("sys", [{"role": "user", "content": "hi"}], Answer)


def test_generate_text_prepends_system_prompt() -> None:
    llm, fake = _client(["Hello there"])

    assert llm.generate_text("be nice", [{"role": "user", "content": "hi"}]) == "Hello there"
    assert fake.calls[0]["messages"][0] == {"role": "system", "content": "be nice"}
