import json
import logging
from typing import TypeVar

import groq
from groq.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from groq.types.chat.completion_create_params import ResponseFormat
from pydantic import BaseModel, ValidationError

from app.core.exceptions import LLMError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_JSON_FORMAT: ResponseFormat = {"type": "json_object"}
_TEXT_FORMAT: ResponseFormat = {"type": "text"}

# What callers pass in as conversation history (system prompt is passed separately).
LLMMessage = ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam

_JSON_INSTRUCTIONS = """

Respond ONLY with a single JSON object (no markdown, no extra text) that matches this JSON schema:
{schema}"""


class LLMClient:
    """Small wrapper around the Groq SDK so the rest of the app is provider-agnostic."""

    def __init__(
        self,
        api_key: str | None,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        max_retries: int = 1,
    ) -> None:
        # api_key=None lets the SDK read GROQ_API_KEY from the environment
        self.client = groq.Groq(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

    def generate_text(self, system: str, messages: list[LLMMessage]) -> str:
        text = self._complete(system, messages, temperature=self.temperature, json_mode=False)
        if not text:
            raise LLMError("The language model returned an empty response.")
        return text

    def generate_structured(self, system: str, messages: list[LLMMessage], output_model: type[T]) -> T:
        """Ask the model for JSON matching `output_model`, validated with Pydantic.

        Uses Groq's JSON mode (works on all their main models) and puts the schema in the
        system prompt. If the output doesn't validate we retry, telling the model what was wrong.
        """
        schema = json.dumps(output_model.model_json_schema())
        system_with_schema = system + _JSON_INSTRUCTIONS.format(schema=schema)
        conversation: list[LLMMessage] = list(messages)

        for attempt in range(self.max_retries + 1):
            raw = self._complete(system_with_schema, conversation, temperature=0.0, json_mode=True)
            try:
                return output_model.model_validate_json(raw)
            except ValidationError as exc:
                logger.warning("Structured output invalid (attempt %d): %s", attempt + 1, exc)
                conversation = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"That JSON was invalid: {exc.errors(include_url=False)}. "
                        "Return the corrected JSON object only.",
                    },
                ]

        raise LLMError("The language model did not return valid structured output.")

    def _complete(
        self,
        system: str,
        messages: list[LLMMessage],
        *,
        temperature: float,
        json_mode: bool,
    ) -> str:
        payload: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}, *messages]
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=temperature,
                max_completion_tokens=self.max_tokens,
                response_format=_JSON_FORMAT if json_mode else _TEXT_FORMAT,
            )
        except groq.BadRequestError as exc:
            # Groq returns 400 json_validate_failed when the model produces broken JSON
            if json_mode and "json_validate_failed" in str(exc):
                logger.warning("Groq rejected malformed JSON from the model")
                return ""
            raise self._wrap(exc) from exc
        except groq.APIError as exc:
            raise self._wrap(exc) from exc

        choice = completion.choices[0]
        if choice.finish_reason == "length":
            logger.warning("LLM response hit max tokens and may be truncated")
        return (choice.message.content or "").strip()

    @staticmethod
    def _wrap(exc: groq.APIError) -> LLMError:
        if isinstance(exc, groq.AuthenticationError):
            return LLMError("LLM authentication failed - check GROQ_API_KEY.")
        if isinstance(exc, groq.NotFoundError):
            return LLMError("LLM model not found - check LLM_MODEL against the models available on Groq.")
        if isinstance(exc, groq.RateLimitError):
            return LLMError("LLM rate limit reached, please retry shortly.")
        if isinstance(exc, groq.APIConnectionError):
            return LLMError("Could not reach the LLM provider.")
        logger.exception("LLM call failed")
        return LLMError("The language model request failed.")
