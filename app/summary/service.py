"""Transcript → MedicalSummary, with the extractors running concurrently."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from .extractors import EXTRACTORS, Extractor, strict_json_schema
from .schemas import MedicalSummary, SummarizeResponse


class ChatClient(Protocol):
    async def chat_completion(self, payload: dict[str, Any], *, timeout_s: float) -> dict[str, Any]: ...


class SummaryError(Exception):
    """The model's answer could not be turned into a summary."""


class SummaryService:
    def __init__(
        self,
        client: ChatClient,
        *,
        model: str,
        timeout_s: float,
        extractors: Sequence[Extractor] = EXTRACTORS,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout_s = timeout_s
        self._extractors = tuple(extractors)

    async def summarize(self, transcript: str) -> SummarizeResponse:
        started = time.perf_counter()
        results = await asyncio.gather(
            *(self._extract(extractor, transcript) for extractor in self._extractors),
            return_exceptions=True,
        )

        sections = [r for r in results if isinstance(r, BaseModel)]
        if not sections:
            first_error = results[0]
            raise first_error if isinstance(first_error, Exception) else SummaryError("Summary generation failed")

        merged: dict[str, Any] = {}
        for section in sections:
            merged.update(section.model_dump())
        return SummarizeResponse(
            summary=MedicalSummary.model_validate(merged),
            model=self._model,
            latency_ms=round((time.perf_counter() - started) * 1000),
            partial=len(sections) < len(results),
        )

    async def _extract(self, extractor: Extractor, transcript: str) -> BaseModel:
        completion = await self._client.chat_completion(self._payload(extractor, transcript), timeout_s=self._timeout_s)
        try:
            content = completion["choices"][0]["message"]["content"] or ""
            return extractor.model.model_validate(json.loads(content))
        except (KeyError, IndexError, json.JSONDecodeError, ValidationError) as error:
            raise SummaryError(f"The model returned an invalid {extractor.name} section") from error

    def _payload(self, extractor: Extractor, transcript: str) -> dict[str, Any]:
        is_gpt_oss = self._model.startswith("openai/gpt-oss")
        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": 0,
            "max_completion_tokens": 2048,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": extractor.name,
                    "strict": is_gpt_oss,  # constrained decoding is available for gpt-oss on Groq
                    "schema": strict_json_schema(extractor.model),
                },
            },
            "messages": [
                {"role": "system", "content": extractor.system_prompt},
                {"role": "user", "content": f"Consultation transcript:\n<<<\n{transcript}\n>>>"},
            ],
        }
        if is_gpt_oss:
            payload["reasoning_effort"] = "low"  # extraction needs little reasoning; keeps latency down
        return payload
