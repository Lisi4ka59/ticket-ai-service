import json
import logging
import re
from typing import Any

import httpx
from pydantic import ValidationError

from app.config import settings
from app.knowledge_base import KnowledgeRecord, build_context, score_records_for_text
from app.models import LLMStructuredResult, TicketRequest

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    """OpenAI-compatible client for local llama.cpp server.

    llama.cpp exposes `/v1/chat/completions`, so the service can call the
    local model with plain HTTP and validate the model JSON response itself.
    """

    def __init__(self) -> None:
        self.base_url = settings.llm_base_url.rstrip("/")
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_seconds
        self.api_key = settings.llm_api_key

    async def analyze(self, ticket: TicketRequest, records: list[KnowledgeRecord]) -> LLMStructuredResult:
        payload = self._build_payload(ticket, records)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        logger.info("Calling LLM provider=%s model=%s ticket_id=%s", settings.llm_provider, self.model, ticket.ticket_id)

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise LLMError("LLM HTTP call failed") from exc
        except json.JSONDecodeError as exc:
            raise LLMError("LLM server returned non-JSON HTTP response") from exc

        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not content or not str(content).strip():
            raise LLMError("LLM returned empty response")
        return parse_llm_response(str(content), ticket, records)

    def _build_payload(self, ticket: TicketRequest, records: list[KnowledgeRecord]) -> dict[str, Any]:
        schema_hint = {
            "summary": "string",
            "probable_cause": "string",
            "recommended_actions": ["string"],
            "risk_level": "low|medium|high|critical",
            "needs_human_review": "boolean",
        }
        review_policy = (
            "Set needs_human_review to true only when the case is ambiguous, "
            "the risk is high or critical, the knowledge context is missing or weak, "
            "or the recommended actions require human escalation. "
            "Set it to false for clear low or medium risk cases with enough context "
            "and routine automated checks."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a NOC engineer. /no_think\n"
                    "Return exactly one valid JSON object and nothing else. "
                    "Do not use markdown, comments, or text outside JSON. "
                    "Do not add fields outside the requested schema. "
                    "Do not invent facts outside the ticket and knowledge context. "
                    f"{review_policy} "
                    "All string values in the JSON response must be written in English."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "ticket": ticket.model_dump(mode="json"),
                        "knowledge_context": build_context(records),
                        "required_json_schema": schema_hint,
                        "needs_human_review_policy": review_policy,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return {
            "model": self.model,
            "messages": messages,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "response_format": {"type": "json_object"},
        }


def _normalize_local_model_output(content: str) -> str:
    """Remove common wrappers produced by local chat models.

    Qwen3 may emit reasoning in `<think>...</think>`, and some templates wrap
    JSON in markdown fences. After cleanup, only one JSON object is accepted.
    """
    text = content.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*```$", "", text).strip()
    return text


def parse_llm_response(content: str, ticket: TicketRequest, records: list[KnowledgeRecord]) -> LLMStructuredResult:
    normalized = _normalize_local_model_output(content)
    try:
        raw = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise LLMError("LLM returned invalid JSON") from exc
    if not isinstance(raw, dict):
        raise LLMError("LLM response must be a JSON object")
    try:
        parsed = LLMStructuredResult.model_validate(raw)
    except ValidationError as exc:
        raise LLMError("LLM response has invalid structure") from exc

    validate_llm_consistency(ticket, records, parsed)
    return parsed


def validate_llm_consistency(ticket: TicketRequest, records: list[KnowledgeRecord], parsed: LLMStructuredResult) -> None:
    if ticket.priority.value == "critical" and parsed.risk_level.value not in {"critical", "high"}:
        raise LLMError("LLM risk contradicts critical input priority")
    if not records:
        return

    response_text = " ".join([parsed.summary, parsed.probable_cause, " ".join(parsed.recommended_actions)])
    response_scores = score_records_for_text(response_text)
    if not response_scores:
        return

    selected_ids = {record.id for record in records}
    selected_score = max((score for score, record in response_scores if record.id in selected_ids), default=0.0)
    top_score, top_record = response_scores[0]
    if top_record.id not in selected_ids and top_score >= 2.0 and top_score >= selected_score + 1.0:
        raise LLMError("LLM response contradicts selected knowledge base context")
