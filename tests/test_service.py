import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from app.config import Settings
from app.knowledge_base import find_relevant_records
from app.llm_client import LLMClient
from app.main import app
from app.llm_client import LLMError, parse_llm_response
from app.models import LLMStructuredResult, Priority, TicketRequest
from app.service import analyze_ticket

client = TestClient(app)


class FakeLLMClient:
    async def analyze(self, ticket, records):
        return LLMStructuredResult(
            summary="The network node shows increased packet loss and errors.",
            probable_cause="Possible link degradation or interface overload.",
            recommended_actions=["Check interfaces", "Check CRC errors", "Notify NOC"],
            risk_level=Priority.high,
            needs_human_review=False,
        )


class LowRiskLLMClient:
    async def analyze(self, ticket, records):
        return LLMStructuredResult(
            summary="A single latency alert was detected without service impact.",
            probable_cause="Likely transient congestion or measurement noise.",
            recommended_actions=["Check the latest latency metric", "Keep monitoring the node"],
            risk_level=Priority.low,
            needs_human_review=False,
        )


def test_llm_payload_requests_english_response():
    ticket = TicketRequest.model_validate(valid_payload())
    payload = LLMClient()._build_payload(ticket, [])
    system_prompt = payload["messages"][0]["content"]
    user_payload = __import__("json").loads(payload["messages"][1]["content"])
    assert "All string values in the JSON response must be written in English." in system_prompt
    assert user_payload["required_json_schema"]["needs_human_review"] == "boolean"
    assert "needs_human_review_policy" in user_payload


@pytest.mark.asyncio
async def test_llm_client_sends_authorization_header_when_api_key_is_set(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({
                                "summary": "Packet loss increased on the node.",
                                "probable_cause": "Possible link degradation.",
                                "recommended_actions": ["Check interface counters"],
                                "risk_level": "high",
                                "needs_human_review": True,
                            })
                        }
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    llm_client = LLMClient()
    llm_client.api_key = "test-secret"

    ticket = TicketRequest.model_validate(valid_payload())
    result = await llm_client.analyze(ticket, find_relevant_records(ticket))

    assert captured["headers"]["Authorization"] == "Bearer test-secret"
    assert result.summary == "Packet loss increased on the node."


class BrokenLLMClient:
    async def analyze(self, ticket, records):
        raise LLMError("invalid json")


class CountingBrokenLLMClient:
    def __init__(self):
        self.calls = 0

    async def analyze(self, ticket, records):
        self.calls += 1
        raise LLMError("LLM returned invalid JSON")


def valid_payload():
    return {
        "ticket_id": "T-1001",
        "source": "monitoring",
        "priority": "high",
        "service": "internet",
        "description": "Packet loss and latency increased on network node sw-01",
        "metrics": {"error_rate": 0.25, "affected_users": 120, "node": "sw-01"},
    }


@pytest.mark.asyncio
async def test_successful_processing_with_mock_llm():
    ticket = TicketRequest.model_validate(valid_payload())
    result = await analyze_ticket(ticket, FakeLLMClient())
    assert result.processing_status == "success"
    assert result.risk_level == "high"
    assert result.recommended_actions
    assert result.used_sources


def test_missing_required_field_description():
    payload = valid_payload()
    payload.pop("description")
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


def test_invalid_priority():
    payload = valid_payload()
    payload["priority"] = "urgent"
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


def test_invalid_source():
    payload = valid_payload()
    payload["source"] = "email"
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_llm_response_returns_safe_failed_result():
    ticket = TicketRequest.model_validate(valid_payload())
    result = await analyze_ticket(ticket, BrokenLLMClient())
    assert result.processing_status == "failed"
    assert result.needs_human_review is True
    assert result.model_info.fallback_used is True
    assert result.recommended_actions


@pytest.mark.asyncio
async def test_llm_failure_is_retried_before_fallback():
    ticket = TicketRequest.model_validate(valid_payload())
    llm_client = CountingBrokenLLMClient()
    result = await analyze_ticket(ticket, llm_client)
    assert llm_client.calls == 2
    assert result.model_info.attempts == 2
    assert result.processing_status == "failed"


@pytest.mark.asyncio
async def test_high_priority_risk_is_not_downgraded():
    ticket = TicketRequest.model_validate(valid_payload())
    result = await analyze_ticket(ticket, FakeLLMClient())
    assert result.risk_level in {"high", "critical"}
    assert result.needs_human_review is True


@pytest.mark.asyncio
async def test_low_risk_with_relevant_context_can_skip_human_review():
    payload = valid_payload()
    payload["ticket_id"] = "T-LOW-001"
    payload["priority"] = "low"
    payload["description"] = "Single latency alert on node sw-01, no packet loss, service remains available"
    payload["metrics"] = {"error_rate": 0.01, "affected_users": 1, "node": "sw-01"}
    ticket = TicketRequest.model_validate(payload)
    result = await analyze_ticket(ticket, LowRiskLLMClient())
    assert result.processing_status == "success"
    assert result.risk_level == "low"
    assert result.needs_human_review is False
    assert result.used_sources


@pytest.mark.asyncio
async def test_no_relevant_knowledge_record_does_not_crash():
    payload = valid_payload()
    payload["priority"] = "low"
    payload["source"] = "manual"
    payload["description"] = "Unclear one-off ticket without known symptoms"
    payload["service"] = "unknown"
    payload["metrics"] = None
    ticket = TicketRequest.model_validate(payload)
    result = await analyze_ticket(ticket, FakeLLMClient())
    assert result.processing_status == "partial_success"
    assert result.used_sources == []
    assert result.needs_human_review is True


def test_invalid_affected_users_type():
    payload = valid_payload()
    payload["metrics"]["affected_users"] = "many"
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


def test_numeric_string_affected_users_is_rejected():
    payload = valid_payload()
    payload["metrics"]["affected_users"] = "10"
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


def test_invalid_error_rate_range():
    payload = valid_payload()
    payload["metrics"]["error_rate"] = 2
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


def test_numeric_string_error_rate_is_rejected():
    payload = valid_payload()
    payload["metrics"]["error_rate"] = "0.25"
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


def test_blank_service_is_rejected():
    payload = valid_payload()
    payload["service"] = "   "
    response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422


def test_validation_failure_is_logged(caplog):
    payload = valid_payload()
    payload["metrics"]["affected_users"] = "10"
    with caplog.at_level(logging.WARNING):
        response = client.post("/api/v1/analyze", json=payload)
    assert response.status_code == 422
    assert "Input validation failed" in caplog.text


def test_llm_response_with_string_bool_is_rejected():
    ticket = TicketRequest.model_validate(valid_payload())
    records = find_relevant_records(ticket)
    content = json.dumps({
        "summary": "The network node shows increased packet loss.",
        "probable_cause": "Possible link degradation.",
        "recommended_actions": ["Check interfaces"],
        "risk_level": "high",
        "needs_human_review": "false",
    })
    with pytest.raises(LLMError, match="invalid structure"):
        parse_llm_response(content, ticket, records)


def test_llm_plain_text_response_is_rejected():
    ticket = TicketRequest.model_validate(valid_payload())
    records = find_relevant_records(ticket)
    with pytest.raises(LLMError, match="invalid JSON"):
        parse_llm_response("Here is the analysis: packet loss is increasing.", ticket, records)


def test_llm_response_with_think_block_and_code_fence_is_accepted():
    ticket = TicketRequest.model_validate(valid_payload())
    records = find_relevant_records(ticket)
    content = """<think>internal reasoning</think>
```json
{
  "summary": "Packet loss increased on the network node.",
  "probable_cause": "Possible link degradation.",
  "recommended_actions": ["Check interface counters", "Check adjacent links"],
  "risk_level": "high",
  "needs_human_review": true
}
```"""
    result = parse_llm_response(content, ticket, records)
    assert result.summary == "Packet loss increased on the network node."


def test_llm_response_conflicting_with_knowledge_is_rejected():
    ticket = TicketRequest.model_validate(valid_payload())
    records = find_relevant_records(ticket)
    content = json.dumps({
        "summary": "Mass PPPoE auth failures are observed.",
        "probable_cause": "RADIUS or BRAS authentication failure.",
        "recommended_actions": ["Check RADIUS logs", "Check AAA availability"],
        "risk_level": "high",
        "needs_human_review": True,
    })
    with pytest.raises(LLMError, match="contradicts selected knowledge base context"):
        parse_llm_response(content, ticket, records)


def test_settings_require_at_least_one_retry():
    with pytest.raises(ValidationError):
        Settings(llm_max_retries=0)
