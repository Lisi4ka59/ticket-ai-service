import logging
import time
from app.config import settings
from app.knowledge_base import find_relevant_records, to_public_source
from app.llm_client import LLMClient, LLMError
from app.models import AnalysisResult, ModelInfo, ProcessingStatus, Priority, TicketRequest

logger = logging.getLogger(__name__)

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def calculate_risk(ticket: TicketRequest, model_risk: Priority | None, has_sources: bool) -> Priority:
    risk = ticket.priority.value
    metrics = ticket.metrics
    if metrics:
        if metrics.error_rate is not None:
            if metrics.error_rate >= 0.5:
                risk = "critical"
            elif metrics.error_rate >= 0.2 and RISK_ORDER[risk] < RISK_ORDER["high"]:
                risk = "high"
            elif metrics.error_rate >= 0.05 and RISK_ORDER[risk] < RISK_ORDER["medium"]:
                risk = "medium"
        if metrics.affected_users is not None:
            if metrics.affected_users >= 1000:
                risk = "critical"
            elif metrics.affected_users >= 100 and RISK_ORDER[risk] < RISK_ORDER["high"]:
                risk = "high"
            elif metrics.affected_users >= 10 and RISK_ORDER[risk] < RISK_ORDER["medium"]:
                risk = "medium"
    if model_risk and RISK_ORDER[model_risk.value] > RISK_ORDER[risk]:
        risk = model_risk.value
    if not has_sources and RISK_ORDER[risk] < RISK_ORDER["medium"]:
        risk = "medium"
    return Priority(risk)


def fallback_result(ticket: TicketRequest, attempts: int, reason: str, used_sources) -> AnalysisResult:
    risk = calculate_risk(ticket, None, bool(used_sources))
    return AnalysisResult(
        ticket_id=ticket.ticket_id,
        summary=f"Could not obtain a valid language model analysis: {reason}.",
        probable_cause="The cause was not determined automatically. Manual review of the ticket and metrics is required.",
        recommended_actions=[
            "Escalate the ticket to the on-call engineer for manual analysis.",
            "Check source metrics and recent changes for the affected service.",
            "Compare the ticket with monitoring logs and network node events.",
        ],
        risk_level=risk,
        needs_human_review=True,
        used_sources=used_sources,
        model_info=ModelInfo(provider=settings.llm_provider, model=settings.llm_model, attempts=attempts, fallback_used=True),
        processing_status=ProcessingStatus.failed,
    )


async def analyze_ticket(ticket: TicketRequest, llm_client: LLMClient | None = None) -> AnalysisResult:
    started = time.perf_counter()
    logger.info("Request received ticket_id=%s source=%s priority=%s", ticket.ticket_id, ticket.source.value, ticket.priority.value)
    logger.info("Input validation passed ticket_id=%s", ticket.ticket_id)

    kb_started = time.perf_counter()
    records = find_relevant_records(ticket)
    used_sources = [to_public_source(record, confidence=max(0.5, 1.0 - idx * 0.2)) for idx, record in enumerate(records)]
    logger.info(
        "Knowledge search finished ticket_id=%s records=%s elapsed_ms=%.2f",
        ticket.ticket_id,
        [source.id for source in used_sources],
        (time.perf_counter() - kb_started) * 1000,
    )

    client = llm_client or LLMClient()
    attempts = 0
    last_error = "unknown error"
    for attempt in range(settings.llm_max_retries + 1):
        attempts = attempt + 1
        try:
            llm_started = time.perf_counter()
            llm_result = await client.analyze(ticket, records)
            logger.info("LLM call succeeded ticket_id=%s elapsed_ms=%.2f", ticket.ticket_id, (time.perf_counter() - llm_started) * 1000)
            risk = calculate_risk(ticket, llm_result.risk_level, bool(used_sources))
            needs_review = llm_result.needs_human_review or risk in {Priority.high, Priority.critical} or not used_sources
            status = ProcessingStatus.success if used_sources else ProcessingStatus.partial_success
            result = AnalysisResult(
                ticket_id=ticket.ticket_id,
                summary=llm_result.summary,
                probable_cause=llm_result.probable_cause,
                recommended_actions=llm_result.recommended_actions,
                risk_level=risk,
                needs_human_review=needs_review,
                used_sources=used_sources,
                model_info=ModelInfo(provider=settings.llm_provider, model=settings.llm_model, attempts=attempts),
                processing_status=status,
            )
            logger.info("LLM response validation passed ticket_id=%s", ticket.ticket_id)
            logger.info("Processing finished ticket_id=%s status=%s elapsed_ms=%.2f", ticket.ticket_id, result.processing_status.value, (time.perf_counter() - started) * 1000)
            return result
        except Exception as exc:
            last_error = str(exc)
            logger.warning("LLM attempt failed ticket_id=%s attempt=%s error=%s", ticket.ticket_id, attempts, exc)
            if not isinstance(exc, LLMError):
                last_error = "LLM call failed"

    result = fallback_result(ticket, attempts, last_error, used_sources)
    logger.info("LLM response validation failed ticket_id=%s", ticket.ticket_id)
    logger.info("Processing finished ticket_id=%s status=%s elapsed_ms=%.2f", ticket.ticket_id, result.processing_status.value, (time.perf_counter() - started) * 1000)
    return result
