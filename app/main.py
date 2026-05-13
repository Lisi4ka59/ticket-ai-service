import json
import logging
import time

from fastapi import FastAPI
from fastapi import Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from app.logging_config import configure_logging
from app.models import AnalysisResult, TicketRequest
from app.service import analyze_ticket

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ticket AI Analysis Service",
    version="1.0.0",
    description="Mini-service for technical ticket analysis with a local language model.",
)


@app.middleware("http")
async def add_request_started_at(request: Request, call_next):
    request.state.started_at = time.perf_counter()
    logger.info("HTTP request received method=%s path=%s", request.method, request.url.path)
    return await call_next(request)


async def _safe_ticket_id_from_body(request: Request) -> str:
    try:
        body = await request.body()
        payload = json.loads(body.decode("utf-8")) if body else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "-"
    ticket_id = payload.get("ticket_id") if isinstance(payload, dict) else None
    return ticket_id.strip() if isinstance(ticket_id, str) and ticket_id.strip() else "-"


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(request: Request, exc: RequestValidationError):
    started_at = getattr(request.state, "started_at", None)
    elapsed_ms = (time.perf_counter() - started_at) * 1000 if started_at else 0.0
    safe_errors = [{"loc": ".".join(map(str, error["loc"])), "type": error["type"]} for error in exc.errors()]
    logger.warning(
        "Input validation failed path=%s ticket_id=%s errors=%s elapsed_ms=%.2f",
        request.url.path,
        await _safe_ticket_id_from_body(request),
        safe_errors,
        elapsed_ms,
    )
    return await request_validation_exception_handler(request, exc)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/analyze", response_model=AnalysisResult)
async def analyze(request: TicketRequest) -> AnalysisResult:
    return await analyze_ticket(request)
