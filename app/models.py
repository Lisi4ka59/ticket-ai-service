from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, ValidationInfo, field_validator


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Source(str, Enum):
    monitoring = "monitoring"
    user_report = "user_report"
    noc = "noc"
    manual = "manual"


class ProcessingStatus(str, Enum):
    success = "success"
    partial_success = "partial_success"
    failed = "failed"


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error_rate: float | None = Field(default=None, ge=0, le=1)
    affected_users: StrictInt | None = Field(default=None, ge=0)
    node: StrictStr | None = None

    @field_validator("error_rate", mode="before")
    @classmethod
    def error_rate_must_be_json_number(cls, value: float | int | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("error_rate must be a JSON number")
        return float(value)


class TicketRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: StrictStr = Field(min_length=1)
    source: Source
    priority: Priority
    service: StrictStr = Field(min_length=1)
    description: StrictStr = Field(min_length=1)
    metrics: Metrics | None = None

    @field_validator("ticket_id", "service", "description")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value


class KnowledgeSource(BaseModel):
    id: str
    title: str
    problem_type: str
    confidence: float = Field(ge=0, le=1)


class ModelInfo(BaseModel):
    provider: str
    model: str
    attempts: int
    fallback_used: bool = False


class AnalysisResult(BaseModel):
    ticket_id: str
    summary: str
    probable_cause: str
    recommended_actions: list[str]
    risk_level: Priority
    needs_human_review: bool
    used_sources: list[KnowledgeSource]
    model_info: ModelInfo
    processing_status: ProcessingStatus


class LLMStructuredResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: StrictStr = Field(min_length=1)
    probable_cause: StrictStr = Field(min_length=1)
    recommended_actions: list[StrictStr] = Field(min_length=1)
    risk_level: Priority
    needs_human_review: StrictBool

    @field_validator("summary", "probable_cause")
    @classmethod
    def llm_text_must_not_be_blank(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")
        return value

    @field_validator("recommended_actions")
    @classmethod
    def actions_must_not_be_blank(cls, value: list[str]) -> list[str]:
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError("recommended_actions must contain non-empty strings")
        return value


class ErrorResponse(BaseModel):
    detail: Any
