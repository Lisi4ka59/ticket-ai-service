import re
from dataclasses import dataclass
from app.models import KnowledgeSource, TicketRequest


@dataclass(frozen=True)
class KnowledgeRecord:
    id: str
    title: str
    problem_type: str
    keywords: tuple[str, ...]
    symptoms: str
    actions: tuple[str, ...]


KNOWLEDGE_BASE: tuple[KnowledgeRecord, ...] = (
    KnowledgeRecord(
        id="kb-pppoe-auth",
        title="PPPoE authentication failures",
        problem_type="auth",
        keywords=("pppoe", "auth", "authentication", "login", "password", "radius", "bras"),
        symptoms="Mass PPPoE session failures, RADIUS or BRAS errors, user reports about inability to connect.",
        actions=("Check RADIUS and BRAS logs", "Verify AAA availability", "Review recent subscriber profile changes"),
    ),
    KnowledgeRecord(
        id="kb-iptv-multicast",
        title="IPTV or multicast issues",
        problem_type="iptv",
        keywords=("iptv", "multicast", "igmp", "channel", "tv", "stream"),
        symptoms="IPTV freezes, missing multicast stream, IGMP snooping or querying issues.",
        actions=("Check multicast groups", "Check IGMP on switches", "Compare affected VLANs with baseline VLANs"),
    ),
    KnowledgeRecord(
        id="kb-loss-latency",
        title="Packet loss or latency increase on a network node",
        problem_type="network_degradation",
        keywords=("loss", "packet", "latency", "delay", "ping", "jitter", "error", "errors"),
        symptoms="Increased packet loss, latency, or jitter on a node with service quality degradation.",
        actions=("Check interface utilization", "Check CRC and input errors", "Check adjacent nodes and backbone links"),
    ),
    KnowledgeRecord(
        id="kb-equipment-down",
        title="Equipment unavailable",
        problem_type="equipment_down",
        keywords=("down", "unavailable", "unreachable", "no response", "equipment", "switch", "router", "olt", "node"),
        symptoms="Equipment does not respond over ICMP or SNMP, possible power or management link failure.",
        actions=("Check power and UPS", "Check access through the backup channel", "Escalate to the on-call team if the outage is confirmed"),
    ),
    KnowledgeRecord(
        id="kb-monitoring-alarm",
        title="Monitoring alarm events",
        problem_type="monitoring_alarm",
        keywords=("alarm", "alert", "monitoring", "incident", "triggered", "metric", "threshold"),
        symptoms="Monitoring threshold alert, possible real incident or false positive.",
        actions=("Check metric trends", "Compare with dependent alerts", "Confirm user impact"),
    ),
)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def score_records_for_text(text: str) -> list[tuple[float, KnowledgeRecord]]:
    haystack = text.lower()
    token_set = _tokens(haystack)
    scored: list[tuple[float, KnowledgeRecord]] = []

    for record in KNOWLEDGE_BASE:
        score = 0.0
        for keyword in record.keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in haystack:
                score += 2.0
            elif keyword_lower in token_set:
                score += 1.0
        if record.problem_type in haystack:
            score += 1.5
        if score > 0:
            scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def find_relevant_records(ticket: TicketRequest, limit: int = 3) -> list[KnowledgeRecord]:
    haystack = " ".join([
        ticket.service,
        ticket.description,
        ticket.metrics.node if ticket.metrics and ticket.metrics.node else "",
        ticket.source.value,
    ])
    scored = score_records_for_text(haystack)
    return [record for _, record in scored[:limit]]


def to_public_source(record: KnowledgeRecord, confidence: float) -> KnowledgeSource:
    return KnowledgeSource(id=record.id, title=record.title, problem_type=record.problem_type, confidence=confidence)


def build_context(records: list[KnowledgeRecord]) -> str:
    if not records:
        return "No relevant knowledge base records were found. State that manual review is required."
    parts = []
    for record in records:
        parts.append(
            f"ID: {record.id}\nTitle: {record.title}\nType: {record.problem_type}\nSymptoms: {record.symptoms}\nRecommended checks: {'; '.join(record.actions)}"
        )
    return "\n\n".join(parts)
