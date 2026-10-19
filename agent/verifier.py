"""Pre-display fact check: every claim in a drafted answer must be supported by an
evidence item retrieved in this turn, or the answer is replaced with a fallback."""
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.llm import agent_llm

FALLBACK = (
    "I couldn't confirm that in our policies, so I won't guess. Please check with one of "
    "our desks, or try rephrasing the question."
)
EVIDENCE_ID = re.compile(r"\b[PMA]\d+\b")
CITATION = re.compile(r"\[([PMA]\d+(?:\s*,\s*[PMA]\d+)*)\]")

PROMPT = """You fact-check a customer-support answer for Velomarsh, a fictional bike-rental service, before it is shown.
Evidence items have ids: P1, P2... are policy passages, A1... are the customer's rental records,
M1... are notes from the customer's earlier conversations.
List every factual claim in the answer: each statement about prices, fees, times, dates, rules,
rentals or the customer. Skip greetings, offers to help, questions to the customer, and
statements that the policies don't cover something.
For each claim, set supported to true only if a single evidence item states it. Numbers, times,
dates and amounts must match exactly. Give that item's id as evidence_id, or null if none supports it."""


class Claim(BaseModel):
    claim: str = Field(description="One factual statement from the answer.")
    supported: bool = Field(description="True only if the evidence item states it.")
    evidence_id: str | None = Field(description="Id of the supporting evidence item, e.g. P2, or null.")


class Verification(BaseModel):
    claims: list[Claim]


def cited_ids(text: str) -> set[str]:
    return {i for group in CITATION.findall(text) for i in EVIDENCE_ID.findall(group)}


async def verify(draft: str, evidence: list[dict]) -> dict:
    ids = {e["pid"] for e in evidence}
    listing = "\n\n".join(f"[{e['pid']}] {e['content']}" for e in evidence) or "(no evidence was retrieved)"
    try:
        result = await agent_llm().with_structured_output(Verification, method="json_schema").ainvoke([
            SystemMessage(PROMPT),
            HumanMessage(f"Evidence:\n{listing}\n\nAnswer to check:\n{draft}"),
        ])
    except Exception as e:  # fail closed: an unchecked answer is never shown
        return {"claims": [], "problems": [f"verifier unavailable: {type(e).__name__}"], "blocked": True}
    claims = []
    for c in result.claims:
        cited = set(EVIDENCE_ID.findall(c.evidence_id or ""))
        retrieved = bool(cited & ids)
        ok = c.supported and retrieved
        reason = None
        if not c.supported:
            reason = "not supported by the evidence"
        elif not retrieved:
            reason = f"cites {c.evidence_id or 'nothing'}, which was not retrieved"
        claims.append({"claim": c.claim, "supported": ok, "evidence_id": c.evidence_id, "reason": reason})
    problems = []
    unknown = sorted(cited_ids(draft) - ids)
    if unknown:
        problems.append(f"the answer cites {', '.join(unknown)}, which were not retrieved")
    blocked = bool(problems) or any(not c["supported"] for c in claims)
    return {"claims": claims, "problems": problems, "blocked": blocked}
