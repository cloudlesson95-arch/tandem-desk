"""Pre-display fact check: every claim in a drafted answer must be supported by evidence
retrieved in this turn, or the answer is replaced with a fallback."""
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.llm import agent_call

FALLBACK = (
    "I couldn't confirm that in our policies, so I won't guess. Please check with one of "
    "our desks, or try rephrasing the question."
)
EVIDENCE_ID = re.compile(r"\b[PMA]\d+\b")
CITATION = re.compile(r"\[\s*([PMA]\d+(?:\s*,\s*[PMA]\d+)*)\s*\]")
# Saying the policies don't cover something is the safe answer, never a claim to block.
NOT_COVERED = re.compile(
    r"(?:do not|don't|does not|doesn't|did not|didn't) (?:cover|mention|say|include|specify|address|state|list|contain)"
    r"|(?:could not|couldn't|can't|cannot) (?:find|confirm)"
    r"|not (?:covered|mentioned|specified|addressed|stated|included)|no (?:information|details|mention|guidance)"
    r"|\bno [\w\s-]{1,40}? (?:is|are) (?:provided|given|listed|stated|mentioned)",
    re.I,
)

PROMPT = """You fact-check a customer-support answer for Velomarsh, a fictional bike-rental service, before it is shown.
Evidence items have ids: P1, P2... are policy passages, A1... are the customer's rental records,
M1... are notes from the customer's earlier conversations.
List every factual claim in the answer: each statement about prices, fees, times, dates, rules,
rentals or the customer. Skip greetings, offers to help, questions to the customer, and
statements that the policies don't cover something.
For each claim, set supported to true only if the evidence states it. A claim may combine facts
from several items; list every item it relies on. Numbers, times, dates and amounts must match
exactly. Give the supporting ids as evidence_ids, or an empty list if nothing supports the claim."""


class Claim(BaseModel):
    claim: str = Field(description="One factual statement from the answer.")
    supported: bool = Field(description="True only if the listed evidence states it.")
    evidence_ids: list[str] = Field(description="Ids of the supporting evidence items, e.g. ['P2'] or ['P1', 'P3'].")


class Verification(BaseModel):
    claims: list[Claim]


def cited_ids(text: str) -> set[str]:
    return {i for group in CITATION.findall(text) for i in EVIDENCE_ID.findall(group)}


async def verify(draft: str, evidence: list[dict]) -> dict:
    ids = {e["pid"] for e in evidence}
    listing = "\n\n".join(f"[{e['pid']}] {e['content']}" for e in evidence) or "(no evidence was retrieved)"
    try:
        result, model = await agent_call(
            lambda m: m.with_structured_output(Verification, method="json_schema"),
            [SystemMessage(PROMPT), HumanMessage(f"Evidence:\n{listing}\n\nAnswer to check:\n{draft}")],
        )
    except Exception as e:  # fail closed: an unchecked answer is never shown
        return {"claims": [], "problems": [f"verifier unavailable: {type(e).__name__}"], "blocked": True}
    claims = []
    for c in result.claims:
        if NOT_COVERED.search(c.claim):
            continue
        cited = {i for ref in c.evidence_ids for i in EVIDENCE_ID.findall(ref)}
        retrieved = bool(cited) and cited <= ids
        reason = None
        if not c.supported:
            reason = "not supported by the evidence"
        elif not retrieved:
            reason = f"cites {', '.join(sorted(cited - ids)) or 'nothing'}, which was not retrieved"
        claims.append({"claim": c.claim, "supported": c.supported and retrieved,
                       "evidence_ids": sorted(cited), "reason": reason})
    problems = []
    unknown = sorted(cited_ids(draft) - ids)
    if unknown:
        problems.append(f"the answer cites {', '.join(unknown)}, which were not retrieved")
    blocked = bool(problems) or any(not c["supported"] for c in claims)
    return {"claims": claims, "problems": problems, "blocked": blocked, "model": model}
