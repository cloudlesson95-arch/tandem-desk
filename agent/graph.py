"""The Tandem Desk graph.

input_guard → supervisor → knowledge_agent | account_agent → (human_approval) → verifier
→ memory_writer. Every node is async and appends one trace entry with its latency and
decisions; refused turns end at the guard and never reach an agent."""
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from psycopg import AsyncConnection
from pydantic import ValidationError

from agent import approval, memory, tools
from agent.guards import Topic, check_input, refusal_for
from agent.llm import agent_llm
from agent.pii import redact

MAX_RETRIES = 3  # rephrased searches after the first one
MAX_ACCOUNT_STEPS = 4
HISTORY = 6  # earlier messages of the thread given to the agents
RESET = {"reset": True}
# gpt-oss sometimes cites in its native style, e.g. 【P1】 or 【P1†L2-L4】, and writes
# narrow no-break spaces and non-breaking hyphens.
NATIVE_CITATION = re.compile(r"【\s*([PMA]\d+)[^】]*】")
TYPOGRAPHY = str.maketrans({" ": " ", " ": " ", " ": " ", "‑": "-"})


def normalize_citations(text: str) -> str:
    return NATIVE_CITATION.sub(r"[\1]", text.translate(TYPOGRAPHY))


def add_trace(old: list | None, new: list | None) -> list:
    """Append trace entries; a RESET entry starts a new turn's trace."""
    out = list(old or [])
    for entry in new or []:
        out = [] if entry.get("reset") else [*out, entry]
    return out


class State(TypedDict, total=False):
    customer_id: str
    thread_id: str
    message: str  # redacted by input_guard before any model sees it
    verify: bool
    memory: bool
    messages: Annotated[list[AnyMessage], add_messages]
    redactions: list[dict]
    guard: dict | None
    route: str | None
    memories: list[dict]
    evidence: list[dict]  # passages, rental lookups and memories an answer may rely on
    last_lookup: list[str]
    draft: str | None
    answer: str | None
    status: str | None  # answered | refused | blocked | pending | done | rejected
    pending_action: dict | None
    action_result: dict | None
    verification: dict | None
    trace: Annotated[list[dict], add_trace]


@dataclass
class Ctx:
    db: AsyncConnection


def turn_input(customer_id: str, thread_id: str, message: str, verify: bool = True,
               use_memory: bool = True) -> dict:
    """Input for a new turn: resets every per-turn field of the thread's state."""
    return {
        "customer_id": customer_id, "thread_id": thread_id, "message": message,
        "verify": verify, "memory": use_memory, "trace": [RESET], "redactions": [],
        "guard": None, "route": None, "memories": [], "evidence": [], "draft": None,
        "answer": None, "status": None, "pending_action": None, "action_result": None,
        "verification": None,
    }


def traced(name: str):
    """Wrap `fn(state, db) -> (update, info)` as a node that records a trace entry."""
    def decorate(fn):
        async def node(state: State, runtime: Runtime[Ctx]) -> dict:
            started = time.perf_counter()
            update, info = await fn(state, runtime.context.db)
            ms = round((time.perf_counter() - started) * 1000)
            return {**update, "trace": [{"node": name, "ms": ms, **info}]}
        node.__name__ = name
        return node
    return decorate


def _history(state: State) -> list[AnyMessage]:
    # The current message was just appended by input_guard; agents get it separately.
    return list(state.get("messages", [])[:-1])[-HISTORY:]


def _short(text: str | None, n: int = 400) -> str | None:
    if not text:
        return None
    return text if len(text) <= n else text[:n] + "…"


def _memory_note(state: State) -> list[SystemMessage]:
    mems = state.get("memories") or []
    if not mems:
        return []
    lines = "\n".join(f"[M{i + 1}] ({m['kind']}) {m['content']}" for i, m in enumerate(mems))
    return [SystemMessage(f"Notes from this customer's earlier conversations:\n{lines}")]


def _memory_evidence(state: State) -> list[dict]:
    return [
        {"pid": f"M{i + 1}", "kind": "memory", "title": m["kind"], "content": m["content"]}
        for i, m in enumerate(state.get("memories") or [])
    ]


# ---- nodes -----------------------------------------------------------------

@traced("input_guard")
async def input_guard(state: State, db: AsyncConnection):
    red = redact(state["message"])
    previous = next(
        (m.content for m in reversed(state.get("messages", [])) if isinstance(m, AIMessage)), None
    )
    guard = await check_input(red.text, _short(previous, 300))
    decision = guard.model_dump(mode="json")
    update: dict[str, Any] = {
        "message": red.text, "redactions": red.redactions, "guard": decision,
        "messages": [HumanMessage(red.text)],
    }
    refusal = refusal_for(guard.topic)
    if refusal:
        status = "answered" if guard.topic == Topic.smalltalk else "refused"
        update |= {"answer": refusal, "status": status,
                   "messages": [HumanMessage(red.text), AIMessage(refusal)]}
    return update, {"redactions": red.redactions, "guard": decision, "refused": bool(refusal)}


@traced("supervisor")
async def supervisor(state: State, db: AsyncConnection):
    memories = []
    if state.get("memory", True):
        memories = await memory.recall(db, state["customer_id"], state["message"])
    route = "account_agent" if state["guard"]["topic"] == Topic.account_action.value else "knowledge_agent"
    return {"route": route, "memories": memories}, {
        "route": route, "recalled": [f"{m['kind']}: {m['content']}" for m in memories],
    }


KNOWLEDGE_PROMPT = """You are the knowledge agent of the support desk for Velomarsh, a fictional bike-rental service.
Answer the customer's question using only policy passages returned by the search_policies tool.
1. Always call search_policies first, with 1 to 3 differently worded queries.
2. If the passages are weak or don't cover the question, call search_policies again with rephrased queries.
3. Then answer in at most 120 words. After each fact, cite its passage id in square brackets, e.g. [P2].
   Notes about the customer can be cited as [M1] etc.
If the passages don't answer the question, say the policies don't cover it and suggest asking a desk.
Never guess or use outside knowledge. Placeholders like [EMAIL_1] stand for redacted personal data."""


@traced("knowledge_agent")
async def knowledge_agent(state: State, db: AsyncConnection):
    search_tool = tools.spec("search_policies", tools.SearchPolicies)
    llm = agent_llm().bind_tools([search_tool])
    must_answer = agent_llm().bind_tools([search_tool], tool_choice="none")
    messages: list[AnyMessage] = [
        SystemMessage(KNOWLEDGE_PROMPT), *_memory_note(state), *_history(state),
        HumanMessage(state["message"]),
    ]
    passages: dict[str, dict] = {}  # chunk id -> passage with its turn-local id P1, P2, ...
    steps, searches, draft = [], 0, ""
    max_steps = MAX_RETRIES + 3
    for step_no in range(max_steps):
        # After the first search plus MAX_RETRIES rephrasings the model must answer.
        forced = searches > MAX_RETRIES or step_no == max_steps - 1
        reply = await (must_answer if forced else llm).ainvoke(messages)
        thought = _short(reply.additional_kwargs.get("reasoning_content"))
        if not reply.tool_calls:
            draft = str(reply.content).strip()
            steps.append({"thought": thought, "action": "answer"})
            break
        messages.append(reply)
        for call in reply.tool_calls:
            if call["name"] not in tools.KNOWLEDGE_TOOLS:
                observation = f"Tool {call['name']} is not available to the knowledge agent."
                steps.append({"thought": thought, "tool": call["name"], "blocked": True})
            else:
                try:
                    args = tools.SearchPolicies.model_validate(call["args"])
                except ValidationError as e:
                    observation = f"Invalid arguments: {e.errors()[0]['msg']}. Pass 1 to 3 queries."
                else:
                    searches += 1
                    found = await tools.search_policies(db, args.queries)
                    for row in found:
                        passages.setdefault(row["id"], {
                            "pid": f"P{len(passages) + 1}", "kind": "policy", "id": row["id"],
                            "title": row["title"], "content": row["content"], "score": row["score"],
                        })
                    shown = [passages[r["id"]] | {"score": r["score"]} for r in found]
                    best = max((p["score"] for p in shown), default=0)
                    observation = "\n\n".join(f"[{p['pid']}] {p['content']}" for p in shown)
                    if best < tools.WEAK_SCORE:
                        observation += (f"\n\nThese passages look weak (best score {best:.2f}). If they "
                                        "don't answer the question, search again with rephrased queries.")
                    steps.append({
                        "thought": thought, "tool": "search_policies", "args": args.queries,
                        "observation": [f"{p['pid']} {p['title']} ({p['score']:.2f})" for p in shown],
                        "weak": best < tools.WEAK_SCORE,
                    })
            messages.append(ToolMessage(observation, tool_call_id=call["id"]))
    draft = normalize_citations(draft) or (
        "I couldn't find that in our policies. A desk can help with this question."
    )
    evidence = [*passages.values(), *_memory_evidence(state)]
    return {"draft": draft, "evidence": evidence}, {"steps": steps, "searches": searches}


ACCOUNT_PROMPT = """You are the account agent of the support desk for Velomarsh, a fictional bike-rental service.
You help the current customer with their own rentals, using only the tools get_rentals, extend_rental and cancel_rental.
- Call get_rentals before proposing a change, and use only rental ids it returned.
- extend_rental and cancel_rental only propose a change; the customer approves it in the app.
  Use exactly the rental and number of days the customer asked for.
- If a tool reports a rule violation, explain it briefly and don't retry with other values.
- If it is unclear which rental the customer means, ask a short question.
- Reply in at most 80 words and state only facts from tool results.
Today is {today}."""


@traced("account_agent")
async def account_agent(state: State, db: AsyncConnection):
    cid = state["customer_id"]
    specs = [tools.spec(name, model) for name, model in tools.ACCOUNT_TOOLS.items()]
    llm = agent_llm().bind_tools(specs)
    messages: list[AnyMessage] = [
        SystemMessage(ACCOUNT_PROMPT.format(today=date.today().isoformat())), *_memory_note(state),
        *_history(state), HumanMessage(state["message"]),
    ]
    last_lookup = list(state.get("last_lookup") or [])
    rentals: dict[str, dict] = {}
    steps, draft = [], ""
    for _ in range(MAX_ACCOUNT_STEPS):
        reply = await llm.ainvoke(messages)
        thought = _short(reply.additional_kwargs.get("reasoning_content"))
        if not reply.tool_calls:
            draft = str(reply.content).strip()
            steps.append({"thought": thought, "action": "answer"})
            break
        messages.append(reply)
        for call in reply.tool_calls:
            name, step = call["name"], {"thought": thought, "tool": call["name"], "args": call["args"]}
            if name not in tools.ACCOUNT_TOOLS:
                observation = f"Tool {name} is not available to the account agent."
                step["blocked"] = True
            elif name == "get_rentals":
                found = await tools.fetch_rentals(db, cid)
                rentals = {r["id"]: r for r in found}
                last_lookup = list(rentals)
                observation = "\n".join(
                    f"{r['id']}: {r['bike']}, {r['status']}, {r['start_date']} to {r['end_date']}, "
                    f"${r['daily_rate']:.2f}/day" for r in found
                ) or "No rentals."
                step["observation"] = last_lookup
            else:
                try:
                    args = tools.ACCOUNT_TOOLS[name].model_validate(call["args"])
                    if not rentals:
                        rentals = {r["id"]: r for r in await tools.fetch_rentals(db, cid)}
                    rental = rentals.get(args.rental_id)
                    tools.check_action(name, args, rental, last_lookup)
                except ValidationError as e:
                    observation = f"Invalid arguments: {e.errors()[0]['msg']}."
                    step["rule"] = observation
                except tools.RuleViolation as e:
                    observation = f"Rule violation: {e}"
                    step["rule"] = str(e)
                else:
                    tier = await tools.fetch_tier(db, cid)
                    summary, after = tools.describe_proposal(name, args, rental, tier, date.today())
                    action_id = await tools.log_pending(db, cid, state["thread_id"], name, args.model_dump())
                    pending = {
                        "action_id": action_id, "thread_id": state["thread_id"], "customer_id": cid,
                        "action": name, "args": args.model_dump(), "summary": summary,
                        "before": rental, "after": after,
                    }
                    step["proposed"] = summary
                    steps.append(step)
                    answer = f"{summary} Please approve or reject this change."
                    return {
                        "pending_action": pending, "last_lookup": last_lookup, "answer": answer,
                        "status": "pending", "messages": [AIMessage(answer)],
                    }, {"steps": steps, "paused": "waiting for approval"}
            steps.append(step)
            messages.append(ToolMessage(observation, tool_call_id=call["id"]))
    evidence = [
        {"pid": f"A{i + 1}", "kind": "rental", "title": r["id"],
         "content": f"{r['id']}: {r['bike']}, status {r['status']}, from {r['start_date']} to "
                    f"{r['end_date']}, ${r['daily_rate']:.2f} per day"}
        for i, r in enumerate(rentals.values())
    ] + _memory_evidence(state)
    draft = normalize_citations(draft) or (
        "Sorry, I couldn't complete that. Could you say which rental you mean?"
    )
    return {"draft": draft, "evidence": evidence, "last_lookup": last_lookup}, {"steps": steps}


@traced("human_approval")
async def human_approval(state: State, db: AsyncConnection):
    pending = state["pending_action"]
    # Pauses the graph here; resumes with {"decision": ..., "token": ...} from /resume.
    reply = interrupt(pending)
    approved = reply.get("decision") == "approve"
    token_ok = approval.verify(pending, reply.get("token"))
    if approved and token_ok:
        try:
            result = await tools.execute_action(db, pending)
        except tools.RuleViolation as e:
            await tools.log_decision(db, pending["action_id"], "failed", {"error": str(e)})
            answer, status, result = f"I couldn't make that change: {e}", "rejected", None
        else:
            await tools.log_decision(db, pending["action_id"], "executed", result)
            what = (f"now ends on {result['end_date']}" if pending["action"] == "extend_rental"
                    else "is cancelled")
            answer, status = f"Done. Rental {result['id']} ({result['bike']}) {what}.", "done"
    else:
        await tools.log_decision(db, pending["action_id"], "rejected",
                                 None if token_ok or not approved else {"error": "invalid approval token"})
        answer, status, result = "OK, I haven't changed anything.", "rejected", None
    return {
        "answer": answer, "status": status, "action_result": result, "draft": answer,
        "messages": [AIMessage(answer)],
    }, {"decision": reply.get("decision"), "token_valid": token_ok, "result": result}


@traced("verifier")
async def verifier(state: State, db: AsyncConnection):
    # Phase 4 replaces this pass-through with the claim-level fact check.
    answer = state["draft"]
    return {"answer": answer, "status": "answered", "messages": [AIMessage(answer)]}, {"checked": False}


@traced("memory_writer")
async def memory_writer(state: State, db: AsyncConnection):
    if not state.get("memory", True):
        return {}, {"skipped": "memory disabled for this run"}
    stored = await memory.write(
        db, state["customer_id"], state["thread_id"], state["message"], state.get("answer") or ""
    )
    return {}, {"stored": stored}


def build_graph() -> StateGraph:
    g = StateGraph(State, context_schema=Ctx)
    for node in (input_guard, supervisor, knowledge_agent, account_agent, human_approval,
                 verifier, memory_writer):
        g.add_node(node.__name__, node)
    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", lambda s: END if s.get("answer") else "supervisor",
                            ["supervisor", END])
    g.add_conditional_edges("supervisor", lambda s: s["route"], ["knowledge_agent", "account_agent"])
    g.add_edge("knowledge_agent", "verifier")
    g.add_conditional_edges(
        "account_agent", lambda s: "human_approval" if s.get("pending_action") else "verifier",
        ["human_approval", "verifier"],
    )
    g.add_edge("human_approval", "memory_writer")
    g.add_edge("verifier", "memory_writer")
    g.add_edge("memory_writer", END)
    return g
