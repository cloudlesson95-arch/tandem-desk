"""Deterministic parts of the agents: business rules, approval tokens, trace resets."""
import os
from datetime import date

import pytest

from agent import approval
from agent.graph import RESET, add_trace, normalize_citations
from agent.tools import CancelRental, ExtendRental, RuleViolation, check_action, describe_proposal

ACTIVE = {"id": "R-2001", "bike": "City e-bike", "status": "active", "start_date": "2026-10-18",
          "end_date": "2026-10-21", "daily_rate": 32.0}
BOOKED = {**ACTIVE, "id": "R-2002", "status": "booked", "start_date": "2026-10-30", "end_date": "2026-11-01"}


def test_extension_rules():
    check_action("extend_rental", ExtendRental(rental_id="R-2001", days=3), ACTIVE, ["R-2001"])
    with pytest.raises(RuleViolation, match="limited to 3 days"):
        check_action("extend_rental", ExtendRental(rental_id="R-2001", days=4), ACTIVE, ["R-2001"])
    with pytest.raises(RuleViolation, match="not one of this customer's"):
        check_action("extend_rental", ExtendRental(rental_id="R-2009", days=1), None, ["R-2001"])
    with pytest.raises(RuleViolation, match="last get_rentals"):
        check_action("extend_rental", ExtendRental(rental_id="R-2001", days=1), ACTIVE, [])
    with pytest.raises(RuleViolation, match="only booked or active"):
        returned = {**ACTIVE, "status": "returned"}
        check_action("extend_rental", ExtendRental(rental_id="R-2001", days=1), returned, ["R-2001"])


def test_cancel_only_booked():
    check_action("cancel_rental", CancelRental(rental_id="R-2002"), BOOKED, ["R-2002"])
    with pytest.raises(RuleViolation, match="only booked rentals"):
        check_action("cancel_rental", CancelRental(rental_id="R-2001"), ACTIVE, ["R-2001"])


def test_proposal_summaries():
    summary, after = describe_proposal("extend_rental", ExtendRental(rental_id="R-2001", days=2),
                                       ACTIVE, "plus", date(2026, 10, 19))
    assert after["end_date"] == "2026-10-23" and "$57.60" in summary
    summary, after = describe_proposal("cancel_rental", CancelRental(rental_id="R-2002"),
                                       BOOKED, "standard", date(2026, 10, 19))
    assert after["status"] == "cancelled" and "fee: none" in summary


def test_approval_token_binds_exact_arguments(monkeypatch):
    monkeypatch.setenv("APPROVAL_SECRET", os.getenv("APPROVAL_SECRET") or "test-secret")
    pending = {"action_id": 7, "thread_id": "t-12345678", "customer_id": "C-1001",
               "action": "extend_rental", "args": {"rental_id": "R-2001", "days": 2}}
    token = approval.sign(pending)
    assert approval.verify(pending, token)
    assert not approval.verify({**pending, "args": {"rental_id": "R-2001", "days": 3}}, token)
    assert not approval.verify({**pending, "customer_id": "C-1002"}, token)
    assert not approval.verify(pending, None)


def test_trace_resets_each_turn():
    trace = add_trace([{"node": "a"}], [RESET, {"node": "b"}])
    assert trace == [{"node": "b"}]
    assert add_trace(trace, [{"node": "c"}]) == [{"node": "b"}, {"node": "c"}]


def test_native_citations_normalized():
    assert normalize_citations("Open 8:00【P1】 and 9:00【P2†L3-L4】.") == "Open 8:00[P1] and 9:00[P2]."
