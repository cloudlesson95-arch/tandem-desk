"""40 synthetic PII strings: fake emails, test card numbers, generated IBANs with valid
checksums, fictional 555 phone numbers and made-up addresses."""
import pytest

from agent.pii import iban_ok, luhn_ok, redact

CASES = [
    # Emails
    ("My email is jane.doe@example.com", "My email is [EMAIL_1]"),
    ("Contact: bob_smith+rentals@mail.example.org.", "Contact: [EMAIL_1]."),
    ("Send it to A.B@Example.co.uk please", "Send it to [EMAIL_1] please"),
    ("Use x1@example.net or y2@example.net", "Use [EMAIL_1] or [EMAIL_2]"),
    ("Same twice: z@example.com, z@example.com", "Same twice: [EMAIL_1], [EMAIL_1]"),
    ("email me at ops-team@sub.example.io", "email me at [EMAIL_1]"),
    # Phones
    ("Call me on +1-555-0101", "Call me on [PHONE_1]"),
    ("My number is (555) 010-4477", "My number is [PHONE_1]"),
    ("Phone: 555-010-9876.", "Phone: [PHONE_1]."),
    ("Reach me at +44 20 7946 0958 after 6pm", "Reach me at [PHONE_1] after 6pm"),
    ("mobile 0612 345 678", "mobile [PHONE_1]"),
    ("+49 30 901820 is my office", "[PHONE_1] is my office"),
    ("Call 555.010.2222 or +1 555 010 3333", "Call [PHONE_1] or [PHONE_2]"),
    # Cards (test numbers that pass Luhn)
    ("My card is 4111 1111 1111 1111", "My card is [CARD_1]"),
    ("Card: 5555-5555-5555-4444", "Card: [CARD_1]"),
    ("Amex 3782 822463 10005 expires soon", "Amex [CARD_1] expires soon"),
    ("Use 4012888888881881 for the deposit", "Use [CARD_1] for the deposit"),
    ("Pay with 6011 1111 1111 1117 today", "Pay with [CARD_1] today"),
    ("Card 4242424242424242.", "Card [CARD_1]."),
    # IBANs (generated, valid mod-97)
    ("IBAN GB02 ZZZZ 0000 1234 5678 90", "IBAN [IBAN_1]"),
    ("Refund to DE91100000000123456789 please", "Refund to [IBAN_1] please"),
    ("Account: FR80 1000 0000 0000 0000 0000 000", "Account: [IBAN_1]"),
    ("NL51ZZZZ0123456789 is mine", "[IBAN_1] is mine"),
    ("iban: es96 0000 0000 0001 2345 6789", "iban: [IBAN_1]"),
    # Addresses
    ("I live at 42 Elm Street", "I live at [ADDRESS_1]"),
    ("Deliver to 1600 Maple Avenue, Apt 5B", "Deliver to [ADDRESS_1]"),
    ("My address is 17 Harbour Road.", "My address is [ADDRESS_1]"),
    ("Office at 350 North Lake Blvd", "Office at [ADDRESS_1]"),
    ("Pick up from 8 Old Mill Way please", "Pick up from [ADDRESS_1] please"),
    # Mixed
    (
        "I'm jane@example.com, card 4111111111111111, call +1-555-0199",
        "I'm [EMAIL_1], card [CARD_1], call [PHONE_1]",
    ),
    (
        "Refund GB02ZZZZ00001234567890 and mail 9 Willow Lane",
        "Refund [IBAN_1] and mail [ADDRESS_1]",
    ),
    ("Card 4111 1111 1111 1112 fails the check", "Card 4111 1111 1111 1112 fails the check"),
    # Clean strings stay unchanged
    ("What is the late fee for a city e-bike?", None),
    ("Can I extend R-2001 by 2 days?", None),
    ("My rental runs from 2026-10-21 to 2026-10-23.", None),
    ("The desk opens at 8:00 and closes at 20:00.", None),
    ("It costs $45 per day and the deposit is $300.", None),
    ("I want 3 Road bikes near Main Street for 7 days.", None),
    ("Order number 1234567890 was confirmed.", None),
    ("IBAN GB00 ZZZZ 0000 1234 5678 90 is a typo", None),
]


def test_case_count():
    assert len(CASES) == 40


@pytest.mark.parametrize("text,expected", CASES)
def test_redact(text, expected):
    result = redact(text)
    assert result.text == (text if expected is None else expected)
    if expected is None:
        assert result.redactions == []
    else:
        placeholders = {r["placeholder"] for r in result.redactions}
        assert all(p in result.text for p in placeholders)


def test_previews_hide_values():
    result = redact("card 4111 1111 1111 1111, mail jane.doe@example.com")
    previews = [r["preview"] for r in result.redactions]
    assert previews == ["j***@example.com", "****1111"]


def test_checksums():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    assert iban_ok("GB02 ZZZZ 0000 1234 5678 90") and not iban_ok("GB00ZZZZ00001234567890")
