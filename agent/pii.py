"""Deterministic PII redaction applied before any text reaches an external LLM.

Each match is replaced with a typed placeholder ([EMAIL_1], [CARD_2], ...); the same value
gets the same placeholder within one text. Card numbers must pass the Luhn check and IBANs
the mod-97 check, so order numbers and dates are left alone."""
import re
from dataclasses import dataclass, field

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b", re.I)
CARD = re.compile(
    r"(?<![\d-])(?:\d{13,19}"  # contiguous
    r"|\d{4}([ -])\d{4}\1\d{4}\1\d{1,7}"  # 4-4-4-x
    r"|\d{4}([ -])\d{6}\2\d{4,5})(?![\d-])"  # 4-6-5 (Amex)
)
PHONE = re.compile(
    r"(?<![\w+])(?:"
    r"\+\d{1,3}(?:[ .-]?\(?\d{1,4}\)?){2,5}"  # +44 20 7946 0958, +1-555-0101
    r"|\(\d{3}\)[ .-]?\d{3}[ .-]\d{4}"  # (555) 010-1234
    r"|\d{3}[ .-]\d{3}[ .-]\d{4}"  # 555-010-1234
    r"|0\d{2,4}[ .-]?\d{3,4}[ .-]?\d{3,4}"  # 0612 345 678
    r")(?!\w)"
)
_SUFFIX = (
    r"(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Boulevard|Blvd|Drive|Dr|Way|Court|Ct"
    r"|Place|Pl|Terrace|Close|Crescent|Square|Sq)"
)
ADDRESS = re.compile(
    rf"\b\d{{1,5}}[A-Z]?\s+(?:[A-Z][a-z]+\s+){{1,3}}{_SUFFIX}\b\.?"
    r"(?:,?\s+(?:Apt|Apartment|Unit|Suite|Flat)\.?\s*[A-Za-z0-9]+)?"
    r"(?:,?\s+\d{5}(?:-\d{4})?)?"
)


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def iban_ok(value: str) -> bool:
    s = value.replace(" ", "").upper()
    if not 15 <= len(s) <= 34:
        return False
    return int("".join(str(int(c, 36)) for c in s[4:] + s[:4])) % 97 == 1


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def _phone_ok(value: str) -> bool:
    n = len(_digits(value))
    return 8 <= n <= 15 if value.startswith("+") else 10 <= n <= 11


def _mask(kind: str, value: str) -> str:
    """A preview for the trace that never shows the full value."""
    if kind == "EMAIL":
        local, _, domain = value.partition("@")
        return f"{local[:1]}***@{domain}"
    if kind in ("CARD", "IBAN"):
        compact = value.replace(" ", "").replace("-", "")
        return f"****{compact[-4:]}"
    if kind == "PHONE":
        return f"***{_digits(value)[-2:]}"
    return f"*** {value.split()[-1]}"


# Order matters: IBANs and cards contain digit runs that the phone pattern would also take.
RULES = [
    ("EMAIL", EMAIL, lambda v: True),
    ("IBAN", IBAN, iban_ok),
    ("CARD", CARD, lambda v: luhn_ok(_digits(v))),
    ("PHONE", PHONE, _phone_ok),
    ("ADDRESS", ADDRESS, lambda v: True),
]


@dataclass
class Redaction:
    text: str
    redactions: list[dict] = field(default_factory=list)


def redact(text: str) -> Redaction:
    redactions: list[dict] = []
    for kind, pattern, valid in RULES:
        seen: dict[str, str] = {}

        def replace(m: re.Match) -> str:
            value = m.group(0)
            if not valid(value):
                return value
            if value not in seen:
                seen[value] = f"[{kind}_{len(seen) + 1}]"
                redactions.append(
                    {"type": kind, "placeholder": seen[value], "preview": _mask(kind, value)}
                )
            return seen[value]

        text = pattern.sub(replace, text)
    return Redaction(text, redactions)
