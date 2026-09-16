"""Sensitive-data detection catalogue — the shapes the guardrail layer recognises.

One definition read by `redact_text` (persistence) and `output_guard.py` (layer-7 screen), so
a shape added here is masked at every boundary; tiering lives in `output_guard.OutputPolicy`.
Rules: every finding carries a category (what the policy acts on); checksums and context words
gate ambiguous digit runs; names match only when a role word anchors them (unanchored names are
the recorded residual); precision over recall — a false positive costs a reader a placeholder.
Sources: OWASP LLM02:2025, Presidio, gitleaks, detect-secrets, Google DLP, AWS Bedrock, PCI DSS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

# ── categories ──────────────────────────────────────────────────────────────
#
# The policy vocabulary. Each finding belongs to exactly one, and
# `output_guard.OutputPolicy` maps each to an action.
CREDENTIAL = "credential"  # token / key / private key / URL credentials
SECRET_ASSIGNMENT = "secret_assignment"  # noqa: S105 — a category name, not a value
GOVERNMENT_ID = "government_id"  # SSN, NINO, passport
PAYMENT = "payment"  # card numbers, CVV, expiry
BANK = "bank"  # IBAN, account/routing numbers
HEALTH_ID = "health_id"  # MRN, NHS number, patient/record identifiers
HEALTH_CONDITION = "health_condition"  # diagnoses, treatment, medication
PERSON = "person"  # a named individual
CONTACT = "contact"  # email, phone
DOB = "dob"  # date of birth
NETWORK = "network"  # internal hostnames, private IPs, server paths

CATEGORIES: tuple[str, ...] = (
    CREDENTIAL,
    SECRET_ASSIGNMENT,
    GOVERNMENT_ID,
    PAYMENT,
    BANK,
    HEALTH_ID,
    HEALTH_CONDITION,
    PERSON,
    CONTACT,
    DOB,
    NETWORK,
)

# The vendors' high tier: never delivered raw whatever the local policy says.
# `config_store` refuses a published policy that sets any of these to `allow`.
NEVER_ALLOW: frozenset[str] = frozenset(
    {CREDENTIAL, SECRET_ASSIGNMENT, GOVERNMENT_ID, PAYMENT, BANK, HEALTH_ID}
)

# The "PII tier" — what OUTPUT_PII_MASKING switches. Everything else is masked regardless: a
# token is a leak in any artifact, while a stakeholder's name inside a drafted HLD may be content.
PII_TIER: frozenset[str] = frozenset({PERSON, CONTACT, DOB, HEALTH_CONDITION})

# Counts towards the bulk-disclosure threshold: five card numbers are a data dump, five
# emails a distribution list. Defined as `NEVER_ALLOW` itself so "high tier" means one thing.
BULK_CATEGORIES: frozenset[str] = NEVER_ALLOW


# Tie-break rank when two shapes claim the same span: lower wins.
_TIER_RANK: dict[str, int] = {
    **{category: 0 for category in NEVER_ALLOW},
    SECRET_ASSIGNMENT: 1,
    HEALTH_CONDITION: 2,
    PERSON: 2,
    CONTACT: 2,
    DOB: 2,
    NETWORK: 3,
}


@dataclass(frozen=True)
class Finding:
    """One sensitive value located in a text.

    `label`/`category` are the strongest classification in the span; `labels`/`categories`
    list everything that matched, because overlapping matches are merged, not resolved (`scan`).
    """

    label: str
    category: str
    start: int
    end: int
    value: str
    labels: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()

    @property
    def placeholder(self) -> str:
        return f"[redacted:{self.label}]"


#: A placeholder this module already wrote. Nothing inside one is ever a finding (the
#: `url-credentials` shape matches the placeholder itself), so a second pass is a genuine no-op.
_PLACEHOLDER = re.compile(r"\[redacted:[a-z0-9-]+\]")


# ── validators ──────────────────────────────────────────────────────────────


def luhn_valid(digits: str) -> bool:
    """The Luhn checksum every payment card number satisfies (ISO/IEC 7812).

    ASCII-only on purpose: `\\D` would keep a non-ASCII digit and `ord(char) - 48` would
    compute nonsense from it. `scan` folds such digits before they reach here anyway.
    """
    digits = re.sub(r"[^0-9]", "", digits)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        number = ord(char) - 48
        if index % 2 == 1:
            number *= 2
            if number > 9:
                number -= 9
        total += number
    return total % 10 == 0


# Country -> IBAN length, ISO 13616 registry. Deliberately broad: an unknown country means
# *not masking*, so a narrow table fails in the unsafe direction for a detector.
_IBAN_LENGTHS = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16, "BG": 22,
    "BH": 22, "BR": 29, "BY": 28, "CH": 21, "CR": 22, "CY": 28, "CZ": 24, "DE": 22,
    "DK": 18, "DO": 28, "EE": 20, "EG": 29, "ES": 24, "FI": 18, "FO": 18, "FR": 27,
    "GB": 22, "GE": 22, "GI": 23, "GL": 18, "GR": 27, "GT": 28, "HR": 21, "HU": 28,
    "IE": 22, "IL": 23, "IQ": 23, "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20,
    "LB": 28, "LC": 32, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MD": 24,
    "ME": 22, "MK": 19, "MR": 27, "MT": 31, "MU": 30, "NL": 18, "NO": 15, "PK": 24,
    "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "SA": 24, "SC": 31,
    "SE": 24, "SI": 19, "SK": 24, "SM": 27, "ST": 25, "SV": 28, "TL": 23, "TN": 24,
    "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20,
}  # fmt: skip


def iban_valid(candidate: str) -> bool:
    """ISO 13616 mod-97 check plus the per-country length table."""
    compact = re.sub(r"[\s-]", "", candidate).upper()
    if len(compact) < 15 or _IBAN_LENGTHS.get(compact[:2]) != len(compact):
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def ssn_valid(candidate: str) -> bool:
    """SSA issuance rules: no 000/666/9xx area, no 00 group, no 0000 serial.

    Presidio also refuses the published sample numbers; deliberately not done here, since
    over-masking a sample costs nothing and sample numbers are what test fixtures carry.
    """
    digits = re.sub(r"\D", "", candidate)
    if len(digits) != 9:
        return False
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in ("000", "666") or area.startswith("9"):
        return False
    if group == "00" or serial == "0000":
        return False
    return len(set(digits)) > 1


def nhs_valid(candidate: str) -> bool:
    """The NHS number's mod-11 check digit."""
    digits = re.sub(r"\D", "", candidate)
    if len(digits) != 10:
        return False
    total = sum(int(d) * (10 - i) for i, d in enumerate(digits[:9]))
    check = 11 - (total % 11)
    if check == 11:
        check = 0
    return check != 10 and check == int(digits[9])


def nino_valid(candidate: str) -> bool:
    """HMRC's prefix rules for a UK National Insurance number."""
    compact = re.sub(r"\s", "", candidate).upper()
    if len(compact) != 9:
        return False
    prefix = compact[:2]
    if prefix in ("BG", "GB", "KN", "NK", "NT", "TN", "ZZ"):
        return False
    if any(ch in "DFIQUV" for ch in prefix) or compact[1] == "O":
        return False
    return compact[-1] in "ABCD"


# ── context windows ─────────────────────────────────────────────────────────


def _has_context(text: str, start: int, end: int, words: re.Pattern[str], window: int = 40) -> bool:
    """Whether a context keyword sits within `window` chars before or after."""
    before = text[max(0, start - window) : start]
    after = text[end : end + window]
    return bool(words.search(before) or words.search(after))


# ── digit and width folding ─────────────────────────────────────────────────
#
# Every pattern is ASCII, so non-ASCII digits evaded them all. Folding is length-preserving, so
# spans still index the original; homoglyphs (Cyrillic а for a) are the documented residual.
def _build_fold() -> dict[int, int]:
    table: dict[int, int] = {}
    for codepoint in range(0x0000, 0x1E950 + 10):
        char = chr(codepoint)
        if codepoint < 128:
            continue
        if char.isdigit():
            try:
                table[codepoint] = ord(str(int(char)))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
    for codepoint in range(0xFF01, 0xFF5F):
        table[codepoint] = codepoint - 0xFEE0
    return table


_FOLD = _build_fold()


def _fold(text: str) -> str:
    """ASCII-fold digits and full-width forms, preserving length and offsets."""
    if text.isascii():
        return text
    return text.translate(_FOLD)


# ── pattern table ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Shape:
    label: str
    category: str
    pattern: re.Pattern[str]
    # Which group holds the value (0 = whole match), so evidence like `DB_PASSWORD=` survives.
    group: int = 0
    # Extra acceptance test on the matched value (checksum, issuance rules).
    validate: Optional[Callable[[str], bool]] = None
    # Keyword gate for ambiguous shapes: the match counts only when one of
    # these words sits within `window` characters of it.
    context: Optional[re.Pattern[str]] = None
    window: int = 40


def _ci(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# Capitalised words after "user"/"customer"/"patient" in SDLC prose that are not surnames. Generous
# on purpose: a missed name is masked at the next boundary; a mangled HLD heading loses the reader.
_NAME_STOPWORDS = frozenset(
    """
    story stories journey journeys interface experience acceptance testing test tests
    guide manual account accounts profile profiles management portal access login
    logout role roles agent agents service services data flow flows table tables model
    models record records support team teams success feedback research segment segments
    base onboarding registration dashboard settings preferences persona personas type
    types group groups id ids name names list lists report reports facing engagement
    retention lifecycle value values details detail information info address addresses
    email emails phone number numbers api app application applications system systems
    corp inc ltd llc gmbh plc co bank group holdings limited company technologies
    systems solutions software labs partners international global services network
    form forms intake churn prediction predictions score scoring segmentation
    satisfaction sentiment analytics insights metrics dashboard summary overview
    request requests response responses ticket tickets case cases queue queues
    portal directory registry catalogue catalog inventory schedule scheduler
    notification notifications alert alerts consent preferences history audit
    """.split()
)


def _not_a_stopword_name(value: str) -> bool:
    return not any(token.lower() in _NAME_STOPWORDS for token in value.split())


# ── internal hostnames ──────────────────────────────────────────────────────
#
# Suffixes meaning "does not resolve on the public internet". Not bare `local`/`lan`/`svc`:
# `.local` is also a filename convention (`.env.local`); `x.svc.cluster.local` is still caught.
_INTERNAL_SUFFIXES = (
    "svc.cluster.local",
    "cluster.local",
    "home.arpa",
    "internal",
    "intranet",
    "localdomain",
    "corp",
)
# A label named exactly one of these marks the whole name internal even under a
# public suffix — `db.internal.example.com` is not a public host.
_INTERNAL_LABELS = frozenset({"internal", "intranet", "corp", "private", "intern"})
# Names that are conventions rather than deployment detail.
_PUBLIC_HOSTS = ("localhost", "example.com", "example.org", "example.net", "invalid", "test")


def _is_internal_host(value: str) -> bool:
    """Whether a dotted name names something inside a private network.

    The validator half of `internal-hostname`, and why its regex is one flat class: the nested
    label-star form backtracked cubically — 353 s on one 8000-char message, holding the turn lock.
    """
    host = value.split(":", 1)[0].rstrip(".").lower()
    if not host or "." not in host:
        return False
    if host.startswith(_PUBLIC_HOSTS) or host.endswith(_PUBLIC_HOSTS):
        return False
    labels = host.split(".")
    if any(not label or len(label) > 63 for label in labels):
        return False
    if any(label in _INTERNAL_LABELS for label in labels):
        return True
    return any(host.endswith(suffix) for suffix in _INTERNAL_SUFFIXES)


# A *reference* to a secret rather than one — template variable, env interpolation, placeholder,
# auth scheme name. Masking `password: ${DB_PASSWORD}` is noise; a literal value stays masked.
_SECRET_REFERENCE = re.compile(
    r"^(?:\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*|%[A-Za-z_][A-Za-z0-9_]*%|\{\{[^}]*\}\}|"
    r"<[^>]*>|\[[^\]]*\]|bearer|basic|digest|negotiate|none|null|true|false|required|optional|"
    r"string|integer|boolean|auto-increment)$",
    re.IGNORECASE,
)


def _is_real_secret_value(value: str) -> bool:
    return not _SECRET_REFERENCE.match(value.strip())


def _identifier_with_a_digit(value: str) -> bool:
    """A record identifier has a digit in it; a schema word does not.

    `patient id: field` and `patient number: auto-increment` are ordinary SDLC output, and
    `health_id` is a withhold tier, so matching them turned an artifact into a refusal.
    """
    return any(char.isdigit() for char in value)


# The shapes, most specific first. Overlaps are resolved after matching (see
# `scan`), so order only matters for which label wins on a tie.
SHAPES: tuple[Shape, ...] = (
    # ── credentials (block tier) ────────────────────────────────────────
    Shape(
        "private-key",
        CREDENTIAL,
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----(?:[A-Za-z0-9+/=\s]|-)*?"
            r"-----END [A-Z0-9 ]*PRIVATE KEY-----|-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"
        ),
    ),
    Shape("databricks-token", CREDENTIAL, re.compile(r"\bdapi[0-9a-f]{32}(?:-\d)?\b")),
    Shape(
        "aws-key-id",
        CREDENTIAL,
        re.compile(r"\b(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}\b"),
    ),
    # The 40-char secret that travels with an AWS key id. Bare 40-char base64 runs are common in
    # SDLC text (git SHAs), so the shape is keyword-gated — the detect-secrets construction.
    Shape(
        "aws-secret-key",
        CREDENTIAL,
        re.compile(r"(?<![A-Za-z0-9/+])[A-Za-z0-9/+]{40}(?![A-Za-z0-9/+=])"),
        context=_ci(r"aws|secret[_ -]?access[_ -]?key|secret[_ -]?key|AKIA|ASIA"),
        window=120,
        validate=lambda v: not re.fullmatch(r"[0-9a-f]{40}", v),  # not a git SHA
    ),
    Shape(
        "github-token",
        CREDENTIAL,
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,255}\b|\bgithub_pat_[A-Za-z0-9_]{20,255}\b"),
    ),
    Shape("slack-token", CREDENTIAL, re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    Shape("google-api-key", CREDENTIAL, re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    Shape("anthropic-key", CREDENTIAL, re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    Shape("openai-key", CREDENTIAL, re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    Shape(
        "azure-client-secret",
        CREDENTIAL,
        re.compile(
            r"(?<![A-Za-z0-9_~.-])[a-zA-Z0-9_~.]{3}\dQ~[a-zA-Z0-9_~.-]{31,34}(?![A-Za-z0-9_~.-])"
        ),
    ),
    Shape(
        "jwt",
        CREDENTIAL,
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    ),
    Shape("bearer-token", CREDENTIAL, re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")),
    # scheme://user:password@host — only the credential part is the value, so
    # the host survives as evidence of *which* system was exposed.
    Shape(
        "url-credentials",
        CREDENTIAL,
        re.compile(r"\b[a-z][a-z0-9+.-]*://([^/\s:@]+:[^@\s]+)@"),
        group=1,
    ),
    # ── secret assignments (mask tier) ──────────────────────────────────
    # KEY = value where the key names a secret; the key survives, the value goes. Bounded flat
    # prefix class, not a star-of-plus: that form backtracked 24.6 s on a 24000-char reply.
    Shape(
        "secret-assignment",
        SECRET_ASSIGNMENT,
        re.compile(
            r"(?i)(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]{0,40}"
            r"(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|"
            r"client[_-]?secret|private[_-]?key|auth)[A-Za-z0-9_-]{0,30}"
            # `['\"]?` before the separator as well as after: in JSON the key's own closing quote
            # sits between them (`"DB_PASSWORD": "…"`). Same fix as the `person` shape below.
            r"['\"]?\s*[:=]\s*['\"]?(?!\[redacted:)([^\s'\";,]{6,})['\"]?"
        ),
        group=1,
        validate=_is_real_secret_value,
    ),
    # A secret handed over in prose ("the token … is invalid") has no `=`/`:` to key on, so it is
    # matched by shape: 16+ token chars carrying a digit or base64 symbol after a secret noun.
    Shape(
        "prose-secret",
        SECRET_ASSIGNMENT,
        re.compile(
            r"(?i)\b(?:token|api[ _-]?key|secret|password|passwd|credential)\b\s+(?:is\s+|was\s+|of\s+)?"
            r"['\"`]?((?=[A-Za-z0-9+/_=-]*[0-9+/=])[A-Za-z0-9+/_-]{16,}={0,2})['\"`]?"
        ),
        group=1,
    ),
    # ── government identifiers (block tier) ─────────────────────────────
    Shape(
        "us-ssn",
        GOVERNMENT_ID,
        re.compile(r"\b\d{3}[- .]\d{2}[- .]\d{4}\b"),
        validate=ssn_valid,
    ),
    # Nine bare digits are a ticket number until "SSN" sits beside them.
    Shape(
        "us-ssn",
        GOVERNMENT_ID,
        re.compile(r"\b\d{9}\b"),
        validate=ssn_valid,
        context=_ci(r"\bssn\b|social\s+security"),
    ),
    Shape(
        "uk-nino",
        GOVERNMENT_ID,
        re.compile(r"\b[A-Za-z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Da-d]\b"),
        validate=nino_valid,
    ),
    Shape(
        "passport-number",
        GOVERNMENT_ID,
        re.compile(r"\b[A-Z0-9]{6,9}\b"),
        context=_ci(r"passport"),
        window=30,
        validate=lambda v: any(ch.isdigit() for ch in v),
    ),
    # ── payment (block tier) ────────────────────────────────────────────
    Shape(
        "card-number",
        PAYMENT,
        re.compile(
            r"\b(?:4\d{3}|5[1-5]\d{2}|2[2-7]\d{2}|3[47]\d{2}|6(?:011|5\d{2})|3(?:0[0-5]|[68]\d)\d)"
            r"[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,4}(?:[ -]?\d{1,3})?\b"
        ),
        validate=luhn_valid,
    ),
    Shape(
        "card-cvv",
        PAYMENT,
        re.compile(
            r"(?i)\b(?:cvv|cvc|cvv2|cvc2|security\s+code|card\s+code)\s*[:=#]?\s*(\d{3,4})\b"
        ),
        group=1,
    ),
    Shape(
        "card-expiry",
        PAYMENT,
        re.compile(
            r"(?i)\b(?:exp(?:iry|iration|ires)?(?:\s+date|\s+month)?)\s*[:=]?\s*((?:0[1-9]|1[0-2])\s?/\s?(?:\d{2}|\d{4}))\b"
        ),
        group=1,
        context=_ci(r"card|pan|visa|mastercard|amex|payment"),
        window=120,
    ),
    # PCI DSS allows last-four *display* for a business function; a reply that volunteers it
    # next to card context is the semantic leak this shape exists for — no full number to match.
    Shape(
        "card-fragment",
        PAYMENT,
        # `expir`/`cvv`/`cardholder` are in the gate because the natural answer repeats no
        # payment noun — the *question* established it. Do NOT widen the `{0,20}` gap to reach
        # more: `_has_context` looks either side of the match, so a wider gap eats the keyword.
        re.compile(
            r"(?i)\b(?:last|first)\s+(?:4|four|6|six)\s+digits?\b[^\n:]{0,20}[:=]?\s*(\d{4,6})\b"
        ),
        group=1,
        context=_ci(r"card|pan|visa|mastercard|amex|payment|account|expir|cvv|cvc|cardholder"),
        window=160,
    ),
    # ── bank (block tier) ───────────────────────────────────────────────
    Shape(
        "iban",
        BANK,
        re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]{4}){2,7}(?:[ -]?[A-Z0-9]{1,4})?\b"),
        validate=iban_valid,
    ),
    # `account` alone is not the keyword: "Given the account: 10203040" is a Gherkin scenario
    # and `bank` is a withhold tier. The keyword must name the *number* — "account number", etc.
    Shape(
        "bank-account",
        BANK,
        re.compile(
            r"(?i)\b(?:(?:bank\s+)?account\s*(?:number|no\.?|#)|acct\.?\s*(?:number|no\.?|#)?|"
            r"routing(?:\s*(?:number|no\.?|#))?|sort\s+code|swift(?:\s*code)?|bic)"
            r"\s*[:=#]?\s*([0-9][0-9 -]{5,20}[0-9])\b"
        ),
        group=1,
        context=_ci(r"\bbank|iban|routing|sort\s+code|swift|payee|payment"),
        window=80,
    ),
    # ── health identifiers (block tier) ─────────────────────────────────
    # The value must carry a digit: without it `patient id: field` and `patient number:
    # auto-increment` matched a withhold-tier shape, refusing ordinary healthcare output.
    Shape(
        "medical-record-number",
        HEALTH_ID,
        re.compile(
            r"(?i)\b(?:mrn|medical\s+record\s+(?:number|no|id|#)|patient\s+(?:id|number|no|#)|health\s+(?:record|card|insurance)\s+(?:number|no|id))\s*[:=#-]?\s*([A-Za-z0-9][A-Za-z0-9-]{3,20})\b"
        ),
        group=1,
        validate=_identifier_with_a_digit,
    ),
    Shape(
        "nhs-number",
        HEALTH_ID,
        re.compile(r"\b\d{3}\s?\d{3}\s?\d{4}\b"),
        validate=nhs_valid,
        context=_ci(r"\bnhs\b"),
    ),
    # A record id sitting inside a patient object — `{patient: 'Tom Reyes',
    # diagnosis: ..., record_id: 445521}` — is a health identifier by
    # association, which is exactly HIPAA's definition of the thing.
    Shape(
        "patient-record-id",
        HEALTH_ID,
        re.compile(
            r"(?i)\b(?:record[_ -]?id|record[_ -]?number|case[_ -]?id)['\"]?\s*[:=]\s*['\"]?([A-Za-z0-9-]{3,20})['\"]?"
        ),
        group=1,
        context=_ci(r"patient|diagnos|clinical|medical"),
        window=160,
        # The same guard `medical-record-number` carries: a record identifier has a digit, a
        # schema word does not — otherwise "patient id: field" captured "field" as a health id.
        validate=_identifier_with_a_digit,
    ),
    # ── health conditions (mask tier, PII switch) ────────────────────────
    Shape(
        "health-condition",
        HEALTH_CONDITION,
        re.compile(
            r"(?i)\b(?:diagnos(?:ed|is)['\"]?\s*(?:with|of|:)?|treatment\s+plan\s+for|treated\s+for|"
            r"suffers?\s+from|condition['\"]?\s*:|tested\s+positive\s+for)\s*['\"]?([^.,;:{}\[\]'\"\n]{3,60}?)['\"]?(?=[.,;:{}\[\]'\"\n]|$)"
        ),
        group=1,
    ),
    Shape(
        "medication",
        HEALTH_CONDITION,
        re.compile(
            r"(?i)\b(?:on|taking|prescribed|medication\s*:?)\s+([A-Z][a-z]{3,}\s+\d+(?:\.\d+)?\s?(?:mg|mcg|ml|g|iu|units?))\b"
        ),
        group=1,
    ),
    # ── people (mask tier, PII switch) ──────────────────────────────────
    # Strong context: a role noun or honorific, optional separator, then two
    # or three capitalised tokens. Case-sensitive on the name part on purpose.
    Shape(
        "person",
        PERSON,
        re.compile(
            r"(?:(?i:\b(?:name|named|full\s+name|patient|customer|client|user|employee|contact|applicant|"
            r"cardholder|account\s+holder|mr|mrs|ms|mx|dr|prof)\.?))"
            # Separator between role word and name. Two optional quotes, not one: the quoted
            # key form `'name': 'John Mercer'` puts a closing quote *before* the colon.
            r"['\"]?\s*[:=-]?\s*['\"]?"
            r"([A-Z][a-z]{1,20}(?:\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]{1,20}){1,2})['\"]?(?![a-z])"
        ),
        group=1,
        validate=_not_a_stopword_name,
    ),
    # ── contact (mask tier, PII switch) ─────────────────────────────────
    # Bounded to RFC 5321 maximums and fenced with a lookbehind: the unbounded `+@` form is
    # the classic backtracking blowup — 1.9 s on one 24000-character reply.
    Shape(
        "email",
        CONTACT,
        re.compile(
            r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}"
            r"(?:\.[A-Za-z0-9-]{1,63}){0,8}\.[A-Za-z]{2,24}(?![A-Za-z0-9-])"
        ),
    ),
    # North American and E.164 shapes; both need separators or a plus sign so
    # a build number or a timestamp never qualifies.
    Shape(
        "phone",
        CONTACT,
        re.compile(r"(?<![\w.-])(?:\+\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\w-])"),
    ),
    Shape(
        "phone",
        CONTACT,
        re.compile(r"(?<![\w.-])\+\d{1,3}[\s.-]?\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?![\w-])"),
    ),
    # ── dates of birth (mask tier, PII switch) ──────────────────────────
    Shape(
        "date-of-birth",
        DOB,
        re.compile(
            r"\b(?:\d{4}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])|"
            r"(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:\d{4}|\d{2})|"
            r"(?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])[-/.](?:\d{4}|\d{2})|"
            r"(?:\d{1,2}\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+(?:\d{1,2},?\s+)?\d{4})\b"
        ),
        context=_ci(r"\bdob\b|date\s+of\s+birth|birth\s*date|\bborn\b|birthday"),
        window=30,
    ),
    # ── network (mask tier, always) ─────────────────────────────────────
    # Public hostnames are left alone — an HLD names api.example.com on purpose. One flat
    # class plus a validator: the nested label regex backtracked cubically (`_is_internal_host`).
    Shape(
        "internal-hostname",
        NETWORK,
        re.compile(
            r"(?<![\w.-])[a-z0-9][a-z0-9.-]{0,252}[a-z0-9](?::\d{2,5})?(?![\w-])",
            re.IGNORECASE,
        ),
        validate=_is_internal_host,
    ),
    Shape(
        "private-ip",
        NETWORK,
        re.compile(
            r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
            r"192\.168\.\d{1,3}\.\d{1,3}|169\.254\.\d{1,3}\.\d{1,3})(?::\d{2,5})?\b"
        ),
    ),
    # Absolute server-side paths reveal deployment layout and user names; relative source
    # paths are left alone. `/etc` and bare `/var` are out: distro-standard paths are public.
    Shape(
        "server-path",
        NETWORK,
        re.compile(
            r"(?<![\w/])(?:/(?:opt|srv|home|root|mnt|data|usr/local|var/(?:lib|www|task))/"
            r"[A-Za-z0-9._\-/]{2,200}"
            r"|[A-Za-z]:\\(?:Users|ProgramData|inetpub)\\[A-Za-z0-9._\-\\ ]{2,200})"
        ),
    ),
)


# ── scanning ────────────────────────────────────────────────────────────────


def scan(text: str, categories: Optional[Iterable[str]] = None) -> list[Finding]:
    """Every sensitive value in `text`, non-overlapping and in order; `categories` narrows it.

    Overlaps are merged, not resolved — one finding over the union at the strongest tier —
    because picking a winner left the loser's overhang in the clear and could under-tier it.
    """
    if not text:
        return []
    wanted = set(categories) if categories is not None else set(CATEGORIES)
    # Matched against the ASCII-folded copy so a non-ASCII digit cannot hide a value. Folding
    # is length-preserving, so every span below indexes the *original* text.
    folded = _fold(text)
    reserved = [match.span() for match in _PLACEHOLDER.finditer(folded)]
    raw: list[Finding] = []
    for shape in SHAPES:
        if shape.category not in wanted:
            continue
        for match in shape.pattern.finditer(folded):
            value = match.group(shape.group)
            if not value:
                continue
            start, end = match.span(shape.group)
            if any(start < stop and begin < end for begin, stop in reserved):
                continue
            if shape.validate is not None and not shape.validate(value):
                continue
            if shape.context is not None and not _has_context(
                folded, match.start(), match.end(), shape.context, shape.window
            ):
                continue
            raw.append(Finding(shape.label, shape.category, start, end, text[start:end]))
    return _merge(raw, text)


def _merge(raw: list[Finding], text: str) -> list[Finding]:
    """Collapse overlapping findings into non-overlapping spans."""
    if not raw:
        return []
    raw.sort(key=lambda f: (f.start, f.end))
    clusters: list[list[Finding]] = [[raw[0]]]
    for finding in raw[1:]:
        if finding.start < max(member.end for member in clusters[-1]):
            clusters[-1].append(finding)
        else:
            clusters.append([finding])

    merged: list[Finding] = []
    for cluster in clusters:
        start = min(member.start for member in cluster)
        end = max(member.end for member in cluster)
        # Strongest tier first, then the longest of that tier: the label a
        # reader sees names the most serious thing in the span.
        primary = min(
            cluster,
            key=lambda f: (_TIER_RANK.get(f.category, 9), -(f.end - f.start), f.start),
        )
        merged.append(
            Finding(
                label=primary.label,
                category=primary.category,
                start=start,
                end=end,
                value=text[start:end],
                labels=tuple(dict.fromkeys(member.label for member in cluster)),
                categories=tuple(dict.fromkeys(member.category for member in cluster)),
            )
        )
    return merged


def mask(text: str, findings: Iterable[Finding]) -> str:
    """Replace every finding with its placeholder, right to left."""
    out = text
    for finding in sorted(findings, key=lambda f: f.start, reverse=True):
        out = out[: finding.start] + finding.placeholder + out[finding.end :]
    return out


def redact(text: str, categories: Optional[Iterable[str]] = None) -> tuple[str, list[Finding]]:
    """Scan and mask, sweeping until nothing new appears (bounded at three passes).

    One pass is not enough: masking *creates word boundaries*, so a permissive neighbour that
    swallowed a strict shape's anchor stops hiding it. Spans index the text of their own pass.
    """
    out = text
    found: list[Finding] = []
    for round_number in range(3):
        # Only the withhold tier is worth re-sweeping: those are the shapes a neighbour can
        # hide, and a full second pass would re-cost every mask for no gain.
        wanted = categories if round_number == 0 else NEVER_ALLOW
        if categories is not None and round_number > 0:
            wanted = [c for c in NEVER_ALLOW if c in set(categories)]
        findings = scan(out, wanted)
        if not findings:
            break
        found.extend(findings)
        out = mask(out, findings)
    return out, found


# ── redaction at a persistence boundary ─────────────────────────────────────
#
# For text about to be written to a sink a human queries later, never the live conversation: a
# worker needs real content, and the output guard — not this — decides mask/withhold/escalate.

_SECRET_TIER = frozenset({CREDENTIAL, SECRET_ASSIGNMENT})


def redact_text(text: str, *, pii: bool = True) -> tuple[str, int]:
    """Redact known secret/PII shapes. Returns (clean text, replacement count).

    `pii=False` applies only the secret tier — credentials and secret assignments — for a
    caller that wants a token masked without deciding personal-data policy for the text.
    """
    if not text:
        return text or "", 0
    cleaned, findings = redact(text, None if pii else _SECRET_TIER)
    return cleaned, len(findings)


def redact_structure(value):
    """Redact every string inside a JSON-shaped structure. Returns (value, count).

    Keys are left alone: they are schema written by the agent, and redacting them would break
    the queries an audit table exists to answer.
    """
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        total = 0
        out = {}
        for key, item in value.items():
            cleaned, count = redact_structure(item)
            out[key] = cleaned
            total += count
        return out, total
    if isinstance(value, (list, tuple)):
        total = 0
        items = []
        for item in value:
            cleaned, count = redact_structure(item)
            items.append(cleaned)
            total += count
        return (items if isinstance(value, list) else tuple(items)), total
    return value, 0
