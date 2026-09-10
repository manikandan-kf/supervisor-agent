"""Sensitive-data detection catalogue — the shapes the guardrail layer recognises.

One place, deliberately, for every pattern the supervisor treats as sensitive:
`redact.py` (the persistence redactor) and `output_guard.py` (the layer-7
output screen) both read this catalogue, so a shape added here is masked in
the audit sinks, in the text relayed to a worker, and in the reply on its way
back — three boundaries, one definition.

Without one place the recognised shapes drift towards whatever was easy to
write, which is two families: known credential token shapes and email
addresses. Names, phone numbers, government identifiers, payment cards, bank
accounts, health identifiers, dates of birth, internal hostnames and server
paths then pass through untouched, and "block" is unreachable because nothing
classifies a finding. This module is the catalogue half of the layer-7 screen;
the tiering half is `output_guard.OutputPolicy`.

Design rules, grounded in what the vendors and the research actually publish:

* **Every finding carries a category.** The category is what the policy acts
  on (mask / block / escalate); the label is what the reader sees in the
  placeholder. Categories mirror the sensitivity tiers Google DLP, AWS Bedrock
  Guardrails and Databricks `detect_sensitive_data` agree on: credentials,
  national identifiers, payment and bank data and health identifiers are the
  high tier; names, contact details, dates of birth and network detail are the
  moderate tier.
* **Checksums and structure over bare regexes.** Card numbers must pass Luhn
  (Presidio `CreditCardRecognizer`), IBANs must pass mod-97 and match their
  country's length, NHS numbers must pass mod-11, US SSNs must not use an
  impossible area/group/serial. A bare digit run is not evidence.
* **Context words gate the ambiguous shapes.** A ten-digit number is a
  timestamp until the word "NHS" sits beside it; a date is a release date until
  "DOB" does. This is Presidio's context-enhancer idea applied deterministically:
  weak shapes require a keyword within a short window, strong shapes do not.
* **Names are anchored, not guessed.** No regex recognises a person's name in
  free text — that is an NER problem, and Presidio's answer is a spaCy model.
  What a regex *can* do with high precision is catch a name introduced by a
  role word ("patient Sarah Kim", "customer: Jane Whitfield", "name John
  Mercer"), which is how names arrive in test fixtures and requirement
  excerpts. Unanchored names are the recorded residual; the upgrade path is
  Presidio's NER recognizer in-container.
* **Precision over recall, still.** Every pattern here is tuned against the
  SDLC prose this system carries — user stories, HLDs, test cases, YAML,
  stack traces — and a false positive costs a reader a placeholder in a
  document they were meant to read. The stoplists and validators exist for
  that reason.

Sources: OWASP LLM02:2025 (sensitive information disclosure — sanitisation,
tokenisation and redaction), Microsoft Presidio predefined recognizers (SSN,
credit card + Luhn, IBAN + mod-97, UK NINO, date, IP, URL), gitleaks and
detect-secrets credential rules, Google Cloud DLP infoType sensitivity levels,
AWS Bedrock Guardrails PII entity list, PCI DSS truncation guidance.
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

# The categories the vendors put in the high tier: values that may never be
# delivered raw whatever the local policy says. `config_store` refuses a
# published policy that sets any of these to `allow`.
NEVER_ALLOW: frozenset[str] = frozenset(
    {CREDENTIAL, SECRET_ASSIGNMENT, GOVERNMENT_ID, PAYMENT, BANK, HEALTH_ID}
)

# The "PII tier" — what OUTPUT_PII_MASKING switches. Everything else is masked
# regardless, because a token, a card number or an internal hostname is a leak
# whatever the artifact, while a stakeholder's name inside a drafted HLD may be
# the content.
PII_TIER: frozenset[str] = frozenset({PERSON, CONTACT, DOB, HEALTH_CONDITION})

# The categories that count towards the bulk-disclosure threshold. A response
# carrying five card numbers or five secret values is a different event from
# one carrying five email addresses: the first is a data dump, the second is a
# distribution list.
#
# The same set as `NEVER_ALLOW`, and deliberately defined as one: "high tier"
# has to mean one thing. They were briefly different — `secret_assignment`
# counted toward a bulk disclosure but could still be published as `allow` —
# which meant the guardrails document's own description of the high tier and
# the code's enforcement of it disagreed.
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

    `label`/`category` are the *strongest* classification in the span; `labels`
    and `categories` list everything that matched inside it. The distinction
    matters because overlapping matches are merged rather than resolved (see
    `scan`): a span that is both a secret assignment and a URL credential is
    one finding, acted on as a credential, recorded as both.
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


#: A placeholder this module already wrote. Nothing inside one is ever a
#: finding: `[redacted:url-credentials]` contains a colon, so the
#: `scheme://user:pass@host` shape matched the placeholder itself on a second
#: pass — inflating the masked count, nesting placeholders, and making the
#: sweep in `redact` non-idempotent. Excluding these spans is what makes
#: re-screening already-masked text a genuine no-op rather than a coincidence
#: of which shapes happen not to match their own output.
_PLACEHOLDER = re.compile(r"\[redacted:[a-z0-9-]+\]")


# ── validators ──────────────────────────────────────────────────────────────


def luhn_valid(digits: str) -> bool:
    """The Luhn checksum every payment card number satisfies (ISO/IEC 7812).

    ASCII-only on purpose: `\\D` would keep an Arabic-Indic or full-width digit
    and `ord(char) - 48` would then compute nonsense from it. Non-ASCII digits
    never reach here anyway — `scan` folds them to ASCII before matching (see
    `_fold`), which is what stops `4242 4242 4242 ٤٢٤٢` from evading the shape
    entirely rather than merely failing the checksum.
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


# Country -> IBAN length, ISO 13616 registry (the entries that matter for an
# SDLC platform's likely traffic; an unknown country is refused rather than
# guessed, which is the fail-closed direction for a *detector*... except that
# refusing means *not masking*, so the table is kept deliberately broad).
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

    Presidio additionally refuses the published sample numbers (123-45-6789
    among them). Deliberately not done here: over-masking a sample costs
    nothing, and sample numbers are exactly what a test fixture carries.
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
# Every pattern here is ASCII, and a detector that only reads ASCII is evaded
# by writing the same value in another script: `4242 4242 4242 ٤٢٤٢` matched no
# payment shape at all, so it was neither blocked nor masked. Folding is
# **length-preserving** — one codepoint in, one codepoint out — which is what
# lets `scan` match against the folded text while every offset, slice and
# replacement still refers to the original. Anything that changed length here
# would silently misplace a mask.
#
# Two families, both confirmed evasions and both single-codepoint:
#   * Unicode decimal digits (Arabic-Indic, Devanagari, full-width, …) → 0-9
#   * Full-width ASCII forms (U+FF01–U+FF5E) → their ASCII equivalents
#
# Not attempted: general homoglyph normalisation (Cyrillic а for Latin a and
# the rest of that long tail). It needs a confusables table, it is not
# length-preserving in every case, and it is the documented residual — the
# gateway's boundary classification is the layer for it.
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
    # Which group holds the sensitive value (0 = the whole match). Lets a
    # pattern keep its evidence — `DB_PASSWORD=` survives, the value goes.
    group: int = 0
    # Extra acceptance test on the matched value (checksum, issuance rules).
    validate: Optional[Callable[[str], bool]] = None
    # Keyword gate for ambiguous shapes: the match counts only when one of
    # these words sits within `window` characters of it.
    context: Optional[re.Pattern[str]] = None
    window: int = 40


def _ci(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


# Words that, capitalised, follow "user"/"customer"/"patient" in ordinary SDLC
# prose and are not surnames. A candidate name containing any of them is not a
# name. Deliberately generous: a missed name is masked at the next boundary,
# a mangled heading in an HLD is a reader who stops trusting the redactor.
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
# The suffixes that mean "this name does not resolve on the public internet".
# Deliberately *not* including bare `local`, `lan` or `svc`: `.local` is both
# mDNS and a widespread filename convention (`.env.local`,
# `docker-compose.local`, `settings.local.json`), and masking those mangles
# ordinary developer prose for no gain. A real internal name such as
# `auth-internal.prod.svc.cluster.local` is still caught by `cluster.local`.
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

    This is the *validator* half of the `internal-hostname` shape, and the
    reason the shape's regex is a single flat character class: the previous
    pattern expressed the label structure with two adjacent, mutually ambiguous
    label stars around an optional middle group, which is a cubic-backtracking
    shape. `"corp."` repeated to the 8000-character input limit took **353
    seconds** inside one node — an uninterruptible denial of service reachable
    from one ordinary-looking message, since the regex is C-level, the deadline
    is only checked *between* calls, and the conversation lock is held
    throughout. Structure that a validator can check in linear time does not
    belong in a regex.
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


# Values that are a *reference* to a secret rather than one: a template
# variable, an environment interpolation, an angle-bracket placeholder, or the
# name of an auth scheme. Masking these is pure noise — `auth: bearer` in an
# OpenAPI security scheme and `password: ${DB_PASSWORD}` in a compose file are
# both documents a reader was meant to read. A *generic-looking but literal*
# value stays masked: `DB_PASSWORD=changeme123` is still a credential, and a
# credential is not made safe by resembling a placeholder.
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

    `patient id: field` and `patient number: auto-increment` are an intake form
    and a column definition — the everyday output of a healthcare SDLC
    assistant, and `health_id` is a *withhold* tier, so matching them turned an
    artifact into a refusal.
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
    # The 40-character secret that travels with an AWS key id. Bare 40-char
    # base64 runs are far too common in SDLC text (git SHAs are 40 hex), so
    # the shape is gated on an AWS/secret keyword nearby — the detect-secrets
    # and AWS Security Blog construction.
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
        re.compile(r"(?<![A-Za-z0-9_~.-])[a-zA-Z0-9_~.]{3}\dQ~[a-zA-Z0-9_~.-]{31,34}(?![A-Za-z0-9_~.-])"),
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
    # KEY = value where the key names a secret. The key survives, the value
    # goes. Admits prefixed keys (`DB_PASSWORD`, `app.secret`). The value must
    # not already be a placeholder, so a second pass is a no-op rather than a
    # double count.
    # `[A-Za-z0-9_.-]{0,40}` rather than `(?:[A-Za-z0-9]+[_.-])*` for the key
    # prefix. The star-of-plus form backtracks per iteration — quadratic, 24.6
    # seconds on a 24000-character worker reply of `a-a-a-…`, which is inside
    # the response size limit. A bounded flat class admits the same prefixed
    # keys (`DB_PASSWORD`, `app.secret`) in linear time.
    Shape(
        "secret-assignment",
        SECRET_ASSIGNMENT,
        re.compile(
            r"(?i)(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]{0,40}"
            r"(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|"
            r"client[_-]?secret|private[_-]?key|auth)[A-Za-z0-9_-]{0,30}"
            r"\s*[:=]\s*['\"]?(?!\[redacted:)([^\s'\";,]{6,})['\"]?"
        ),
        group=1,
        validate=_is_real_secret_value,
    ),
    # A secret handed over in prose — "the token dGVzdC1z... is invalid". No
    # `=` or `:` for the assignment rule to key on, so the value is recognised
    # by shape instead: sixteen or more token characters carrying at least one
    # digit or base64 symbol, introduced by a secret noun. Mask tier because
    # the confidence is lower than a typed credential's.
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
        re.compile(r"(?i)\b(?:cvv|cvc|cvv2|cvc2|security\s+code|card\s+code)\s*[:=#]?\s*(\d{3,4})\b"),
        group=1,
    ),
    Shape(
        "card-expiry",
        PAYMENT,
        re.compile(r"(?i)\b(?:exp(?:iry|iration|ires)?(?:\s+date|\s+month)?)\s*[:=]?\s*((?:0[1-9]|1[0-2])\s?/\s?(?:\d{2}|\d{4}))\b"),
        group=1,
        context=_ci(r"card|pan|visa|mastercard|amex|payment"),
        window=120,
    ),
    # "last 4 digits: 4242" — the partial-extraction pattern. PCI DSS allows
    # last-four *display* where a business function needs it; a response that
    # volunteers it next to card context is the semantic-leakage case this
    # shape exists for, and the point of a fragment is that a full-number
    # regex never sees it.
    Shape(
        "card-fragment",
        PAYMENT,
        re.compile(r"(?i)\b(?:last|first)\s+(?:4|four|6|six)\s+digits?\b[^\n:]{0,20}[:=]?\s*(\d{4,6})\b"),
        group=1,
        context=_ci(r"card|pan|visa|mastercard|amex|payment|account"),
        window=160,
    ),
    # ── bank (block tier) ───────────────────────────────────────────────
    Shape(
        "iban",
        BANK,
        re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]{4}){2,7}(?:[ -]?[A-Z0-9]{1,4})?\b"),
        validate=iban_valid,
    ),
    # `account` alone is not the keyword — "Given the account: 10203040 has a
    # payment pending" is a Gherkin scenario, and `bank` is a *withhold* tier.
    # The keyword has to name the *number*: "account number", "acct", or a
    # routing/sort-code/SWIFT identifier.
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
    # The value must carry a digit. Without that, `patient id: field` (an
    # intake-form requirement) and `patient number: auto-increment` (a column
    # definition) matched a *withhold*-tier shape, so a healthcare assistant's
    # ordinary output came back as a refusal.
    Shape(
        "medical-record-number",
        HEALTH_ID,
        re.compile(r"(?i)\b(?:mrn|medical\s+record\s+(?:number|no|id|#)|patient\s+(?:id|number|no|#)|health\s+(?:record|card|insurance)\s+(?:number|no|id))\s*[:=#-]?\s*([A-Za-z0-9][A-Za-z0-9-]{3,20})\b"),
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
        re.compile(r"(?i)\b(?:record[_ -]?id|record[_ -]?number|case[_ -]?id)\s*[:=]\s*['\"]?([A-Za-z0-9-]{3,20})['\"]?"),
        group=1,
        context=_ci(r"patient|diagnos|clinical|medical"),
        window=160,
    ),
    # ── health conditions (mask tier, PII switch) ────────────────────────
    Shape(
        "health-condition",
        HEALTH_CONDITION,
        re.compile(
            r"(?i)\b(?:diagnos(?:ed|is)\s*(?:with|of|:)?|treatment\s+plan\s+for|treated\s+for|"
            r"suffers?\s+from|condition\s*:|tested\s+positive\s+for)\s*['\"]?([^.,;:{}\[\]'\"\n]{3,60}?)['\"]?(?=[.,;:{}\[\]'\"\n]|$)"
        ),
        group=1,
    ),
    Shape(
        "medication",
        HEALTH_CONDITION,
        re.compile(r"(?i)\b(?:on|taking|prescribed|medication\s*:?)\s+([A-Z][a-z]{3,}\s+\d+(?:\.\d+)?\s?(?:mg|mcg|ml|g|iu|units?))\b"),
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
            # The separator between the role word and the name. `['\"]?` twice
            # rather than once, because the quoted-key form is how a name
            # actually arrives in a fixture — `'name': 'John Mercer'` and
            # `"customer": "Jane Whitfield"` put a closing quote *before* the
            # colon, and a single optional quote never matched it. That was a
            # live leak, because a worker reply carrying a record is a dict
            # literal as often as it is prose.
            r"['\"]?\s*[:=-]?\s*['\"]?"
            r"([A-Z][a-z]{1,20}(?:\s+(?:[A-Z]\.?\s+)?[A-Z][a-z]{1,20}){1,2})['\"]?(?![a-z])"
        ),
        group=1,
        validate=_not_a_stopword_name,
    ),
    # ── contact (mask tier, PII switch) ─────────────────────────────────
    # Bounded to the RFC 5321 maximums (64-char local part, 255-char domain)
    # and fenced with a lookbehind. The unbounded `+@` form is the classic
    # backtracking blowup: `"a."` repeated to a 24000-character reply took 1.9
    # seconds on this one pattern alone.
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
    # Internal hostnames: a label that says so, or a TLD that never resolves
    # publicly. Public hostnames are left alone — an HLD names api.example.com
    # on purpose.
    # One flat character class, then a Python validator. See `_is_internal_host`
    # for why the label structure is not expressed as a regex: the nested form
    # was a cubic-backtracking denial of service reachable from one message.
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
    # Absolute server-side paths reveal deployment layout and user names.
    # Relative source paths (`src/app/main.py`) are a coding agent's daily
    # bread and are left alone.
    # Paths that reveal *this deployment's* layout. `/etc` is deliberately
    # absent: it names distro-standard locations, so `/etc/nginx/nginx.conf` in
    # an ops runbook is public knowledge and masking it only mangles the
    # document. Bare `/var` is out for the same reason; `/var/lib`, `/var/www`
    # and `/var/task` are in, because those are where an application lives.
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
    """Every sensitive value in `text`, non-overlapping, in order of position.

    `categories` narrows the catalogue; None scans everything.

    **Overlaps are merged, not resolved.** Two shapes that claim overlapping
    spans become one finding covering their union, classified by the strongest
    tier present and recording every label that matched. Both halves of that
    matter, and both were bugs:

      * *Merging.* Picking one winner and discarding the other left the loser's
        overhang in the clear. `payment account: 12345678 4242 4242 4242 4242`
        produced a `bank-account` match over the first 18 characters and a
        `card-number` match starting inside it; discarding the card left its
        last eight digits unmasked, in the reply, in the text relayed to the
        worker, and in the Postgres decision trail — with no record that a card
        had been present. Masking the union cannot leak a fragment.
      * *Strongest tier wins.* The old sort compared length before tier, so a
        longer weak match beat a shorter strong one:
        `DB_PASSWORD=postgres://admin:s3cr3t99@db.internal/app` was reported as
        a mask-tier secret assignment rather than a withhold-tier credential,
        which meant the response was *delivered*, the stream was not stopped,
        and the finding never counted toward the bulk threshold.
    """
    if not text:
        return []
    wanted = set(categories) if categories is not None else set(CATEGORIES)
    # Matched against the ASCII-folded copy so a non-ASCII digit cannot hide a
    # value from an ASCII pattern. Folding is length-preserving, so every span
    # below indexes the *original* text and every slice returns what the user
    # or worker actually wrote.
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
            raw.append(
                Finding(shape.label, shape.category, start, end, text[start:end])
            )
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
    """Scan and mask, sweeping until nothing new appears.

    One pass is not enough, because masking *creates word boundaries*. Every
    strict shape is anchored (`\\b`, or a lookbehind) so that a 16-digit run
    inside a git SHA is not read as a card number — and that anchor is also
    what a neighbouring permissive match can deny it. Concretely:

        /var/lib/exports/<183 chars>4242-4242-4242-4242

    the `server-path` shape (a 200-character cap over a class that admits
    digits and hyphens) matches through the first half of the card, and the
    card's own leading `4242` sits against a letter, so `card-number` never
    matched at all — half a payment card was delivered, masked as a file path.
    Replacing the path with `[redacted:server-path]` puts a `]` in front of
    the remaining digits, and the second pass sees the card.

    Bounded at three passes: each one replaces at least one span with a
    placeholder that no shape matches (the `(?!\\[redacted:)` guard on the
    assignment rule, and the fact that a placeholder contains no value shape),
    so the sequence terminates well inside that. The bound is belt and braces
    against a future shape that could match a placeholder.

    Spans in the returned findings refer to the text as it was when each was
    found, so the first pass indexes the original and later passes index the
    partially-masked copy. Callers use the labels, categories and counts —
    `mask` has already been applied here.
    """
    out = text
    found: list[Finding] = []
    for round_number in range(3):
        # Only the withhold tier is worth re-sweeping: those are the shapes a
        # neighbour can hide, and a second pass over the whole catalogue would
        # re-cost every mask for no gain.
        wanted = categories if round_number == 0 else NEVER_ALLOW
        if categories is not None and round_number > 0:
            wanted = [c for c in NEVER_ALLOW if c in set(categories)]
        findings = scan(out, wanted)
        if not findings:
            break
        found.extend(findings)
        out = mask(out, findings)
    return out, found
