"""
Skillcase lead pipeline.

Raw messy leads -> clean/dedupe (rules) -> classify (AI) -> enrich (AI)
-> prioritize (rules) -> outreach (AI) -> QC (rules).

Each stage is a standalone function so it can be tested and explained
independently. The AI-touching functions (classify_leads, enrich_leads,
generate_outreach) are the only ones that call the Gemini API; everything
else is deterministic Python.

Gemini integration notes:
- All AI calls share one helper that uses JSON response mode, a strict
  timeout, exponential backoff with jitter on 429/500/502/503/504, and a
  fallback from `response_json_schema` to plain JSON mode on 400s.
- Automatic function calling (AFC) is explicitly disabled; this pipeline
  never uses function calling, and disabling it keeps the SDK on the
  plain generate path (and silences the SDK's AFC warning).
- Each stage sends its leads in bounded-size batches (with modest
  concurrency) instead of one giant request, which avoids max-token
  truncation and keeps individual calls fast.
- Every stage validates that the model echoed every lead id back; a
  missing id is backfilled with a conservative placeholder record and
  flagged for human review by rule_based_qc, so no lead is ever
  silently dropped.
"""
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterator

from google import genai
from google.genai import errors as genai_errors

try:  # httpx ships with google-genai; transport errors are retryable.
    import httpx
    _TRANSIENT_NETWORK_ERRORS: tuple[type[BaseException], ...] = (httpx.TransportError,)
except ImportError:  # pragma: no cover
    _TRANSIENT_NETWORK_ERRORS = ()

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
REQUIRED_FIELDS = ["phone", "email", "city", "education", "experience_years", "goal", "german_level"]

# --- Gemini request tuning -------------------------------------------------
HTTP_TIMEOUT_MS = int(os.environ.get("GEMINI_TIMEOUT_MS", "120000"))
MAX_ATTEMPTS = 4                      # 1 try + up to 3 retries
_RETRY_BASE_DELAY = float(os.environ.get("GEMINI_RETRY_BASE_DELAY", "1.0"))
_RETRY_MAX_DELAY = 8.0                # seconds, before jitter
_RETRYABLE_CODES = {429, 500, 502, 503, 504}

# Batch sizes: small enough that each response cannot plausibly hit the
# max-token ceiling, large enough that 30 leads = 2-4 calls per stage.
CLASSIFY_BATCH = 15
ENRICH_BATCH = 8
OUTREACH_BATCH = 6
REVIEW_BATCH = 8
MAX_CONCURRENT_REQUESTS = 2           # stay comfortably under free-tier RPM

# Per-batch output budgets, sized to the content (not inflated).
CLASSIFY_MAX_TOKENS = 2000
ENRICH_MAX_TOKENS = 6000
OUTREACH_MAX_TOKENS = 2000
REVIEW_MAX_TOKENS = 3000

_client = None

CLASSIFY_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "lead_id": {"type": "string"},
            "relevant": {"type": "boolean"},
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["lead_id", "relevant", "reason", "confidence"],
    },
}

ENRICH_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "lead_id": {"type": "string"},
            "profile": {"type": "string"},
            "intent": {"type": "string"},
            "intent_level": {
                "type": "string",
                "enum": ["high", "medium", "low", "unclear"],
            },
            "need": {"type": "string"},
            "objection": {"type": "string"},
            "objection_category": {
                "type": "string",
                "enum": [
                    "price",
                    "timeline",
                    "confidence",
                    "eligibility",
                    "qualification",
                    "none",
                ],
            },
            "objection_severity": {
                "type": "string",
                "enum": ["mild", "moderate", "strong", "none"],
            },
            "urgency": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "engagement_level": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "missing_info": {"type": "string"},
            "opportunity": {"type": "string"},
            "next_action": {"type": "string"},
        },
        "required": [
            "lead_id",
            "profile",
            "intent",
            "intent_level",
            "need",
            "objection",
            "objection_category",
            "objection_severity",
            "urgency",
            "engagement_level",
            "missing_info",
            "opportunity",
            "next_action",
        ],
    },
}

OUTREACH_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "lead_id": {"type": "string"},
            "outreach": {"type": "string"},
        },
        "required": ["lead_id", "outreach"],
    },
}

REVIEW_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "lead_id": {"type": "string"},
            "status": {"type": "string", "enum": ["pass", "review"]},
            "confidence": {"type": "number"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "recommended_action": {"type": "string"},
        },
        "required": ["lead_id", "status", "confidence", "issues", "recommended_action"],
    },
}


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your environment or .env file.")
        try:
            _client = genai.Client(
                api_key=api_key,
                http_options={"timeout": HTTP_TIMEOUT_MS},
            )
        except (TypeError, ValueError):  # older SDK without http_options dict support
            _client = genai.Client(api_key=api_key)
    return _client


def _safe_api_error_message(exc: BaseException) -> str:
    msg = getattr(exc, "message", None) or str(exc)
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        msg = msg.replace(key, "[redacted]")
    return msg


# ---------------------------------------------------------------------------
# Stage 1: clean + dedupe (pure rules, no AI)
# ---------------------------------------------------------------------------

def parse_experience_years(raw: str | None) -> float | None:
    if not raw:
        return None
    t = str(raw).lower().strip()
    m = re.search(r"([\d.]+)\s*(year|yr)", t)
    if m:
        return float(m.group(1))
    m = re.search(r"([\d.]+)\s*month", t)
    if m:
        return round(float(m.group(1)) / 12, 2)
    m = re.match(r"^([\d.]+)$", t)
    if m:
        return float(m.group(1))
    return None


def normalize_education(raw: str | None) -> str | None:
    if not raw:
        return None
    key = re.sub(r"\s+", " ", str(raw).lower().replace(".", "")).strip()
    mapping = {
        "bsc nursing": "BSc Nursing",
        "gnm": "GNM",
        "bpharm": "BPharm",
        "bba": "BBA",
        "engineer": "Engineer",
    }
    return mapping.get(key, str(raw).strip())


def title_case(raw: str) -> str:
    return " ".join(w.capitalize() for w in str(raw).strip().lower().split())


def normalize_goal(raw: str | None) -> str | None:
    if not raw:
        return None
    t = str(raw).strip().lower()
    if "canada" in t:
        return "Canada"
    if "uk" in t:
        return "UK"
    if "germany" in t:
        return "Work in Germany"
    if "explore" in t:
        return "Explore options"
    if "preparation" in t:
        return "B2 preparation"
    if "abroad" in t:
        return "Work abroad (unspecified)"
    return title_case(str(raw))


def _completeness(lead: dict) -> int:
    return sum(1 for f in REQUIRED_FIELDS if lead.get(f) not in (None, ""))


def clean_and_dedupe(raw_leads: list[dict]) -> dict:
    cleaned = []
    for i, r in enumerate(raw_leads):
        # lead_id is the record's identity; synthesize one if a messy source
        # row omits it instead of crashing.
        lead_id = r.get("lead_id") or f"RAW-{i + 1:03d}"
        cleaned.append({
            "lead_id": lead_id,
            "name": title_case(r.get("name") or ""),
            "phone": str(r.get("phone") or "").replace(" ", ""),
            "email": (r.get("email") or "").strip().lower() or None,
            "city": (r.get("city") or "").strip() or None,
            "education": normalize_education(r.get("education")),
            "experience_raw": r.get("experience"),
            "experience_years": parse_experience_years(r.get("experience")),
            "goal": normalize_goal(r.get("goal")),
            "german_level": str(r.get("german_level") or "").strip().upper() or None,
            "source": (r.get("source") or "").strip(),
            "last_contacted": r.get("last_contacted"),
            "conversation": r.get("conversation"),
            "notes": r.get("notes"),
        })

    groups: dict[str, list[dict]] = {}
    for c in cleaned:
        # Only real phone numbers group records together; leads with no
        # phone at all are never merged with each other by accident.
        group_key = c["phone"] if c["phone"] else f"__no_phone__{c['lead_id']}"
        groups.setdefault(group_key, []).append(c)

    lead_audit = []
    duplicate_groups = []
    primaries = []
    exact_fields = ["name", "email", "city", "education", "goal", "german_level", "conversation"]

    for group in groups.values():
        if len(group) == 1:
            primaries.append(group[0])
            lead_audit.append({
                "lead_id": group[0]["lead_id"],
                "status": "retained",
                "primary_id": group[0]["lead_id"],
                "duplicate_relationship": "primary",
                "reason": "Retained as unique primary lead record",
                "requires_human_review": False,
            })
            continue
        sorted_group = sorted(group, key=lambda c: (-_completeness(c), c["lead_id"]))
        primary, others = sorted_group[0], sorted_group[1:]
        exact, fuzzy = [], []
        for o in others:
            if all((o.get(f) or "") == (primary.get(f) or "") for f in exact_fields):
                exact.append(o["lead_id"])
            else:
                fuzzy.append(o["lead_id"])
        primary["duplicate_count"] = len(others)
        primary["_dup_exact"] = exact
        primary["_dup_fuzzy"] = fuzzy
        duplicate_groups.append({
            "primary": primary["lead_id"], "exact": exact, "fuzzy": fuzzy,
            "all": [g["lead_id"] for g in group],
        })
        primaries.append(primary)

        # Audit primary lead
        lead_audit.append({
            "lead_id": primary["lead_id"],
            "status": "retained",
            "primary_id": primary["lead_id"],
            "duplicate_relationship": "primary",
            "reason": f"Retained as primary lead record for duplicate cluster ({', '.join(g['lead_id'] for g in group)})",
            "requires_human_review": len(fuzzy) > 0,
        })

        # Audit other duplicates in the group
        for o in others:
            lid = o["lead_id"]
            if lid in exact:
                lead_audit.append({
                    "lead_id": lid,
                    "status": "merged",
                    "primary_id": primary["lead_id"],
                    "duplicate_relationship": "exact_duplicate",
                    "reason": f"Exact duplicate of {primary['lead_id']} (auto-merged)",
                    "requires_human_review": False,
                })
            else:
                diff_fields = [f for f in exact_fields if (o.get(f) or "") != (primary.get(f) or "")]
                diff_text = f" differing in: {', '.join(diff_fields)}" if diff_fields else ""
                lead_audit.append({
                    "lead_id": lid,
                    "status": "flagged",
                    "primary_id": primary["lead_id"],
                    "duplicate_relationship": "conflicting_duplicate",
                    "reason": f"Conflicting duplicate of {primary['lead_id']}{diff_text} (excluded from active batch pending human reconciliation)",
                    "requires_human_review": True,
                })

    for p in primaries:
        p.setdefault("duplicate_count", 0)
        p.setdefault("_dup_exact", [])
        p.setdefault("_dup_fuzzy", [])
        p["missing_fields"] = [f for f in REQUIRED_FIELDS if not p.get(f)]
        p["email_valid"] = bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", p["email"])) if p.get("email") else None
        p["phone_valid"] = bool(re.match(r"^\+91\d{10}$", p["phone"]))

    primaries.sort(key=lambda p: p["lead_id"])
    lead_audit.sort(key=lambda a: a["lead_id"])
    return {"primaries": primaries, "duplicate_groups": duplicate_groups, "lead_audit": lead_audit}


# ---------------------------------------------------------------------------
# Helpers for talking to Gemini
# ---------------------------------------------------------------------------

def _batches(seq: list, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield list(seq[i:i + size])


def _run_batched(
    batches: list[list],
    build_prompt: Callable[[list], str],
    merge: Callable[[list, list], list],
    max_tokens: int,
    schema: dict | None,
    max_workers: int = MAX_CONCURRENT_REQUESTS,
) -> list[list]:
    """Run one Gemini call per batch, concurrently, and merge the results.

    `merge(batch, parsed_items)` must return a list of validated records
    covering every lead in the batch (backfilling placeholders for ids the
    model failed to echo), so no lead is ever silently dropped.
    """
    if not batches:
        return []

    if len(batches) == 1:
        return [merge(batches[0], _ask_for_json_array_with_retry(
            build_prompt(batches[0]), max_tokens=max_tokens, schema=schema))]

    results: list[list | None] = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as ex:
        futures = {
            ex.submit(
                _ask_for_json_array_with_retry,
                build_prompt(batch),
                max_tokens,
                schema,
            ): i
            for i, batch in enumerate(batches)
        }
        first_error: BaseException | None = None
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                parsed = fut.result()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                if first_error is None:
                    first_error = exc
                continue
            results[i] = merge(batches[i], parsed)

    if first_error is not None:
        raise first_error
    return [r for r in results if r is not None]


def _parse_json_array(text: str) -> list[dict]:
    if not text or not str(text).strip():
        raise ValueError("Empty response from model.")
    stripped = str(text).strip()
    # Strip markdown code fences if the model added them despite instructions.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.S)
    if fence:
        stripped = fence.group(1).strip()
    start, end = stripped.find("["), stripped.rfind("]")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(stripped[start:end + 1])
        if isinstance(parsed, list):
            return parsed
        raise ValueError("Model response JSON was not an array.")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("No JSON array found in model response.") from exc
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                return value
    raise ValueError("No JSON array found in model response.")


def _salvage_truncated_array(text: str) -> list[dict] | None:
    """Best-effort recovery of a JSON array cut off by max_output_tokens.

    Cuts at the last complete object boundary and closes the array, so
    already-finished records are kept instead of discarding the whole
    (slow, billed) response.
    """
    if not text:
        return None
    start = text.find("[")
    if start == -1:
        return None
    candidate = text[start:]
    last_obj = candidate.rfind("}")
    if last_obj == -1:
        return None
    try:
        repaired = json.loads(candidate[:last_obj + 1] + "]")
    except json.JSONDecodeError:
        return None
    return repaired if isinstance(repaired, list) else None


def _retry_delay_from(exc: BaseException) -> float | None:
    """Extract a server-suggested retry delay (seconds) from a 429 body."""
    detail = getattr(exc, "message", None) or str(exc)
    m = re.search(
        r"retry(?:[-_ ]?delay)?[\"':=\s]+(\d+(?:\.\d+)?)\s*(ms|s|seconds?)?",
        detail,
        re.I,
    )
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    if m.group(2) and m.group(2).lower() == "ms":
        value /= 1000.0
    return max(0.5, min(value, 20.0))


def _backoff_delay(attempt: int, exc: BaseException | None) -> float:
    delay = min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY)
    delay += random.uniform(0, 0.4)  # jitter so concurrent batches don't sync up
    if exc is not None:
        suggested = _retry_delay_from(exc)
        if suggested is not None:
            delay = max(delay, suggested)
    return delay


def _generate_config_kwargs(max_tokens: int, schema: dict | None) -> list[dict]:
    """Candidate config kwargs, tried in order.

    `response_json_schema` is the modern structured-output field; if the
    backend model rejects it with a 400 we fall back to plain JSON mode
    rather than failing the whole run.
    """
    base = {
        "response_mime_type": "application/json",
        "max_output_tokens": max_tokens,
        "automatic_function_calling": {"disable": True},
    }
    if schema is not None:
        return [dict(base, response_json_schema=schema), dict(base)]
    return [dict(base)]


def _ask_for_json_array(
    prompt: str, max_tokens: int = 4000, schema: dict | None = None
) -> list[dict]:
    """Single Gemini request -> parsed JSON array of dicts."""
    client = get_client()
    kwargs_options = _generate_config_kwargs(max_tokens, schema)

    last_error: BaseException | None = None
    for kwargs in kwargs_options:
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=kwargs,
            )
            parsed = getattr(resp, "parsed", None)
            if isinstance(parsed, list) and parsed:
                return parsed
            if isinstance(parsed, dict):
                for value in parsed.values():
                    if isinstance(value, list):
                        return value
            text = getattr(resp, "text", None) or ""
            try:
                return _parse_json_array(text)
            except ValueError:
                salvage = _salvage_truncated_array(text)
                if salvage:
                    return salvage
                raise
        except genai_errors.APIError as exc:
            last_error = exc
            code = getattr(exc, "code", None)
            if code == 400 and len(kwargs_options) > 1:
                continue  # retry with next fallback config (e.g. no schema)
            raise RuntimeError(
                f"Gemini API error ({code}): {_safe_api_error_message(exc)}"
            ) from None
        except Exception as exc:
            raise RuntimeError(
                f"Gemini request failed: {_safe_api_error_message(exc)}"
            ) from None
    raise RuntimeError(
        f"Gemini API error: {_safe_api_error_message(last_error)}"
    ) from None


def _ask_for_json_array_with_retry(
    prompt: str, max_tokens: int = 4000, schema: dict | None = None
) -> list[dict]:
    """Retry transient Gemini failures (429/5xx, timeouts, malformed JSON)
    with exponential backoff + jitter; respect server retry hints."""
    last_error: BaseException | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return _ask_for_json_array(prompt, max_tokens=max_tokens, schema=schema)
        except RuntimeError as exc:
            last_error = exc
            message = str(exc)
            code_match = re.search(r"Gemini API error \((\d+)\)", message)
            code = int(code_match.group(1)) if code_match else None
            transient_api = code in _RETRYABLE_CODES
            transient_network = any(
                type(e).__name__ in message for e in _TRANSIENT_NETWORK_ERRORS
            ) or "timed out" in message.lower()
            malformed_output = "No JSON array found" in message or "Empty response" in message
            if not (transient_api or transient_network or malformed_output):
                raise
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(_backoff_delay(attempt, last_error))
    raise last_error  # type: ignore[misc]


def _match_records_to_leads(
    batch: list[dict],
    parsed: list,
    fallback: Callable[[dict], dict],
    get_id: Callable[[dict], Any] = lambda item: item.get("lead_id"),
) -> list[dict]:
    """Validate AI records against the batch's lead ids.

    - keeps only records whose echoed id belongs to this batch (defends
      against the model echoing an id from another batch or hallucinating
      one)
    - backfills a conservative placeholder (falling back to the batch's own
      data, never invented facts) for any id the model failed to return,
      so downstream stages never hit a KeyError and QC can flag the gap.
    """
    lead_ids = {str(get_id(l)) for l in batch}
    by_id: dict[str, dict] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        lid = str(item.get("lead_id") or "")
        if lid and lid in lead_ids:
            by_id[lid] = item
    out = []
    for lead in batch:
        lid = str(get_id(lead))
        if lid in by_id:
            out.append(by_id[lid])
        else:
            out.append(fallback(lead))
    return out


def _classify_placeholder(lead: dict) -> dict:
    return {
        "lead_id": lead.get("lead_id"),
        "relevant": False,
        "reason": "Classification missing from AI response -- treated as not relevant and routed to human review.",
        "confidence": 0.0,
    }


def _enrich_placeholder(lead: dict) -> dict:
    return {
        "lead_id": lead.get("lead_id"),
        "profile": "Not stated",
        "intent": "Not stated",
        "intent_level": "unclear",
        "need": "Not stated",
        "objection": "Not stated",
        "objection_category": "none",
        "objection_severity": "none",
        "urgency": "low",
        "engagement_level": "low",
        "missing_info": "Enrichment missing from AI response",
        "opportunity": "Not stated",
        "next_action": "Review this lead manually -- enrichment was not returned.",
    }


def _outreach_placeholder(lead: dict) -> dict:
    return {
        "lead_id": lead.get("lead_id"),
        "outreach": "",
    }


# ---------------------------------------------------------------------------
# Stage 2: classify (AI)
# ---------------------------------------------------------------------------

_CLASSIFICATION_PROMPT = """You are triaging inbound leads for Skillcase, a company that prepares nurses and allied healthcare workers in India to learn German and get placed as nurses in Germany.

Relevance criteria:
- RELEVANT: the person has (or is credibly pursuing) a nursing/allied-healthcare background AND is interested in working or preparing for work in Germany specifically, at any stage (even early exploration, even if only asking preparatory questions).
- NOT RELEVANT: the person is targeting a different country only (e.g. Canada, UK), is in an unrelated profession with no stated intent to pivot into healthcare, or is asking on behalf of a background that does not fit.
- Missing fields lower confidence but do not automatically make a lead not relevant.

For each lead below, return a relevance judgment. Respond with ONLY a JSON array, no prose, no markdown fences, one object per lead, in this exact shape:
[{{"lead_id":"L001","relevant":true,"reason":"short reason under 15 words","confidence":0.0}}]

Leads:
{payload}"""


def classify_leads(leads: list[dict]) -> dict[str, dict]:
    batches = list(_batches(leads, CLASSIFY_BATCH))

    def build_prompt(batch: list[dict]) -> str:
        payload = [{
            "lead_id": l.get("lead_id"),
            "education": l.get("education"),
            "experience_years": l.get("experience_years"),
            "german_level": l.get("german_level"),
            "profession": l.get("profession"),
            "source": l.get("source"),
            "conversation": l.get("conversation"),
        } for l in batch]
        return _CLASSIFICATION_PROMPT.format(payload=json.dumps(payload, ensure_ascii=False))

    def merge(batch: list[dict], parsed: list) -> list[dict]:
        return _match_records_to_leads(batch, parsed, _classify_placeholder)

    results = _run_batched(
        batches, build_prompt, merge,
        max_tokens=CLASSIFY_MAX_TOKENS, schema=CLASSIFY_SCHEMA,
    )
    out: dict[str, dict] = {}
    for batch_results in results:
        for r in batch_results:
            lid = r.get("lead_id")
            if lid:
                out[lid] = r
    return out


# ---------------------------------------------------------------------------
# Stage 3: enrich (AI)
# ---------------------------------------------------------------------------

_ENRICH_PROMPT = """For each Skillcase lead below, extract structured sales context from the conversation.

Be concise and evidence-based. Use ONLY information supported by the lead's
original data and conversation. If information genuinely isn't present, say
"Not stated" rather than guessing.

Fields per lead:

- profile: one short line describing their background, education,
  experience and location.

- intent: a short description of what the person appears to be trying to
  achieve.

- intent_level: classify their level of intent as high | medium | low | unclear.
    * high: explicitly wants to pursue the opportunity, asks to start,
      requests a call/next step, or shows clear decision intent.
    * medium: actively interested and gathering information but has not
      indicated readiness to act.
    * low: early exploration, curiosity, or general information-seeking.
    * unclear: there is not enough evidence to determine their intent.

- need: what they appear to need help with next.

- objection: their stated hesitation, concern, or barrier. If none is stated,
  return "Not stated".

- objection_category: one of price | timeline | confidence | eligibility |
  qualification | none.

- objection_severity: one of mild | moderate | strong | none.
    * mild: passing concern or light question.
    * moderate: real concern or hesitation but not explicitly blocking.
    * strong: explicitly stated as a blocker or barrier.
    * none: no objection stated.

- urgency: classify urgency as high | medium | low based on the holistic
  meaning of the conversation, NOT keyword matching.
    * high: concrete readiness, such as requesting a call, giving a clear
      timeline, or stating they are ready to start.
    * medium: interested but no immediate action planned.
    * low: early-stage exploration or passive inquiry.

- engagement_level: classify the lead's level of meaningful engagement as
  high | medium | low.
    * high: asks specific questions, provides meaningful personal context,
      requests a call/next step, or actively discusses their situation.
    * medium: provides some context or asks a relevant question but shows
      limited interaction.
    * low: very short, vague, passive, or purely exploratory interaction.

- missing_info: the single most important piece of information a salesperson
  should still collect. If nothing important is missing, say "None".

- opportunity: what Skillcase could concretely help this person with, based
  only on what is supported by the conversation.

- next_action: the one recommended next step for the salesperson.

IMPORTANT:
- Do not infer intent from education, experience, German level, or profession
  alone.
- Do not assume someone is highly interested simply because they appear to
  be a good fit.
- Do not assume urgency without evidence from the conversation.
- Do not assume high engagement from a single vague message.
- Keep intent, urgency, fit-related information, and engagement conceptually
  separate.
- The original conversation is the primary source of truth.

Respond with ONLY a JSON array, no prose, no markdown fences, one object per
lead, containing every field above for every lead in the input.

Leads:
{payload}
"""


def enrich_leads(leads: list[dict]) -> dict[str, dict]:
    """Enrich the given leads (callers pass relevant leads only)."""
    if not leads:
        return {}
    batches = list(_batches(leads, ENRICH_BATCH))

    def build_prompt(batch: list[dict]) -> str:
        payload = [{
            "lead_id": l.get("lead_id"), "name": l.get("name"), "city": l.get("city"),
            "education": l.get("education"),
            "experience_years": l.get("experience_years"), "goal": l.get("goal"),
            "german_level": l.get("german_level"),
            "source": l.get("source"), "conversation": l.get("conversation"),
            "missing_fields": l.get("missing_fields", []),
        } for l in batch]
        return _ENRICH_PROMPT.format(payload=json.dumps(payload, ensure_ascii=False))

    def merge(batch: list[dict], parsed: list) -> list[dict]:
        return _match_records_to_leads(batch, parsed, _enrich_placeholder)

    results = _run_batched(
        batches, build_prompt, merge,
        max_tokens=ENRICH_MAX_TOKENS, schema=ENRICH_SCHEMA,
    )
    out: dict[str, dict] = {}
    for batch_results in results:
        for r in batch_results:
            lid = r.get("lead_id")
            if lid:
                out[lid] = r
    return out


# ---------------------------------------------------------------------------
# Stage 4: prioritize (pure rules, no AI)
# ---------------------------------------------------------------------------

INTENT_SCORE = {
    "high": 100,
    "medium": 60,
    "low": 25,
    "unclear": 0,
}

URGENCY_SCORE = {
    "high": 100,
    "medium": 60,
    "low": 20,
}

ENGAGEMENT_SCORE = {
    "high": 100,
    "medium": 60,
    "low": 20,
}

GERMAN_SCORE = {
    "A1": 20,
    "A2": 40,
    "B1": 60,
    "B2": 80,
    "C1": 100,
    "C2": 100,
}


def compute_fit_score(lead: dict) -> float:
    """
    Calculate fit from deterministic lead attributes.

    Fit is currently based on:
    - German proficiency: 50%
    - Relevant experience: 50%

    Missing information is treated as unknown rather than automatically
    penalizing the lead.
    """

    german_level = str(lead.get("german_level") or "").upper()
    german_score = GERMAN_SCORE.get(german_level, 0)

    experience = lead.get("experience_years")

    try:
        experience = float(experience) if experience is not None else 0
    except (TypeError, ValueError):
        experience = 0

    if experience >= 5:
        experience_score = 100
    elif experience >= 3:
        experience_score = 80
    elif experience >= 2:
        experience_score = 60
    elif experience >= 1:
        experience_score = 40
    elif experience > 0:
        experience_score = 20
    else:
        experience_score = 0

    return round((german_score + experience_score) / 2, 1)


def compute_priority(lead: dict, classification: dict, enrichment: dict) -> dict:
    """
    Calculate transparent sales-priority score.

    Priority:
        Intent      = 40%
        Urgency     = 25%
        Fit         = 20%
        Engagement  = 15%

    Final score is 0-100.

    This is a sales follow-up heuristic, not a conversion prediction.
    """

    if not classification.get("relevant"):
        return {
            "score": None,
            "band": "N/A",
            "priority_reason": (
                "Not scored because the lead is not relevant "
                "to Skillcase's healthcare program."
            ),
        }

    # ---------------------------------------------------------
    # 1. INTENT — 40%
    # ---------------------------------------------------------

    intent = str(
        enrichment.get("intent_level") or "unclear"
    ).lower().strip()

    if intent not in INTENT_SCORE:
        intent = "unclear"

    intent_score = INTENT_SCORE[intent]

    # ---------------------------------------------------------
    # 2. URGENCY — 25%
    # ---------------------------------------------------------

    urgency = str(
        enrichment.get("urgency") or "low"
    ).lower().strip()

    if urgency not in URGENCY_SCORE:
        urgency = "low"

    urgency_score = URGENCY_SCORE[urgency]

    # ---------------------------------------------------------
    # 3. FIT — 20%
    # ---------------------------------------------------------

    fit_score = compute_fit_score(lead)

    # ---------------------------------------------------------
    # 4. ENGAGEMENT — 15%
    # ---------------------------------------------------------

    engagement = str(
        enrichment.get("engagement_level") or "low"
    ).lower().strip()

    if engagement not in ENGAGEMENT_SCORE:
        engagement = "low"

    engagement_score = ENGAGEMENT_SCORE[engagement]

    # ---------------------------------------------------------
    # FINAL SCORE
    # ---------------------------------------------------------

    score = round(
        (intent_score * 0.40)
        + (urgency_score * 0.25)
        + (fit_score * 0.20)
        + (engagement_score * 0.15),
        1,
    )

    # ---------------------------------------------------------
    # PRIORITY BAND
    # ---------------------------------------------------------

    if score >= 80:
        band = "High"
    elif score >= 50:
        band = "Medium"
    else:
        band = "Low"

    # ---------------------------------------------------------
    # HUMAN-READABLE REASON
    # ---------------------------------------------------------

    priority_reason = (
        f"Intent: {intent.title()} ({intent_score}/100); "
        f"Urgency: {urgency.title()} ({urgency_score}/100); "
        f"Fit: {fit_score}/100; "
        f"Engagement: {engagement.title()} ({engagement_score}/100)."
    )

    return {
        "score": score,
        "band": band,
        "priority_reason": priority_reason,
        "components": {
            "intent": intent_score,
            "urgency": urgency_score,
            "fit": fit_score,
            "engagement": engagement_score,
        },
    }


# ---------------------------------------------------------------------------
# Stage 5: outreach (AI)
# ---------------------------------------------------------------------------

_OUTREACH_PROMPT = """
You are writing a real sales follow-up for Skillcase.

Skillcase helps nurses and allied healthcare professionals in India prepare
for German language requirements and the process of pursuing healthcare
opportunities in Germany.

Your job is NOT to invent a sales pitch.

Your job is to respond naturally to what the lead actually said.

SOURCE OF TRUTH
---------------
The lead's original conversation is the primary source of truth.

The structured enrichment is only supporting context. It may contain
interpretations, so NEVER treat an enrichment field as a fact if the
original conversation does not support it.

STRICT GROUNDING RULES
----------------------
1. Read the original conversation carefully before writing.
2. Identify the specific thing the person is asking about, considering,
   worried about, or trying to achieve.
3. Respond to THAT specific situation.
4. If they raised a concern, address that concern directly.
5. If they asked a question, respond to that question rather than changing
   the subject into a generic sales pitch.
6. Do not invent facts about the person's career, qualifications,
   experience, timeline, budget, German level, family situation, or goals.
7. Do not invent Skillcase features, prices, guarantees, outcomes,
   eligibility decisions, placement statistics, or timelines.
8. Do not say that Skillcase can guarantee a job, placement, visa,
   salary, or any other outcome.
9. Do not mention information that exists only in the enrichment if it is
   not supported by the original lead data.
10. Do not use generic filler such as:
    - "I noticed you're interested..."
    - "We'd love to help you on your journey..."
    - "Take the next step toward your dreams..."
    unless the actual conversation makes that wording genuinely relevant.
11. Do not simply repeat the lead's message back to them.
12. Ask at most ONE useful next-step question.
13. Keep the message conversational and appropriate for the channel.
14. Use the person's name naturally if appropriate.
15. Keep it under 70 words.
16. Do not use emojis unless the lead's own tone clearly supports them.
17. Do not use corporate language or marketing jargon.

MESSAGE LOGIC
-------------
Use this sequence:

A. What did the lead actually say?
B. What do they appear to need right now?
C. Is there a specific concern/objection?
D. Respond directly to that situation.
E. If appropriate, suggest ONE concrete next step.

IMPORTANT:
A lead who is only exploring should receive an exploratory response.
A lead asking about cost should receive a response focused on cost.
A lead asking about eligibility should receive a response focused on
eligibility.
A lead asking about a call should receive a response that moves toward
that call.
A lead expressing concern about unrealistic promises should receive a
transparent response and should NOT receive another promise.

Return ONLY a JSON array in this exact format, one object per lead:

[
  {{
    "lead_id": "L001",
    "outreach": "message text"
  }}
]

Leads:
{payload}
"""


def generate_outreach(items: list[dict]) -> dict[str, str]:
    """Generate grounded, situation-specific outreach from the original lead conversation."""
    if not items:
        return {}
    batches = list(_batches(items, OUTREACH_BATCH))

    def build_payload(i: dict) -> dict:
        lead = i.get("lead") or {}
        enrichment = i.get("enrichment") or {}
        return {
            "lead_id": lead.get("lead_id"),
            "name": lead.get("name"),
            "city": lead.get("city"),
            "education": lead.get("education"),
            "experience_years": lead.get("experience_years"),
            "goal": lead.get("goal"),
            "german_level": lead.get("german_level"),
            "source": lead.get("source"),
            "conversation": lead.get("conversation"),
            "notes": lead.get("notes"),
            # AI enrichment is supporting context, NOT the source of truth.
            "enrichment": {
                "profile": enrichment.get("profile"),
                "intent": enrichment.get("intent"),
                "need": enrichment.get("need"),
                "objection": enrichment.get("objection"),
                "objection_category": enrichment.get("objection_category"),
                "objection_severity": enrichment.get("objection_severity"),
                "urgency": enrichment.get("urgency"),
                "missing_info": enrichment.get("missing_info"),
                "opportunity": enrichment.get("opportunity"),
                "next_action": enrichment.get("next_action"),
            },
        }

    def build_prompt(batch: list[dict]) -> str:
        payload = [build_payload(i) for i in batch]
        return _OUTREACH_PROMPT.format(payload=json.dumps(payload, ensure_ascii=False))

    def merge(batch: list[dict], parsed: list) -> list[dict]:
        # Batch items are {"lead": ..., "enrichment": ...} wrappers, so both
        # the id lookup and the placeholder fallback must unwrap the lead.
        unwrap = lambda item: (item.get("lead") or {}).get("lead_id")  # noqa: E731
        return _match_records_to_leads(
            batch, parsed,
            fallback=lambda item: _outreach_placeholder(item.get("lead") or {}),
            get_id=unwrap,
        )

    results = _run_batched(
        batches, build_prompt, merge,
        max_tokens=OUTREACH_MAX_TOKENS, schema=OUTREACH_SCHEMA,
    )
    out: dict[str, str] = {}
    for batch_results in results:
        for r in batch_results:
            lid = r.get("lead_id")
            if lid:
                out[lid] = r.get("outreach") or ""
    return out


# ---------------------------------------------------------------------------
# Stage 7: AI review (semantic second-pass; identifies problems, never fixes)
# ---------------------------------------------------------------------------

_AI_REVIEW_PROMPT = """You are a meticulous second-pass reviewer for Skillcase, a company that prepares nurses and allied healthcare workers in India to learn German and get placed as nurses in Germany.

A first pipeline already produced, for each lead: a relevance classification, a structured enrichment, a rule-based priority score and a personalized outreach draft. Rule-based quality checks (exact word matching) have already run. Your job is the SEMANTIC pass those rules cannot do: does the generated lead intelligence actually make sense based on the ORIGINAL lead information?

The original lead data and conversation below are ALWAYS the source of truth.

For each lead, check:

1. GROUNDING — Are the enrichment's profile, intent, need and objection actually supported by what the lead said? Flag invented facts and unsupported assumptions. Do not credit claims inferred merely from education, profession, experience or German level; those are background, not evidence of intent or need.

2. CLASSIFICATION — Does the Relevant / Not Relevant verdict make sense given the original lead information? Flag obvious contradictions only.

3. ENRICHMENT — Is the stated intent grounded in the conversation? Is the need grounded? Is the objection grounded? Are urgency and engagement reasonable for what the lead actually said and did? Does the next action make sense? Do not infer facts from background fields alone.

4. PRIORITY — The score is computed by a fixed deterministic formula (intent 40% + urgency 25% + fit 20% + engagement 15%). Do NOT recalculate or second-guess the arithmetic. Only flag obvious semantic inconsistencies, e.g. the lead clearly said one thing that contradicts the inputs the score relies on.

5. OUTREACH — Does the draft genuinely reflect this specific lead's situation, address their concern where appropriate, and avoid introducing information the lead never gave? Flag drafts that invent features, prices, eligibility, timelines or outcomes; promise jobs, placement, visas, salaries or guaranteed outcomes; or ignore an important concern the lead actually raised.

6. EXISTING RULE-BASED QC FLAGS — Treat them as context only. A rule flag alone is NOT automatically a problem: assess whether it is meaningful for this lead.

Be appropriately strict, not hair-trigger: PASS leads whose intelligence is grounded and internally consistent; REVIEW leads with real, specific, actionable problems a human should look at. Most well-grounded leads should PASS. Every issue you raise must point at something concrete in the source data.

You are a REVIEWER ONLY. Never rewrite the enrichment, the outreach, the classification or the priority. Never invent missing information. Only identify problems.

Respond with ONLY a JSON array, no prose, no markdown fences, one object per lead, in this exact shape:
[{{"lead_id":"L001","status":"pass","confidence":0.95,"issues":[],"recommended_action":"No action needed."}}]

status must be "pass" or "review"; confidence is 0.0-1.0; issues is a list of short specific strings (empty when passing); recommended_action is one short sentence ("No action needed." when passing).

Leads to review:
{payload}"""


def _review_placeholder(lead: dict) -> dict:
    """Conservative fallback when the reviewer fails to echo a lead.

    Routes to human review rather than silently passing a lead the reviewer
    never actually assessed.
    """
    return {
        "lead_id": lead.get("lead_id"),
        "status": "review",
        "confidence": 0.0,
        "issues": ["AI review did not return a result for this lead -- verify manually."],
        "recommended_action": "Verify this lead's intelligence manually.",
    }


def _normalize_review(record: dict) -> dict:
    """Clamp a review record to the documented output shape.

    Tolerates model drift (status casing, missing fields, out-of-range
    confidence, non-string issues) without crashing the pipeline or letting
    malformed values leak into the UI/CSV.
    """
    status = str(record.get("status") or "review").strip().lower()
    if status not in ("pass", "review"):
        status = "review"
    try:
        confidence = float(record.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    raw_issues = record.get("issues")
    if isinstance(raw_issues, str):
        raw_issues = [raw_issues]
    issues = [str(i).strip() for i in raw_issues if str(i).strip()] if isinstance(raw_issues, list) else []
    if status == "pass" and issues:
        status = "review"  # a pass with issues is a contradiction
    if status == "review" and not issues:
        issues = ["AI review flagged this lead without a specific issue -- verify manually."]
    action = str(record.get("recommended_action") or "").strip()
    if not action:
        action = "No action needed." if status == "pass" else "Review this lead before sending."
    return {
        "lead_id": record.get("lead_id"),
        "status": status,
        "confidence": round(confidence, 2),
        "issues": issues,
        "recommended_action": action,
    }


def review_leads(items: list[dict]) -> dict[str, dict]:
    """Second-pass semantic AI review of the finalized lead intelligence.

    `items` are {"lead": ..., "classification": ..., "enrichment": ...,
    "priority": ..., "outreach": ..., "qc_flags": [...]} wrappers for the
    leads to review (callers pass processed relevant leads only). The lead's
    original data/conversation is included as the source of truth; the AI
    reviewer only identifies problems -- it never rewrites anything.

    Returns {lead_id: {status, confidence, issues, recommended_action}}.
    Uses the shared Gemini batching/retry/schema infrastructure.
    """
    if not items:
        return {}
    batches = list(_batches(items, REVIEW_BATCH))

    def build_payload_item(item: dict) -> dict:
        lead = item.get("lead") or {}
        priority = item.get("priority") or {}
        qc_flags = item.get("qc_flags") or []
        return {
            "lead_id": lead.get("lead_id"),
            "original_lead": {
                "name": lead.get("name"),
                "city": lead.get("city"),
                "education": lead.get("education"),
                "experience_years": lead.get("experience_years"),
                "goal": lead.get("goal"),
                "german_level": lead.get("german_level"),
                "source": lead.get("source"),
                "conversation": lead.get("conversation"),
                "notes": lead.get("notes"),
                "missing_fields": lead.get("missing_fields"),
            },
            "classification": item.get("classification") or {},
            "enrichment": item.get("enrichment") or {},
            "priority": {
                "band": priority.get("band"),
                "score": priority.get("score"),
                "priority_reason": priority.get("priority_reason"),
            },
            "outreach": item.get("outreach") or "",
            "rule_based_qc_flags": qc_flags,
        }

    def build_prompt(batch: list[dict]) -> str:
        payload = [build_payload_item(i) for i in batch]
        return _AI_REVIEW_PROMPT.format(payload=json.dumps(payload, ensure_ascii=False))

    def merge(batch: list[dict], parsed: list) -> list[dict]:
        # Batch items are {"lead": ...} wrappers, so both the id lookup and
        # the placeholder fallback must unwrap the lead.
        unwrap = lambda item: (item.get("lead") or {}).get("lead_id")  # noqa: E731
        normalized = [_normalize_review(r) for r in parsed if isinstance(r, dict)]
        return _match_records_to_leads(
            batch, normalized,
            fallback=lambda item: _review_placeholder(item.get("lead") or {}),
            get_id=unwrap,
        )

    results = _run_batched(
        batches, build_prompt, merge,
        max_tokens=REVIEW_MAX_TOKENS, schema=REVIEW_SCHEMA,
    )
    out: dict[str, dict] = {}
    for batch_results in results:
        for r in batch_results:
            lid = r.get("lead_id")
            if lid:
                out[lid] = r
    return out


# ---------------------------------------------------------------------------
# Stage 6: QC (pure rules, no AI)
# ---------------------------------------------------------------------------

CONTRADICTION_TERMS = ["wrong market", "different profession", "not a healthcare", "not relevant", "unrelated profession"]

GENERIC_OUTREACH_PHRASES = re.compile(
    r"we'?d love to help|take the next step toward|notice you'?re interested",
    re.I,
)

UNSUPPORTED_CLAIMS = re.compile(
    r"guaranteed?\s+(?:a\s+)?(?:job|placement|visa|salary|admission|outcome)"
    r"|(?:job|placement|visa|salary)\s+(?:is\s+)?guaranteed"
    r"|\bplacement rate\b"
    r"|\b\d+\s?%\s?(?:success|placement|pass)\b"
    r"|\bsalary of\s",
    re.I,
)


def rule_based_qc(
    lead: dict,
    classification: dict,
    enrichment: dict,
    outreach_text: str | None,
    priority: dict | None = None,
) -> list[str]:
    flags = []
    relevant = classification.get("relevant")
    reason = (classification.get("reason") or "").lower()

    # Classification must exist and be a real boolean; anything else routes
    # the lead to human review rather than silently counting as relevant
    # or not relevant.
    if not isinstance(relevant, bool):
        flags.append(
            "Classification missing or malformed -- route to human review."
        )

    # Enrichment must exist for relevant leads; an empty enrichment would
    # otherwise masquerade as a low-priority lead instead of a data gap.
    if relevant and not enrichment:
        flags.append(
            "No enrichment returned for a relevant lead -- route to human review."
        )

    # Contradiction detection
    if relevant and any(t in reason for t in CONTRADICTION_TERMS):
        flags.append(
            "Classified relevant but reason text reads disqualifying -- verify manually."
        )

    # Low classification confidence check
    confidence = classification.get("confidence")
    if (
        confidence is not None
        and isinstance(confidence, (int, float))
        and confidence < 0.6
    ):
        flags.append(
            f"Low classification confidence ({confidence}) -- route to human review."
        )

    # Missing required contact fields
    if lead.get("missing_fields"):
        flags.append(
            f"Missing field(s) before contact: {', '.join(lead['missing_fields'])}."
        )

    # Conflicting duplicate records
    if lead.get("_dup_fuzzy"):
        flags.append(
            f"Possible re-submission with conflicting details "
            f"({', '.join(lead['_dup_fuzzy'])}) -- merge manually."
        )

    # Outreach language safety check
    if outreach_text and "guarantee" in outreach_text.lower():
        flags.append(
            'Outreach draft used the word "guarantee" -- rewrite before sending.'
        )

    if outreach_text and UNSUPPORTED_CLAIMS.search(outreach_text):
        flags.append(
            "Outreach makes an unsupported placement/salary/visa claim -- rewrite before sending."
        )

    # Outreach hygiene: length and question count (the prompt asks for
    # under 70 words and at most one question).
    if outreach_text:
        words = len(outreach_text.split())
        if words > 90:
            flags.append(
                f"Outreach draft is long ({words} words) -- tighten before sending."
            )
        questions = outreach_text.count("?")
        if questions > 2:
            flags.append(
                f"Outreach asks {questions} questions -- keep at most one next-step question."
            )

    if outreach_text and GENERIC_OUTREACH_PHRASES.search(outreach_text):
        flags.append(
            "Outreach uses generic filler phrasing -- personalize before sending."
        )

    # Missing or invalid urgency
    raw_urgency = enrichment.get("urgency")
    norm_urgency = str(raw_urgency).lower().strip() if raw_urgency else ""

    if not norm_urgency or norm_urgency not in URGENCY_SCORE:
        flags.append(
            f"Missing or invalid urgency ('{raw_urgency}') -- defaulted to low."
        )

    # Missing or invalid objection severity
    obj_cat = (
        str(enrichment.get("objection_category") or "")
        .lower()
        .strip()
    )

    raw_sev = enrichment.get("objection_severity")
    norm_sev = str(raw_sev).lower().strip() if raw_sev else ""

    valid_severities = {
        "mild",
        "moderate",
        "strong",
        "none",
    }

    if obj_cat and obj_cat != "none":
        if not norm_sev or norm_sev not in valid_severities:
            flags.append(
                f"Missing or invalid objection severity "
                f"('{raw_sev}') for category '{obj_cat}' -- "
                f"defaulted to moderate."
            )

    elif norm_sev and norm_sev not in ("none", ""):
        flags.append(
            f"Objection severity '{raw_sev}' specified "
            f"but objection category is none."
        )

    # Suspicious or unsupported AI outputs
    valid_objection_categories = {
        "price",
        "timeline",
        "confidence",
        "eligibility",
        "qualification",
        "none",
    }

    if obj_cat and obj_cat not in valid_objection_categories:
        flags.append(
            f"Unsupported objection category "
            f"'{enrichment.get('objection_category')}'."
        )

    # Healthcare qualification sanity check
    edu = (lead.get("education") or "").lower()

    if relevant and ("engineer" in edu or "bba" in edu):
        flags.append(
            f"Non-healthcare qualification ({lead.get('education')}) "
            "classified as relevant -- verify candidate intent."
        )

    # Priority score formula validation
    pr = priority or lead.get("priority")

    if pr and pr.get("score") is not None and relevant:
        expected = compute_priority(
            lead,
            classification,
            enrichment,
        )

        if expected.get("score") is not None:
            if abs(pr["score"] - expected["score"]) > 0.01:
                flags.append(
                    f"Priority score mismatch: recorded {pr['score']} "
                    f"vs formula expected {expected['score']}."
                )

            if pr.get("band") != expected.get("band"):
                flags.append(
                    f"Priority band mismatch: recorded '{pr.get('band')}' "
                    f"vs formula expected '{expected.get('band')}'."
                )

    return flags


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

STAGE_ORDER = ["clean", "classify", "enrich", "prioritize", "outreach", "qc", "done"]


def run_full_pipeline_stream(raw_leads: list[dict]) -> Iterator[tuple[str, str, str, dict | None]]:
    """Run every stage in order, yielding (stage, status, message, payload)
    after each one so callers can stream real per-stage progress.

    Payloads are incremental result fragments; the final ("done") payload is
    the complete, exportable dataset. `run_full_pipeline` returns exactly
    that final dict.
    """
    started = time.time()

    # --- Stage 1: clean + dedupe (rules) -----------------------------------
    yield ("clean", "active", f"Cleaning {len(raw_leads)} raw leads...", None)
    cleaned = clean_and_dedupe(raw_leads)
    primaries = cleaned["primaries"]
    lead_audit = cleaned["lead_audit"]
    n_dupes = sum(len(g["all"]) - 1 for g in cleaned["duplicate_groups"])
    yield (
        "clean", "done",
        f"{len(primaries)} primary records retained ({n_dupes} duplicate record(s) resolved).",
        {"primaries": primaries, "duplicate_groups": cleaned["duplicate_groups"], "lead_audit": lead_audit},
    )

    # --- Stage 2: classify (AI) --------------------------------------------
    yield ("classify", "active", f"Classifying relevance for {len(primaries)} leads...", None)
    classifications = classify_leads(primaries)
    n_relevant = sum(1 for c in classifications.values() if c.get("relevant"))
    yield (
        "classify", "done",
        f"{n_relevant} of {len(primaries)} leads classified relevant.",
        {"classifications": classifications},
    )

    # --- Stage 3: enrich (AI, relevant leads only) --------------------------
    relevant_leads = [
        p for p in primaries if classifications.get(p["lead_id"], {}).get("relevant")
    ]
    yield ("enrich", "active", f"Enriching {len(relevant_leads)} relevant leads...", None)
    enrichments = enrich_leads(relevant_leads)
    yield (
        "enrich", "done",
        f"{len(enrichments)} of {len(relevant_leads)} relevant leads enriched.",
        {"enrichments": enrichments},
    )

    # --- Stage 4: prioritize (rules) ----------------------------------------
    yield ("prioritize", "active", "Scoring priority with the fixed formula...", None)
    for p in primaries:
        cls = classifications.get(p["lead_id"], {"relevant": False})
        enr = enrichments.get(p["lead_id"], {})
        p["priority"] = compute_priority(p, cls, enr)
    high = sum(1 for p in primaries if (p.get("priority") or {}).get("band") == "High")
    yield (
        "prioritize", "done",
        f"Priorities computed ({high} high-priority lead(s)).",
        {"priorities_done": True},
    )

    # --- Stage 5: outreach (AI, relevant leads only) -------------------------
    outreach_items = [
        {"lead": p, "enrichment": enrichments.get(p["lead_id"], {})}
        for p in relevant_leads
    ]
    yield ("outreach", "active", f"Drafting personalized outreach for {len(outreach_items)} lead(s)...", None)
    outreach = generate_outreach(outreach_items)
    yield (
        "outreach", "done",
        f"{len(outreach)} outreach draft(s) generated.",
        {"outreach": outreach},
    )

    # --- Stage 6: QC (rules) -------------------------------------------------
    yield ("qc", "active", "Running rule-based quality checks...", None)
    qc: dict[str, list[str]] = {}
    for p in primaries:
        cls = classifications.get(p["lead_id"], {})
        enr = enrichments.get(p["lead_id"], {})
        qc[p["lead_id"]] = rule_based_qc(p, cls, enr, outreach.get(p["lead_id"]), p.get("priority"))

    # Include duplicate / merged leads in QC so all original lead IDs are traceable
    for audit in lead_audit:
        lid = audit["lead_id"]
        if lid not in qc:
            if audit["status"] == "merged":
                qc[lid] = [f"Auto-merged into {audit['primary_id']} ({audit['reason']})."]
            elif audit["status"] == "flagged" or audit["requires_human_review"]:
                qc[lid] = [f"Excluded from active batch pending review: {audit['reason']}."]

    n_flagged = sum(1 for flags in qc.values() if flags)
    yield (
        "qc", "done",
        f"{n_flagged} record(s) carry QC flags.",
        {"qc": qc},
    )

    # --- Stage 7: AI review (semantic second-pass, relevant leads only) ------
    # Records excluded from active processing (unresolved conflicting
    # duplicates, e.g. L028) are NOT reviewed: they never went through
    # classification/enrichment/priority/outreach, and forcing them through
    # would change the existing duplicate-handling behavior.
    review_items = [
        {
            "lead": p,
            "classification": classifications.get(p["lead_id"], {}),
            "enrichment": enrichments.get(p["lead_id"], {}),
            "priority": p.get("priority") or {},
            "outreach": outreach.get(p["lead_id"], ""),
            "qc_flags": qc.get(p["lead_id"], []),
        }
        for p in relevant_leads
    ]
    yield ("ai_review", "active", f"AI-reviewing final intelligence for {len(review_items)} lead(s)...", None)
    ai_review = review_leads(review_items)
    n_review = sum(1 for r in ai_review.values() if r.get("status") == "review")
    yield (
        "ai_review", "done",
        f"AI review: {n_review} lead(s) need a human look, {len(ai_review) - n_review} passed.",
        {"ai_review": ai_review},
    )

    runtime = round(time.time() - started, 2)
    result = {
        "primaries": primaries,
        "duplicate_groups": cleaned["duplicate_groups"],
        "lead_audit": lead_audit,
        "classifications": classifications,
        "enrichments": enrichments,
        "outreach": outreach,
        "qc": qc,
        "ai_review": ai_review,
        "summary": {
            "raw_leads": len(raw_leads),
            "primaries": len(primaries),
            "classified": len(classifications),
            "relevant": len(relevant_leads),
            "enriched": len(enrichments),
            "prioritized": sum(1 for p in primaries if (p.get("priority") or {}).get("band") not in (None, "N/A")),
            "with_outreach": len(outreach),
            "needs_review": n_flagged,
            "ai_review_flagged": n_review,
            "ai_review_passed": len(ai_review) - n_review,
            "duplicates_merged": sum(len(g["exact"]) for g in cleaned["duplicate_groups"]),
            "duplicates_flagged": sum(len(g["fuzzy"]) for g in cleaned["duplicate_groups"]),
            "runtime_seconds": runtime,
        },
    }
    yield (
        "done", "done",
        f"Pipeline complete: {len(primaries)} primary leads processed in {runtime}s.",
        result,
    )


def run_full_pipeline(raw_leads: list[dict]) -> dict:
    """Runs every stage in order and returns the assembled, exportable dataset."""
    result: dict | None = None
    for _, _, _, payload in run_full_pipeline_stream(raw_leads):
        if payload is not None and "summary" in payload:
            result = payload
    if result is None:  # pragma: no cover - defensive
        raise RuntimeError("Pipeline produced no result.")
    return result
