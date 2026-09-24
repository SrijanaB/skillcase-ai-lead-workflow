"""
Skillcase lead pipeline.

Raw messy leads -> clean/dedupe (rules) -> classify (AI) -> enrich (AI)
-> prioritize (rules) -> outreach (AI) -> QC (rules).

Each stage is a standalone function so it can be tested and explained
independently. The two AI-touching functions (classify_leads,
enrich_leads, generate_outreach) are the only ones that call the
Gemini API; everything else is deterministic Python.
"""
import json
import os
import re
import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors

MODEL = "gemini-3.5-flash-lite"
REQUIRED_FIELDS = ["phone", "email", "city", "education", "experience_years", "goal", "german_level"]

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
            "need": {"type": "string"},
            "objection": {"type": "string"},
            "objection_category": {
                "type": "string",
                "enum": ["price", "timeline", "confidence", "eligibility", "qualification", "none"],
            },
            "objection_severity": {
                "type": "string",
                "enum": ["mild", "moderate", "strong", "none"],
            },
            "urgency": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "missing_info": {"type": "string"},
            "opportunity": {"type": "string"},
            "next_action": {"type": "string"},
        },
        "required": [
            "lead_id", "profile", "intent", "need", "objection",
            "objection_category", "objection_severity", "urgency",
            "missing_info", "opportunity", "next_action",
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


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your environment or .env file.")
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
    t = raw.lower().strip()
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
    key = re.sub(r"\s+", " ", raw.lower().replace(".", "")).strip()
    mapping = {
        "bsc nursing": "BSc Nursing",
        "gnm": "GNM",
        "bpharm": "BPharm",
        "bba": "BBA",
        "engineer": "Engineer",
    }
    return mapping.get(key, raw.strip())


def title_case(raw: str) -> str:
    return " ".join(w.capitalize() for w in raw.strip().lower().split())


def normalize_goal(raw: str | None) -> str | None:
    if not raw:
        return None
    t = raw.strip().lower()
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
    return title_case(raw)


def _completeness(lead: dict) -> int:
    return sum(1 for f in REQUIRED_FIELDS if lead.get(f) not in (None, ""))


def clean_and_dedupe(raw_leads: list[dict]) -> dict:
    cleaned = []
    for r in raw_leads:
        cleaned.append({
            "lead_id": r["lead_id"],
            "name": title_case(r.get("name") or ""),
            "phone": (r.get("phone") or "").replace(" ", ""),
            "email": (r.get("email") or "").strip().lower() or None,
            "city": (r.get("city") or "").strip() or None,
            "education": normalize_education(r.get("education")),
            "experience_raw": r.get("experience"),
            "experience_years": parse_experience_years(r.get("experience")),
            "goal": normalize_goal(r.get("goal")),
            "german_level": (r.get("german_level") or "").strip().upper() or None,
            "source": (r.get("source") or "").strip(),
            "last_contacted": r.get("last_contacted"),
            "conversation": r.get("conversation"),
            "notes": r.get("notes"),
        })

    groups: dict[str, list[dict]] = {}
    for c in cleaned:
        groups.setdefault(c["phone"], []).append(c)

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

def _parse_json_array(text: str) -> list[dict]:
    if not text or not str(text).strip():
        raise ValueError("Empty response from model.")
    stripped = str(text).strip()
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


def _ask_for_json_array(
    prompt: str, max_tokens: int = 4000, schema: dict | None = None
) -> list[dict]:
    client = get_client()
    config: dict[str, Any] = {
        "response_mime_type": "application/json",
        "max_output_tokens": max_tokens,
    }
    if schema is not None:
        config["response_json_schema"] = schema

    last_error: BaseException | None = None
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config,
            )
            parsed = getattr(resp, "parsed", None)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for value in parsed.values():
                    if isinstance(value, list):
                        return value
            return _parse_json_array(getattr(resp, "text", None) or "")
        except genai_errors.APIError as exc:
            last_error = exc
            code = getattr(exc, "code", None)
            if code in (429, 500, 503) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(
                f"Gemini API error ({code}): {_safe_api_error_message(exc)}"
            ) from None
        except (ValueError, json.JSONDecodeError):
            raise
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
    try:
        return _ask_for_json_array(prompt, max_tokens, schema=schema)
    except (ValueError, json.JSONDecodeError):
        return _ask_for_json_array(prompt, max_tokens, schema=schema)


# ---------------------------------------------------------------------------
# Stage 2: classify (AI)
# ---------------------------------------------------------------------------

def classify_leads(leads: list[dict]) -> dict[str, dict]:
    payload = [{
        "lead_id": l["lead_id"], "education": l["education"], "experience_years": l["experience_years"],
        "goal": l["goal"], "german_level": l["german_level"], "conversation": l["conversation"],
        "missing_fields": l["missing_fields"],
    } for l in leads]

    prompt = f"""You are triaging inbound leads for Skillcase, a company that prepares nurses and allied healthcare workers in India to learn German and get placed as nurses in Germany.

Relevance criteria:
- RELEVANT: the person has (or is credibly pursuing) a nursing/allied-healthcare background AND is interested in working or preparing for work in Germany specifically, at any stage (even early exploration, even if only asking preparatory questions).
- NOT RELEVANT: the person is targeting a different country only (e.g. Canada, UK), is in an unrelated profession with no stated intent to pivot into healthcare, or is asking on behalf of a background that does not fit.
- Missing fields lower confidence but do not automatically make a lead not relevant.

For each lead below, return a relevance judgment. Respond with ONLY a JSON array, no prose, no markdown fences, in this exact shape:
[{{"lead_id":"L001","relevant":true,"reason":"short reason under 15 words","confidence":0.0}}]

Leads:
{json.dumps(payload)}"""

    results = _ask_for_json_array_with_retry(prompt, schema=CLASSIFY_SCHEMA)
    return {r["lead_id"]: r for r in results}


# ---------------------------------------------------------------------------
# Stage 3: enrich (AI)
# ---------------------------------------------------------------------------

def enrich_leads(leads: list[dict]) -> dict[str, dict]:
    payload = [{
        "lead_id": l["lead_id"], "name": l["name"], "city": l["city"], "education": l["education"],
        "experience_years": l["experience_years"], "goal": l["goal"], "german_level": l["german_level"],
        "source": l["source"], "conversation": l["conversation"], "missing_fields": l["missing_fields"],
    } for l in leads]

    prompt = f"""For each Skillcase lead below (a nursing-to-Germany migration prep service), extract structured sales context from the conversation. Be concise -- each field should be one short sentence or phrase. If information genuinely isn't present, say "Not stated" rather than guessing.

Fields per lead:
- profile: one line on their background (education, experience, location)
- intent: what they appear to be trying to achieve
- need: what they likely need help with next
- objection: their stated hesitation or concern, in their words if possible
- objection_category: one of price | timeline | confidence | eligibility | qualification | none
- objection_severity: one of mild | moderate | strong | none
    * mild: passing concern or light question
    * moderate: real concern or hesitation but not explicitly blocking
    * strong: explicitly stated as a blocker or barrier
    * none: no objection stated
- urgency: one of high | medium | low (judge urgency from the holistic meaning and readiness of the full conversation, NOT keyword matching)
    * high: concrete readiness, such as requesting a call, giving a clear timeline, or stating they are ready to start
    * medium: interested but no immediate action planned
    * low: early-stage exploration or passive inquiry
- missing_info: the single most important piece of info a salesperson should still collect
- opportunity: what Skillcase could concretely help them with
- next_action: the one recommended next step for the salesperson

Respond with ONLY a JSON array, no prose, no markdown fences:
[{{"lead_id":"L001","profile":"...","intent":"...","need":"...","objection":"...","objection_category":"...","objection_severity":"...","urgency":"...","missing_info":"...","opportunity":"...","next_action":"..."}}]

Leads:
{json.dumps(payload)}"""

    results = _ask_for_json_array_with_retry(prompt, max_tokens=6000, schema=ENRICH_SCHEMA)
    return {r["lead_id"]: r for r in results}


# ---------------------------------------------------------------------------
# Stage 4: prioritize (pure rules, no AI)
# ---------------------------------------------------------------------------

GERMAN_LEVEL_SCORE = {"B2": 3, "B1": 2, "A2": 1, "A1": 0}
URGENCY_BONUS = {
    "high": 2,
    "medium": 1,
    "low": 0,
}
OBJECTION_BASE = {
    "price": 2.0,
    "qualification": 1.5,
    "eligibility": 1.5,
    "confidence": 1.0,
    "timeline": 0.5,
    "none": 0.0,
}
SEVERITY_MULTIPLIER = {
    "strong": 1.5,
    "moderate": 1.0,
    "mild": 0.5,
    "none": 0.0,
}


def _build_priority_reason(
    lead: dict,
    band: str,
    german_level: str | None,
    exp_years: float,
    source: str | None,
    urgency: str,
    raw_cat: str,
    sev: str,
    obj_penalty: float,
) -> str:
    factors = []

    if german_level == "B2":
        factors.append("strong German proficiency")
    elif german_level == "B1":
        factors.append("intermediate German proficiency")
    elif german_level == "A2":
        factors.append("elementary German proficiency")
    elif german_level == "A1":
        factors.append("beginner German proficiency")

    if exp_years >= 1:
        factors.append("relevant experience")
    elif exp_years > 0:
        factors.append("clinical experience")

    if source == "Referral":
        factors.append("referral bonus")
    elif source == "WhatsApp":
        factors.append("direct WhatsApp channel")

    if urgency == "high":
        factors.append("high urgency")
    elif urgency == "medium":
        factors.append("moderate urgency")

    objection_note = ""
    if raw_cat != "none" and obj_penalty > 0:
        objection_note = f" A {sev} {raw_cat} objection reduced the score."

    if band == "High":
        if factors:
            if len(factors) == 1:
                lead_clause = factors[0]
            elif len(factors) == 2:
                lead_clause = f"{factors[0]} and {factors[1]}"
            else:
                lead_clause = f"{', '.join(factors[:-1])}, and {factors[-1]}"
            return f"High priority because of {lead_clause}.{objection_note}".strip()
        return f"High priority based on overall qualifications.{objection_note}".strip()
    elif band == "Medium":
        if factors:
            if len(factors) == 1:
                lead_clause = factors[0]
            elif len(factors) == 2:
                lead_clause = f"{factors[0]} and {factors[1]}"
            else:
                lead_clause = f"{', '.join(factors[:-1])}, and {factors[-1]}"
            return f"Medium priority with {lead_clause}.{objection_note}".strip()
        return f"Medium priority based on balanced qualifications.{objection_note}".strip()
    else:  # Low
        reasons = []
        if not german_level or german_level in ("A1", "A2"):
            reasons.append(f"lower German proficiency ({german_level or 'none'})")
        if exp_years < 1:
            reasons.append("limited clinical experience")
        if urgency == "low":
            reasons.append("early-stage exploration")
        if not reasons:
            reasons = ["early-stage profile"]

        if len(reasons) == 1:
            clause = reasons[0]
        elif len(reasons) == 2:
            clause = f"{reasons[0]} and {reasons[1]}"
        else:
            clause = f"{', '.join(reasons[:-1])}, and {reasons[-1]}"
        return f"Low priority due to {clause}.{objection_note}".strip()


def compute_priority(lead: dict, classification: dict, enrichment: dict) -> dict:
    if not classification.get("relevant"):
        return {
            "score": None,
            "band": "N/A",
            "priority_reason": "Not scored because lead is not relevant to Skillcase's healthcare program.",
        }

    gl = GERMAN_LEVEL_SCORE.get(lead.get("german_level"), 0)
    german_points = gl * 2
    exp_years = min(lead.get("experience_years") or 0, 5)
    source_bonus = 1 if lead.get("source") == "Referral" else (0.5 if lead.get("source") == "WhatsApp" else 0)

    # Urgency bonus from AI-generated field (with safe validation and sensible default)
    raw_urgency = (enrichment.get("urgency") or "").lower().strip()
    urgency = raw_urgency if raw_urgency in URGENCY_BONUS else "low"
    urgency_bonus = URGENCY_BONUS[urgency]

    # Category-specific objection penalty with severity multiplier
    raw_cat = (enrichment.get("objection_category") or "").lower().strip()
    if raw_cat and raw_cat != "none":
        obj_base = OBJECTION_BASE.get(raw_cat, 1.0)
        raw_sev = (enrichment.get("objection_severity") or "").lower().strip()
        sev = raw_sev if raw_sev in SEVERITY_MULTIPLIER else "moderate"
        multiplier = SEVERITY_MULTIPLIER[sev]
        obj_penalty = round(obj_base * multiplier, 2)
    else:
        raw_cat = "none"
        sev = "none"
        obj_penalty = 0.0

    score = round(german_points + exp_years + source_bonus + urgency_bonus - obj_penalty, 1)
    band = "High" if score >= 8 else ("Medium" if score >= 4 else "Low")

    reason = _build_priority_reason(
        lead=lead,
        band=band,
        german_level=lead.get("german_level"),
        exp_years=lead.get("experience_years") or 0,
        source=lead.get("source"),
        urgency=urgency,
        raw_cat=raw_cat,
        sev=sev,
        obj_penalty=obj_penalty,
    )

    return {
        "score": score,
        "band": band,
        "priority_reason": reason,
    }


# ---------------------------------------------------------------------------
# Stage 5: outreach (AI)
# ---------------------------------------------------------------------------

def generate_outreach(items: list[dict]) -> dict[str, str]:
    """items: [{"lead": <cleaned lead dict>, "enrichment": <enrichment dict>}]"""
    payload = [{
        "lead_id": i["lead"]["lead_id"], "name": i["lead"]["name"], "city": i["lead"]["city"],
        "source": i["lead"]["source"], "profile": i["enrichment"].get("profile"),
        "intent": i["enrichment"].get("intent"), "need": i["enrichment"].get("need"),
        "objection": i["enrichment"].get("objection"), "opportunity": i["enrichment"].get("opportunity"),
    } for i in items]

    if not payload:
        return {}

    prompt = f"""Write a short personalized outreach message (under 70 words) for each Skillcase lead below, to be sent over the channel they came from. Reference their specific situation, intent and objection -- never a generic template. Do not make guarantees about job placement or outcomes; Skillcase supports the process but cannot promise a job. Warm, direct, no corporate filler.

Respond with ONLY a JSON array, no prose, no markdown fences:
[{{"lead_id":"L001","outreach":"message text"}}]

Leads:
{json.dumps(payload)}"""

    results = _ask_for_json_array_with_retry(prompt, max_tokens=4000, schema=OUTREACH_SCHEMA)
    return {r["lead_id"]: r["outreach"] for r in results}


# ---------------------------------------------------------------------------
# Stage 6: QC (pure rules, no AI)
# ---------------------------------------------------------------------------

CONTRADICTION_TERMS = ["wrong market", "different profession", "not a healthcare", "not relevant", "unrelated profession"]


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

    # Contradiction detection
    if relevant and any(t in reason for t in CONTRADICTION_TERMS):
        flags.append("Classified relevant but reason text reads disqualifying -- verify manually.")

    # Low classification confidence check
    confidence = classification.get("confidence")
    if confidence is not None and isinstance(confidence, (int, float)) and confidence < 0.6:
        flags.append(f"Low classification confidence ({confidence}) -- route to human review.")

    # Missing required contact fields
    if lead.get("missing_fields"):
        flags.append(f"Missing field(s) before contact: {', '.join(lead['missing_fields'])}.")

    # Conflicting duplicate records
    if lead.get("_dup_fuzzy"):
        flags.append(f"Possible re-submission with conflicting details ({', '.join(lead['_dup_fuzzy'])}) -- merge manually.")

    # Outreach language safety check
    if outreach_text and "guarantee" in outreach_text.lower():
        flags.append('Outreach draft used the word "guarantee" -- rewrite before sending.')

    # Missing or invalid urgency
    raw_urgency = enrichment.get("urgency")
    norm_urgency = str(raw_urgency).lower().strip() if raw_urgency else ""
    if not norm_urgency or norm_urgency not in URGENCY_BONUS:
        flags.append(f"Missing or invalid urgency ('{raw_urgency}') -- defaulted to low.")

    # Missing or invalid objection severity
    obj_cat = (enrichment.get("objection_category") or "").lower().strip()
    raw_sev = enrichment.get("objection_severity")
    norm_sev = str(raw_sev).lower().strip() if raw_sev else ""
    if obj_cat and obj_cat != "none":
        if not norm_sev or norm_sev not in SEVERITY_MULTIPLIER:
            flags.append(f"Missing or invalid objection severity ('{raw_sev}') for category '{obj_cat}' -- defaulted to moderate.")
    elif norm_sev and norm_sev not in ("none", ""):
        flags.append(f"Objection severity '{raw_sev}' specified but objection category is none.")

    # Suspicious or unsupported AI outputs
    if obj_cat and obj_cat not in OBJECTION_BASE:
        flags.append(f"Unsupported objection category '{enrichment.get('objection_category')}'.")

    edu = (lead.get("education") or "").lower()
    if relevant and ("engineer" in edu or "bba" in edu):
        flags.append(f"Non-healthcare qualification ({lead.get('education')}) classified as relevant -- verify candidate intent.")

    # Priority score formula validation
    pr = priority or lead.get("priority")
    if pr and pr.get("score") is not None and relevant:
        expected = compute_priority(lead, classification, enrichment)
        if expected.get("score") is not None:
            if abs(pr["score"] - expected["score"]) > 0.01:
                flags.append(f"Priority score mismatch: recorded {pr['score']} vs formula expected {expected['score']}.")
            if pr.get("band") != expected.get("band"):
                flags.append(f"Priority band mismatch: recorded '{pr.get('band')}' vs formula expected '{expected.get('band')}'.")

    return flags


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_full_pipeline(raw_leads: list[dict]) -> dict:
    """Runs every stage in order and returns the assembled, exportable dataset."""
    cleaned = clean_and_dedupe(raw_leads)
    primaries = cleaned["primaries"]
    lead_audit = cleaned["lead_audit"]

    classifications = classify_leads(primaries)
    enrichments = enrich_leads(primaries)

    for p in primaries:
        cls = classifications.get(p["lead_id"], {"relevant": False})
        enr = enrichments.get(p["lead_id"], {})
        p["priority"] = compute_priority(p, cls, enr)

    outreach_items = [
        {"lead": p, "enrichment": enrichments.get(p["lead_id"], {})}
        for p in primaries if classifications.get(p["lead_id"], {}).get("relevant")
    ]
    outreach = generate_outreach(outreach_items)

    qc = {}
    for p in primaries:
        cls = classifications.get(p["lead_id"], {})
        enr = enrichments.get(p["lead_id"], {})
        qc[p["lead_id"]] = rule_based_qc(p, cls, enr, outreach.get(p["lead_id"]), p.get("priority"))

    # Include duplicate / merged leads in QC so all 30 original lead IDs are traceable
    for audit in lead_audit:
        lid = audit["lead_id"]
        if lid not in qc:
            if audit["status"] == "merged":
                qc[lid] = [f"Auto-merged into {audit['primary_id']} ({audit['reason']})."]
            elif audit["status"] == "flagged" or audit["requires_human_review"]:
                qc[lid] = [f"Excluded from active batch pending review: {audit['reason']}."]

    return {
        "primaries": primaries,
        "duplicate_groups": cleaned["duplicate_groups"],
        "lead_audit": lead_audit,
        "classifications": classifications,
        "enrichments": enrichments,
        "outreach": outreach,
        "qc": qc,
    }
