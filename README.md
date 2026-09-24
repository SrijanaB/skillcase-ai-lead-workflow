# Skillcase Lead Manifest

Turns 30 messy inbound B2C lead conversations into a cleaned, classified,
enriched, prioritized, and outreach-ready dataset — with a working web UI
anyone can use, no login required.

## Architecture

```
data/leads.json (raw, messy)
        |
        v
  clean_and_dedupe()          <- rules only, no AI
   - normalizes fields
   - groups by phone number
   - exact duplicates auto-merge; conflicting ("fuzzy") duplicates are
     flagged for a human instead of guessed
   - flags missing required fields
        |
        v
  classify_leads()            <- batched Gemini calls (15 leads/call, 2 in parallel)
   - relevant / not relevant / reason / confidence
   - relevance criteria are explicit in the prompt (see pipeline.py)
        |
        v
  enrich_leads()               <- batched Gemini calls, RELEVANT leads only
   - profile, intent (+ level), need, objection (+ category/severity),
     urgency, engagement, missing info, opportunity, next action
        |
        v
  compute_priority()           <- rules only, no AI
   - transparent, documented scoring formula (see pipeline.py):
     0.40*Intent + 0.25*Urgency + 0.20*Fit + 0.15*Engagement
   - objections never lower the score; they steer the follow-up
        |
        v
  generate_outreach()          <- batched Gemini calls (6 leads/call)
   - personalized message per relevant lead, grounded in the original
     conversation; explicitly instructed not to promise job outcomes
        |
        v
  rule_based_qc()               <- rules only, no AI
   - contradiction detection (relevant=true but reason reads
     disqualifying)
   - low-confidence routing (< 0.6 -> human review)
   - missing/malformed classification or enrichment -> human review
   - missing-field flags
   - fuzzy-duplicate flags
   - outreach hygiene: guarantee/placement/salary/visa claims, generic
     filler, length, question count
   - priority score/band recomputed and cross-checked against the formula
        |
        v
  final dataset (table in the UI, or CSV export)
```

Every stage lives in `pipeline.py` as a plain, independently testable
function. `app.py` is a thin Flask layer that exposes one endpoint per
stage plus two end-to-end entry points: `/api/pipeline/run` (single
response, used by the automation script below) and `/api/pipeline/stream`
(server-sent events with real per-stage progress, used by the web UI).

## Gemini robustness

All AI calls go through one helper (`_ask_for_json_array` in
`pipeline.py`) that provides:

- **JSON response mode with a schema**, falling back automatically to
  plain JSON mode if the model rejects `response_json_schema` (400).
- **Batched requests** (classification 15 leads/call, enrichment 8,
  outreach 6) with bounded concurrency, so no single response is large
  enough to hit the max-token ceiling and each call stays fast.
- **Exponential backoff with jitter** on 429/500/502/503/504, network
  timeouts and malformed output, honoring the server's `retryDelay`
  hint on 429s (4 attempts max, capped wait).
- **Truncation salvage**: a response cut off mid-array keeps its
  complete records instead of being discarded.
- **Echo validation**: every stage checks that the model returned every
  lead id; a skipped lead gets a conservative placeholder and a QC flag
  rather than silently vanishing.
- **Explicit timeouts** and automatic function calling (AFC) disabled
  (this pipeline never uses function calling).

Enrichment and outreach run only for leads classified relevant, so no
API time is spent on leads that will never be contacted.

## Quality control

## Automation

`scripts/run_and_export.py` runs the entire pipeline with no UI and
writes a CSV. This is the piece meant to run unattended — on a cron
schedule, in CI, or triggered by a webhook when a new lead lands in a
sheet or CRM:

```bash
GEMINI_API_KEY=your-key python scripts/run_and_export.py --out today.csv
```

Example cron entry (every morning at 7am):

```
0 7 * * * cd /path/to/skillcase-app && python3 scripts/run_and_export.py --out exports/$(date +%F).csv
```

## Quality control

Three independent mechanisms, not just one:

1. **Structured-output validation** — every Gemini call requests strict
   JSON (schema first, plain JSON mode as fallback); malformed or
   truncated output is retried with backoff, salvaged, or backfilled
   with a placeholder and flagged (see `pipeline.py`).
2. **Rule-based sanity checks** — contradiction detection, confidence
   thresholding, missing-field flags, duplicate-conflict flags,
   missing/malformed classification or enrichment, outreach claim and
   hygiene checks, and a full recomputation of the priority formula,
   all deterministic and independent of the AI calls that produced the
   data they're checking.
3. **Human review queue** — the "Needs Human Review" filter in the UI
   surfaces exactly the leads that failed one of the checks above, and
   leads with genuine uncertainty are routed there rather than silently
   accepted.

Three concrete examples the pipeline catches on this dataset (shown
live in the "What the pipeline flagged" panel after a run):

- **L001 / L008 / L028** — same phone number. L008 is an exact
  duplicate and gets auto-merged; L028 shares the phone/email but has a
  different name spelling and experience format, so it's kept separate
  and flagged for manual reconciliation instead of merged automatically.
- **L007** — missing experience entirely; the classifier still returns
  a judgment, but the QC layer flags the record as needing that field
  before contact rather than pretending it's complete.
- **L005** — a BBA graduate asking whether non-nursing people can get
  nursing jobs. This is the kind of case where classifier confidence is
  expected to be lowest, and it's the sort of judgment call intentionally
  routed to a human rather than auto-decided.

## Local setup

```bash
cd skillcase-app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then edit .env and add your GEMINI_API_KEY
python app.py
```

Open http://localhost:5000 and click **Run pipeline**.

## Deploying so anyone can use it (no account needed)

**Render (free tier, ~10 minutes):**

1. Push this folder to a GitHub repo.
2. On [render.com](https://render.com), New -> Web Service -> connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Add an environment variable `GEMINI_API_KEY` with your key.
6. Deploy. Render gives you a public URL — share that link.

Railway, Fly.io, or PythonAnywhere work the same way: install
dependencies, run `gunicorn app:app`, set the one environment variable.

**Cost/abuse note:** because the API key lives on the server, every
visitor's pipeline run is billed to that key. Fine for a demo shared
with a small number of reviewers; for anything wider, add a rate
limiter (e.g. `flask-limiter`) before sharing broadly.

## Design decisions worth knowing for the review

- **Prioritization is rule-based, not AI-scored.** A visible formula is
  easier to defend and debug than a model-generated number, and the
  brief explicitly asks for explainable criteria.
- **Cleaning and QC never depend on the AI calls succeeding.** They're
  pure functions over the raw/cleaned data, so a bad Gemini response
  degrades gracefully rather than corrupting the whole pipeline.
- **Batched, not per-lead, AI calls.** Classification, enrichment, and
  outreach each run as a single call across all leads rather than 30
  separate calls per stage — cheaper, faster, and it's what "automation"
  should look like at this scale.
