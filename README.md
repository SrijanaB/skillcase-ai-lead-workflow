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
  classify_leads()            <- 1 batched Gemini call
   - relevant / not relevant / reason / confidence
   - relevance criteria are explicit in the prompt (see pipeline.py)
        |
        v
  enrich_leads()               <- 1 batched Gemini call
   - profile, intent, need, objection (+ category), missing info,
     opportunity, next action
        |
        v
  compute_priority()           <- rules only, no AI
   - transparent, documented scoring formula (see the app's own
     "How priority is scored" panel, or pipeline.py)
        |
        v
  generate_outreach()          <- 1 batched Gemini call
   - personalized message per relevant lead
   - explicitly instructed not to promise job outcomes
        |
        v
  rule_based_qc()               <- rules only, no AI
   - contradiction detection (relevant=true but reason reads
     disqualifying)
   - low-confidence routing (< 0.6 -> human review)
   - missing-field flags
   - fuzzy-duplicate flags
   - catches the word "guarantee" in outreach drafts before they'd go out
        |
        v
  final dataset (table in the UI, or CSV export)
```

Every stage lives in `pipeline.py` as a plain, independently testable
function. `app.py` is a thin Flask layer that exposes one endpoint per
stage (so the frontend can show live per-stage progress) plus one
`/api/pipeline/run` endpoint that runs everything in one call (used by
the automation script below).

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

1. **Structured-output validation** — every Gemini call is prompted for
   a strict JSON array; a failed parse triggers one automatic retry
   before surfacing an error (see `_ask_for_json_array_with_retry` in
   `pipeline.py`).
2. **Rule-based sanity checks** — contradiction detection, confidence
   thresholding, missing-field flags, duplicate-conflict flags, and an
   outreach-language check, all deterministic and independent of the AI
   calls that produced the data they're checking.
3. **Human review queue** — the "Needs review" filter in the UI surfaces
   exactly the leads that failed one of the checks above, rather than
   the system silently proceeding.

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
