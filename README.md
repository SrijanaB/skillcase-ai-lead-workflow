# Skillcase Lead Manifest

Skillcase AI Lead Manifest is an AI-assisted B2C lead intelligence workflow. It turns messy,
real-world lead conversations — inconsistent fields, duplicates, half-filled
forms, rambling chat transcripts — into a structured, prioritized sales
workspace: every lead cleaned, classified, enriched, scored, and paired with
personalized outreach, with ambiguous cases explicitly routed to a human
instead of silently guessed.

The bundled dataset is 30 raw inbound B2C leads (nurses and allied healthcare
workers in India interested in working in Germany). The result is a
sales-ready lead queue a salesperson can actually work through — in the web
UI or as a CSV — no login required.

## Architecture

```
Raw Leads (data/leads.json, messy)
        |
        v
Clean & Deduplicate          clean_and_dedupe()      <- rules only, no AI
   - normalizes fields
   - groups by phone number
   - exact duplicates auto-merge; conflicting ("fuzzy") duplicates are
     flagged for a human instead of guessed
   - flags missing required fields
        |
        v
AI Classification            classify_leads()        <- batched Gemini calls
   - relevant / not relevant + reason + confidence
   - relevance criteria are explicit in the prompt (see pipeline.py)
        |
        v
AI Enrichment                enrich_leads()          <- batched Gemini calls,
   - profile, intent (+ level), need, objection (+ category/severity),     RELEVANT leads only
     urgency, engagement, missing info, opportunity, next action
        |
        v
Priority Scoring             compute_priority()      <- rules only, no AI
   - transparent, documented scoring formula (see below)
   - objections never lower the score; they steer the follow-up
        |
        v
Personalized Outreach        generate_outreach()     <- batched Gemini calls
   - one message per relevant lead, grounded in the original conversation;
     explicitly instructed not to promise jobs, placement, visas or salaries
        |
        v
Quality Control  ============================================
   |  Structured Validation   <- schema-checked Gemini output, retry +
   |                             salvage + conservative placeholder backfill
   |  Rule-Based QC           rule_based_qc()         <- rules only, no AI
   |                             contradiction detection, confidence
   |                             thresholding, missing-field and
   |                             duplicate-conflict flags, unsupported
   |                             claim checks, outreach hygiene, priority
   |                             formula recomputation
   |  AI Review               review_leads()          <- batched Gemini calls,
   |                             semantic second pass: does the generated
   |                             intelligence hold up against the ORIGINAL
   |                             lead conversation? returns pass/review,
   |                             issues, recommended action; never rewrites
   ============================================
        |
        v
Sales-Ready Lead Queue       (workspace UI, CSV export, or headless script)
```

Quality Control is conceptually one layer with three mechanisms inside it, Structured Validation, Rule-Based QC, and AI Review. In the running pipeline, QC and AI Review stream as two separate progress steps (see "all seven stages" below) since AI Review needs the QC flags as input; but neither modifies the underlying lead data, and both exist to serve the same goal: catching a problem before a human or a lead ever sees it.

Every stage lives in `pipeline.py` as a plain, independently testable
function. `app.py` is a thin Flask layer that exposes one endpoint per stage
plus two end-to-end entry points: `/api/pipeline/run` (single response, used
by the automation script) and `/api/pipeline/stream` (server-sent events
with real per-stage progress, used by the web UI). The UI additionally uses
`/api/outreach/regenerate`, `/api/workspace/result` + `/api/workspace/state`
(current-run state for lead-addressed regeneration and page-load hydration)
and `/api/export/csv` (server-side CSV built from the exact state the user
sees).

## AI workflow

All AI stages run on Gemini through one shared helper
(`_ask_for_json_array` in `pipeline.py`) and produce schema-constrained JSON:

- **Classification** — relevance verdict per lead (`relevant`, `reason`,
`confidence`). Missing fields lower confidence but never auto-reject.
- **Enrichment** — structured sales context: `profile`, `intent` (+
`intent_level`), `need`, `objection` (+ category/severity), `urgency`,
`engagement_level`, `missing_info`, `opportunity`, `next_action`.
Evidence-based: anything not supported by the conversation is "Not
stated", and intent/urgency/engagement may not be inferred from education,
profession, experience or German level alone.
- **Personalized outreach** — one grounded message per relevant lead, written
from the original conversation (the source of truth), addressing the
lead's actual concern and barred from inventing features, prices,
eligibility, timelines or outcomes.
- **AI Review** — a second-pass semantic reviewer that receives the original
lead conversation plus the classification, enrichment, priority, outreach
and rule-based QC flags, and checks grounding, classification consistency,
enrichment support, priority sanity and outreach claims. It returns
`status` (pass/review), `confidence`, specific `issues` and a
`recommended_action`. It is a reviewer only: it never rewrites enrichment,
outreach, classification or priority, and never invents missing
information.

Enrichment, outreach and AI review run only for leads classified relevant,
so no API time is spent on leads that will never be contacted.

## Prioritization

Priority is a transparent, rule-based score (0–100), not an AI judgment:

```
Priority = 40% Intent + 25% Urgency + 20% Fit + 15% Engagement
```

- **Intent (40%)** — AI-assessed intent level (high/medium/low/unclear),
grounded in what the lead actually said.
- **Urgency (25%)** — AI-assessed urgency, evidence-based.
- **Fit (20%)** — deterministic, from lead attributes: German proficiency
and relevant experience (missing information is treated as unknown rather
than penalizing the lead).
- **Engagement (15%)** — AI-assessed engagement level from the interaction.

Bands: **High** (score ≥ 80), **Medium** (≥ 50), **Low** (< 50). Not-relevant
leads are not scored.

This is a **sales follow-up heuristic, not a conversion prediction**. The UI
shows the weighted component breakdown next to every total score. Objections
never reduce the score; they shape how the follow-up is approached.

## Quality control

Quality Control combines three mechanisms; a lead lands in the human-review
queue only if one of them finds a real problem.

1. **Structured-output validation** — every Gemini call requests strict JSON
  (schema first, plain JSON mode as fallback); malformed or truncated
   output is retried with backoff, salvaged, or backfilled with a
   conservative placeholder and flagged, so no lead ever silently vanishes
   (see `pipeline.py`).
2. **Rule-based QC** (`rule_based_qc`, deterministic and independent of the
  AI calls that produced the data it checks):
  - contradiction detection (classified relevant but the reason reads
  disqualifying)
  - low-confidence routing (classification confidence < 0.6 → human review)
  - missing/malformed classification or enrichment → human review
  - missing-field flags (e.g. no experience on record before contact)
  - duplicate-conflict flags
  - unsupported-claim checks on outreach (guarantee/placement/salary/visa
  language), plus hygiene checks (generic filler, length, question count)
  - the priority score and band are recomputed from the formula and
  cross-checked against the recorded values
3. **AI Review** (semantic, see above) — catches what word-matching rules
  can't: invented facts, unsupported enrichment claims, outreach that
   ignores the lead's stated concern. Rule-QC flags are provided to the
   reviewer as context, but a rule flag alone does not automatically mark a
   lead for review — the reviewer assesses whether the issue is meaningful.

Leads with genuine uncertainty are routed to the **Needs Human Review** queue
in the UI rather than silently accepted.

Three concrete cases the pipeline handles on this dataset:

- **L001 / L008 / L028** — same phone number. L008 is an exact duplicate and
gets auto-merged into L001. L028 shares the phone/email but has conflicting
details, so it is kept out of the active batch and surfaced in Needs Human
Review as a flagged duplicate pending manual reconciliation — never merged
automatically, and never forced through classification/enrichment/scoring
just to make the data complete.
- **L007** — missing experience entirely. The classifier still returns a
judgment, but the QC layer flags the record as needing that field before
contact rather than pretending it's complete.
- **L005** — a BBA graduate asking whether people from non-nursing
backgrounds can get nursing jobs in Germany. The classifier returns a
not-relevant verdict for the nursing-focused program (with its missing
German level flagged) — the kind of judgment call the pipeline handles
conservatively and makes visible rather than burying.



## The workspace (web UI)

A single-page workspace on top of the pipeline:

- **Load data** — upload a CSV of leads or use the bundled 30-lead sample,
then run the pipeline with live streamed progress for all seven stages.
- **Four tabs** — *All Leads*, *Needs Human Review*, *Contacted Leads*,
*Removed Leads*, each with a live count. All Leads shows relevant,
uncontacted leads that passed QC; Needs Human Review collects rule-QC
flags, AI-review flags and flagged conflicting duplicates; Removed shows
merged duplicates and not-relevant leads with their reasons.
- **Search, filter, sort** — search by lead ID/name/city, filter by priority
band, sort high→low or low→high.
- **Expandable lead detail** — profile facts, career goal, current need,
objection, missing information, opportunity, recommended next step, the
weighted priority breakdown with total score, and the outreach draft.
- **Personalized outreach** — editable draft with copy-to-clipboard and
**Regenerate**, which re-generates just that lead's message (no full
pipeline re-run), re-runs rule-based QC and AI Review for that lead, and
preserves the old draft if generation fails.
- **Contact workflow** — mark a lead as Contacted (it glides out of All
Leads into Contacted Leads) and undo it from the Contacted tab; removal
never touches classification, enrichment, priority or outreach.
- **Human-review handling** — rule flags appear as a review note in the lead
detail; AI Review surfaces a concise "Review needed" warning only when it
flags an issue (passing reviews stay silent — it's a backend QC mechanism,
not a dashboard metric). Flagged duplicates like L028 are shown with their
conflict details for manual reconciliation.
- **CSV export** — Download CSV sends the *current* workspace state (edited
outreach, contacted/removed status included) to `/api/export/csv` and
downloads the generated file, so the export always matches what the user
sees. On reload, the last completed run is restored automatically for the
current server session.
- A collapsed "What the pipeline flagged" panel below the list explains the
run's notable findings (duplicate clusters, missing fields, lowest
confidence, self-corrected claims, AI-review flags).



## Data & output

The workflow processes the provided 30-lead dataset (`data/leads.json`; 30
raw records → 27 primary records after deduplication) and produces:

- structured lead intelligence per lead (classification, enrichment,
priority with component breakdown, outreach, QC flags, AI-review verdict)
- a final CSV (`skillcase_lead_manifest.csv`) with one row per record —
active, contacted, needs-review, removed, merged and flagged duplicates —
including AI Review findings
- run metadata (counts per stage, duplicates merged/flagged, runtime)

The same result is reachable three ways: the web UI, `POST /api/pipeline/run`
(single JSON response), or the headless script below.

## Automation

`scripts/run_and_export.py` runs the entire pipeline with no UI and writes a
CSV. This is the piece meant to run unattended — on a cron schedule, in CI,
or triggered by a webhook when a new lead lands in a sheet or CRM. It shares
the exact same `pipeline.py` used by the web app, so results are identical
either way:

```bash
GEMINI_API_KEY=your-key python scripts/run_and_export.py --out today.csv
```

Example cron entry (every morning at 7am):

```
0 7 * * * cd /path/to/skillcase-app && python3 scripts/run_and_export.py --out exports/$(date +%F).csv
```



## Gemini robustness

All AI calls (classification, enrichment, outreach, AI review) go through
one helper (`_ask_for_json_array` in `pipeline.py`) that provides:

- **JSON response mode with a schema**, falling back automatically to
plain JSON mode if the model rejects `response_json_schema` (400).
- **Batched requests** (classification 15 leads/call, enrichment 8, outreach
6, AI review 8) with bounded concurrency, so no single response is large
enough to hit the max-token ceiling and each call stays fast.
- **Exponential backoff with jitter** on 429/500/502/503/504, network
timeouts and malformed output, honoring the server's `retryDelay`
hint on 429s (4 attempts max, capped wait).
- **Truncation salvage**: a response cut off mid-array keeps its complete
records instead of being discarded.
- **Echo validation**: every stage checks that the model returned every lead
id; a skipped lead gets a conservative placeholder and a QC flag rather
than silently vanishing.
- **Explicit timeouts** and automatic function calling (AFC) disabled
(this pipeline never uses function calling).



## Local setup

Requires Python 3.10+ and a Gemini API key
([Google AI Studio](https://aistudio.google.com/)).

```bash
cd skillcase-app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then edit .env and add your GEMINI_API_KEY
python app.py
```

Open [http://localhost:5000](http://localhost:5000), pick the sample dataset or upload a CSV, and
click **Run pipeline**. `PORT` is configurable via `.env` (default 5000);
optional overrides `GEMINI_MODEL`, `GEMINI_TIMEOUT_MS` and
`GEMINI_RETRY_BASE_DELAY` are documented in `.env.example`.

## Testing

The test suite runs fully offline — every Gemini call is mocked; no test
depends on the live API or a real key.

```bash
pytest                       # or: .venv/bin/python -m pytest tests/ -q
```

Two suites:

- `tests/test_pipeline.py` — priority scoring (formula, components, bands,
confidence exclusion), dedupe accounting for all 30 leads, QC rules
(contradictions, confidence thresholds, claim/hygiene checks, formula
cross-check), messy-field safety, Gemini-layer robustness (retries,
malformed output, schema fallback, truncation salvage), batching and
echo-validation placeholders, the AI Review stage (pass/review results,
normalization of malformed model output, placeholder behavior,
never-rewrites guarantee), and a full offline 30-lead pipeline run.
- `tests/test_app.py` — Flask route/integration tests via the test client:
the regenerate request shape actually sent by the UI (`{"lead_id": ...}`),
invalid-payload JSON errors, CSV export built from posted current state,
malformed-payload handling, workspace state round-trips, and the AI-review
UI surfacing policy (pass silent, flagged surfaced without confidence,
flagged leads entering the review queue).

Run the pipeline headlessly against the live API any time with:

```bash
GEMINI_API_KEY=your-key python scripts/run_and_export.py --out today.csv
```



## Deploying so anyone can use it (no account needed)

**Render (free tier, ~10 minutes):**

1. Push this folder to a GitHub repo.
2. On [render.com](https://render.com), New -> Web Service -> connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Add an environment variable `GEMINI_API_KEY` with your key.
6. Deploy. Render gives you a public URL — share that link.

Railway, Fly.io, or PythonAnywhere work the same way: install dependencies,
run `gunicorn app:app`, set the one environment variable.

**Cost/abuse note:** because the API key lives on the server, every
visitor's pipeline run is billed to that key. Fine for a demo shared with a
small number of reviewers; for anything wider, add a rate limiter (e.g.
`flask-limiter`) before sharing broadly.

## Design decisions worth knowing for the review

- **Prioritization is rule-based, not AI-scored.** A visible formula is
easier to defend and debug than a model-generated number, and the brief
explicitly asks for explainable criteria.
- **Cleaning and QC never depend on the AI calls succeeding.** They're pure
functions over the raw/cleaned data, so a bad Gemini response degrades
gracefully rather than corrupting the whole pipeline.
- **AI Review identifies problems; it never fixes them.** The reviewer
returns pass/review with specific issues and a recommended action, and the
human resolves review cases — no silent rewrites of generated data.
- **The CSV exports what the user sees, not a cached run.** Exported
outreach, contacted status and review state are captured at click time.
- **Batched, not per-lead, AI calls.** Each AI stage runs as a handful of
concurrent batched calls rather than 30 separate requests — cheaper,
faster, and it's what "automation" should look like at this scale.



## Repository structure

```
skillcase-app/
├── app.py                  # Flask app: UI, per-stage APIs, workspace state,
│                           #   outreach regeneration, CSV export
├── pipeline.py             # All pipeline stages: clean/dedupe, classification,
│                           #   enrichment, priority scoring, outreach,
│                           #   rule-based QC, AI review + Gemini infra
├── data/
│   └── leads.json          # 30 raw lead conversations (the sample dataset)
├── static/
│   ├── index.html          # Single-page workspace UI
│   └── Logo.png            # Logo asset used across the UI
├── scripts/
│   └── run_and_export.py   # Headless end-to-end run + CSV export (cron/CI)
├── tests/
│   ├── test_pipeline.py    # Pipeline unit tests (Gemini mocked)
│   └── test_app.py         # Flask route/integration tests (AI mocked)
├── requirements.txt        # flask, google-genai, python-dotenv, gunicorn
├── .env.example            # GEMINI_API_KEY / PORT template
└── README.md
```

