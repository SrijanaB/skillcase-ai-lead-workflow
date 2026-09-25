import csv
import io
import json
import os
import queue

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

load_dotenv()

import pipeline  # noqa: E402  (import after load_dotenv so GEMINI_API_KEY is set)

app = Flask(__name__, static_folder="static", static_url_path="")

LEADS_PATH = os.path.join(os.path.dirname(__file__), "data", "leads.json")
with open(LEADS_PATH) as f:
    RAW_LEADS = json.load(f)

# ---------------------------------------------------------------------------
# Current-run state (in memory, process lifetime)
# ---------------------------------------------------------------------------
# The pipeline streams its result to the browser, which POSTs the final
# payload back here (/api/workspace/result). This is the single source of
# truth for lead_id-addressed calls (regenerate) so a lead is resolved exactly
# as the UI shows it. In-memory on purpose: no persistence infrastructure is
# introduced for this.
_CURRENT_RESULT: dict = {
    "primaries": [],
    "duplicate_groups": [],
    "lead_audit": [],
    "classifications": {},
    "enrichments": {},
    "outreach": {},
    "qc": {},
}


def _store_pipeline_result(result: dict) -> None:
    """Record the latest completed pipeline run as the current state."""
    _CURRENT_RESULT["primaries"] = result.get("primaries") or []
    _CURRENT_RESULT["duplicate_groups"] = result.get("duplicate_groups") or []
    _CURRENT_RESULT["lead_audit"] = result.get("lead_audit") or []
    _CURRENT_RESULT["classifications"] = result.get("classifications") or {}
    _CURRENT_RESULT["enrichments"] = result.get("enrichments") or {}
    _CURRENT_RESULT["outreach"] = result.get("outreach") or {}
    _CURRENT_RESULT["qc"] = result.get("qc") or {}


def _find_current_lead(lead_id: str) -> dict | None:
    """Resolve a lead_id against the current state (latest run's primaries),
    falling back to the bundled dataset so regeneration always has the lead's
    original conversation as its source of truth."""
    for p in _CURRENT_RESULT.get("primaries") or []:
        if isinstance(p, dict) and p.get("lead_id") == lead_id:
            return p
    for p in RAW_LEADS:
        if isinstance(p, dict) and p.get("lead_id") == lead_id:
            return p
    return None


@app.route("/api/workspace/result", methods=["POST"])
def api_workspace_result():
    """Store the completed pipeline result as the server's current state.

    The browser POSTs the exact final payload it just rendered, so later
    lead_id-addressed calls operate on what the user actually sees.
    """
    try:
        body = request.get_json(silent=True)
        result = body.get("result") if isinstance(body, dict) else None
        if not isinstance(result, dict) or not isinstance(result.get("primaries"), list):
            raise ValueError('Request body must be {"result": {...pipeline result...}} '
                             "with a 'primaries' list.")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    _store_pipeline_result(result)
    return jsonify({"ok": True, "leads": len(_CURRENT_RESULT["primaries"])})


@app.route("/api/workspace/state", methods=["GET"])
def api_workspace_state():
    """Current pipeline result, used by the UI to hydrate on page load."""
    if not _CURRENT_RESULT["primaries"]:
        return "", 204  # no completed run yet: the UI stays on the upload screen
    return jsonify({"result": _CURRENT_RESULT})


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/leads")
def get_leads():
    return jsonify(RAW_LEADS)


def _get_leads_from_request():
    body = request.get_json(silent=True)
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object.")
    leads = body.get("leads") or RAW_LEADS
    if not isinstance(leads, list) or not all(isinstance(l, dict) for l in leads):
        raise ValueError("'leads' must be a list of lead objects.")
    return leads


@app.errorhandler(400)
def handle_400(err):
    return jsonify({"error": getattr(err, "description", "Bad request")}), 400


@app.errorhandler(404)
def handle_404(err):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found"}), 404
    return err


@app.errorhandler(405)
def handle_405(err):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(Exception)
def handle_unexpected(err):
    # Surface the API key redaction + a JSON error instead of an HTML page,
    # so the frontend can show a useful message instead of appearing stuck.
    return jsonify({"error": pipeline._safe_api_error_message(err)}), 500


def _error_response(err: Exception):
    """(body, status) for a pipeline failure."""
    if isinstance(err, (RuntimeError, ValueError)):
        return {"error": pipeline._safe_api_error_message(err)}, 502
    return {"error": pipeline._safe_api_error_message(err)}, 500


@app.route("/api/pipeline/clean", methods=["POST"])
def api_clean():
    try:
        leads = _get_leads_from_request()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        return jsonify(pipeline.clean_and_dedupe(leads))
    except (KeyError, TypeError, AttributeError) as exc:
        return jsonify({"error": f"Cleaning failed: {exc}"}), 500


def _request_json(*keys):
    """Fetch required JSON body keys, raising ValueError with a clear
    message instead of a bare KeyError that becomes an HTML 400."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")
    return [data[k] for k in keys]


@app.route("/api/pipeline/classify", methods=["POST"])
def api_classify():
    try:
        (leads,) = _request_json("leads")
        if not isinstance(leads, list):
            raise ValueError("'leads' must be a list.")
        return jsonify(pipeline.classify_leads(leads))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        body, status = _error_response(exc)
        return jsonify(body), status


@app.route("/api/pipeline/enrich", methods=["POST"])
def api_enrich():
    try:
        (leads,) = _request_json("leads")
        if not isinstance(leads, list):
            raise ValueError("'leads' must be a list.")
        return jsonify(pipeline.enrich_leads(leads))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        body, status = _error_response(exc)
        return jsonify(body), status


@app.route("/api/pipeline/prioritize", methods=["POST"])
def api_prioritize():
    try:
        leads, classifications, enrichments = _request_json(
            "leads", "classifications", "enrichments"
        )
        result = {}
        for lead in leads:
            lid = lead.get("lead_id") if isinstance(lead, dict) else None
            if not lid:
                continue
            cls = classifications.get(lid, {"relevant": False})
            enr = enrichments.get(lid, {})
            result[lid] = pipeline.compute_priority(lead, cls, enr)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except (KeyError, AttributeError, TypeError) as exc:
        return jsonify({"error": f"Prioritization failed: {exc}"}), 500


@app.route("/api/pipeline/outreach", methods=["POST"])
def api_outreach():
    try:
        (items,) = _request_json("items")
        if not isinstance(items, list):
            raise ValueError("'items' must be a list.")
        return jsonify(pipeline.generate_outreach(items))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        body, status = _error_response(exc)
        return jsonify(body), status


@app.route("/api/pipeline/qc", methods=["POST"])
def api_qc():
    try:
        leads, classifications, enrichments, outreach = _request_json(
            "leads", "classifications", "enrichments", "outreach"
        )
        result = {}
        for lead in leads:
            lid = lead.get("lead_id") if isinstance(lead, dict) else None
            if not lid:
                continue
            cls = classifications.get(lid, {})
            enr = enrichments.get(lid, {})
            result[lid] = pipeline.rule_based_qc(
                lead, cls, enr, outreach.get(lid), lead.get("priority")
            )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except (KeyError, AttributeError, TypeError) as exc:
        return jsonify({"error": f"QC failed: {exc}"}), 500


@app.route("/api/outreach/regenerate", methods=["POST"])
def api_outreach_regenerate():
    """Regenerate outreach for one lead, addressed by lead_id only.

    The frontend sends {"lead_id": "L001"}; the server resolves that id
    against the current application state (the last completed pipeline run
    posted to /api/workspace/result), builds the same lead + enrichment
    context the pipeline uses, and reuses pipeline.generate_outreach for a
    genuinely fresh, grounded message. Only that lead's outreach changes —
    classification, enrichment, priority and QC are untouched and the full
    pipeline is NOT re-run. The new draft is written back into the current
    state so a subsequent export includes it. The frontend re-runs
    /api/pipeline/qc on the updated draft so QC flags stay honest.
    """
    try:
        (lead_id,) = _request_json("lead_id")
        if not isinstance(lead_id, str) or not lead_id.strip():
            raise ValueError("'lead_id' must be a non-empty string.")
        lead_id = lead_id.strip()
        lead = _find_current_lead(lead_id)
        if lead is None:
            raise KeyError(f"Unknown lead_id: {lead_id}")
        enrichment = _CURRENT_RESULT.get("enrichments", {}).get(lead_id) or {}
        items = [{"lead": lead, "enrichment": enrichment}]
        result = pipeline.generate_outreach(items)
        new_message = result.get(lead_id, "")
        if not new_message:
            raise RuntimeError("Outreach generation returned no message for this lead.")
        _CURRENT_RESULT.setdefault("outreach", {})[lead_id] = new_message
        return jsonify({"lead_id": lead_id, "outreach": new_message})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except KeyError as exc:
        return jsonify({"error": str(exc.args[0])}), 404
    except Exception as exc:
        body, status = _error_response(exc)
        return jsonify(body), status


# ---------------------------------------------------------------------------
# CSV export from the UI's current state
# ---------------------------------------------------------------------------

CSV_EXPORT_HEADERS = [
    "Lead", "Name", "Relevant", "Reason", "Confidence", "Urgency", "Intent",
    "Profile", "Need", "Objection", "Objection Severity", "Missing Information",
    "Priority", "Priority Reason", "Next Action", "Outreach", "QC Flags",
    "Status", "Primary Lead",
]


def _validated_export_state() -> dict:
    """Validate the current-state payload POSTed by the UI."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object.")
    if "state" not in body:
        raise ValueError("Missing required field(s): state")
    state = body["state"]
    if not isinstance(state, dict):
        raise ValueError("'state' must be an object of current application state.")
    if "primaries" in state and not (
        isinstance(state["primaries"], list)
        and all(isinstance(p, dict) for p in state["primaries"])
    ):
        raise ValueError("'primaries' must be a list of lead objects.")
    if "lead_audit" in state and not (
        isinstance(state["lead_audit"], list)
        and all(isinstance(a, dict) for a in state["lead_audit"])
    ):
        raise ValueError("'lead_audit' must be a list of audit records.")
    for key in ("classifications", "enrichments", "outreach", "qc", "contacted"):
        if key in state and not isinstance(state[key], dict):
            raise ValueError(f"'{key}' must be an object keyed by lead_id.")
    return state


def _export_rows(state: dict) -> list[list]:
    """Flatten the supplied current state into CSV rows.

    Mirrors the UI's tab semantics: primaries become Active / Contacted /
    Needs Human Review / Removed rows, merged audit records become removed
    duplicates, and flagged audit records become pending-reconciliation rows.
    """
    classifications = state.get("classifications") or {}
    enrichments = state.get("enrichments") or {}
    outreach = state.get("outreach") or {}
    qc = state.get("qc") or {}
    contacted = state.get("contacted") or {}
    audit_by_id = {
        a.get("lead_id"): a for a in (state.get("lead_audit") or []) if isinstance(a, dict)
    }

    rows: list[list] = []
    for p in state.get("primaries") or []:
        lead_id = p.get("lead_id")
        cls = classifications.get(lead_id) or {}
        enr = enrichments.get(lead_id) or {}
        pr = p.get("priority") or {}
        flags = qc.get(lead_id) or []
        audit = audit_by_id.get(lead_id)
        if cls.get("relevant") is False and not (audit and audit.get("requires_human_review")):
            status = "Removed — Not Relevant"
        elif lead_id in contacted:
            status = "Contacted"
        elif (audit and audit.get("requires_human_review")) or flags:
            status = "Needs Human Review"
        else:
            status = "Active"
        rows.append([
            lead_id, p.get("name"), "Yes" if cls.get("relevant") else "No",
            cls.get("reason"), cls.get("confidence"),
            enr.get("urgency"), enr.get("intent"), enr.get("profile"), enr.get("need"),
            enr.get("objection"), enr.get("objection_severity"), enr.get("missing_info"),
            pr.get("band") or "N/A", pr.get("priority_reason"), enr.get("next_action"),
            outreach.get(lead_id) or "", " | ".join(flags),
            status, "",
        ])

    for a in state.get("lead_audit") or []:
        status_flag = a.get("status")
        if status_flag == "merged":
            rows.append([
                a.get("lead_id"), a.get("name") or a.get("lead_id"), "",
                a.get("reason") or "Duplicate record", "", "", "", "", "", "", "", "",
                "N/A", "", "", "", "", "Removed — Duplicate (merged)", a.get("primary_id") or "",
            ])
        elif status_flag == "flagged":
            rows.append([
                a.get("lead_id"), "Conflicting duplicate record (flagged)", "",
                a.get("reason") or "", "", "", "", "", "", "", "", "",
                "N/A", "", "", "", "", "Flagged duplicate — pending reconciliation",
                a.get("primary_id") or "",
            ])
    return rows


@app.route("/api/export/csv", methods=["POST"])
def api_export_csv():
    """Generate the CSV from the exact state the UI currently shows.

    The frontend is the source of truth for the visible workspace (regenerated
    outreach, contacted and removed state live there), so it POSTs its current
    exportable state and receives a CSV built from it — nothing is read from
    data/leads.json, cached output or any previous run.
    """
    try:
        state = _validated_export_state()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n")
    writer.writerow(CSV_EXPORT_HEADERS)
    writer.writerows(_export_rows(state))
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=skillcase_lead_manifest.csv"},
    )


@app.route("/api/pipeline/run", methods=["POST"])
def api_run_full():
    """Runs every stage in one call, on either the uploaded leads or the
    bundled sample dataset. Used as a single-shot entry point and by
    scripts/run_and_export.py for scheduled/automated runs."""
    try:
        leads = _get_leads_from_request()
        return jsonify(pipeline.run_full_pipeline(leads))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        body, status = _error_response(exc)
        return jsonify(body), status


@app.route("/api/pipeline/stream", methods=["POST"])
def api_run_stream():
    """Runs every stage, streaming real per-stage progress as
    server-sent events so the UI reflects actual backend work."""
    try:
        leads = _get_leads_from_request()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    def generate():
        q: "queue.Queue[tuple[str, str] | None]" = queue.Queue()
        sentinel = object()

        def worker():
            try:
                for stage, status, message, payload in pipeline.run_full_pipeline_stream(leads):
                    q.put((stage, status, message, payload))
            except Exception as exc:  # noqa: BLE001 - forwarded to the client
                q.put(("error", "error", pipeline._safe_api_error_message(exc), None))
            finally:
                q.put(None)

        import threading
        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = q.get()
            if item is None:
                break
            stage, status, message, payload = item
            event = {"stage": stage, "status": status, "message": message}
            if payload is not None:
                event["payload"] = payload
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering (e.g. nginx)
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # threaded=True keeps health/UI requests responsive while the pipeline
    # endpoint is busy talking to Gemini.
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
