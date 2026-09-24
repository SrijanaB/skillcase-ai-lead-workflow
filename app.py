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
