import json
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

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


@app.route("/api/pipeline/clean", methods=["POST"])
def api_clean():
    body = request.get_json(silent=True) or {}
    leads = body.get("leads") or RAW_LEADS
    return jsonify(pipeline.clean_and_dedupe(leads))


@app.route("/api/pipeline/classify", methods=["POST"])
def api_classify():
    leads = request.json["leads"]
    return jsonify(pipeline.classify_leads(leads))


@app.route("/api/pipeline/enrich", methods=["POST"])
def api_enrich():
    leads = request.json["leads"]
    return jsonify(pipeline.enrich_leads(leads))


@app.route("/api/pipeline/prioritize", methods=["POST"])
def api_prioritize():
    data = request.json
    leads, classifications, enrichments = data["leads"], data["classifications"], data["enrichments"]
    result = {}
    for lead in leads:
        cls = classifications.get(lead["lead_id"], {"relevant": False})
        enr = enrichments.get(lead["lead_id"], {})
        result[lead["lead_id"]] = pipeline.compute_priority(lead, cls, enr)
    return jsonify(result)


@app.route("/api/pipeline/outreach", methods=["POST"])
def api_outreach():
    items = request.json["items"]
    return jsonify(pipeline.generate_outreach(items))


@app.route("/api/pipeline/qc", methods=["POST"])
def api_qc():
    data = request.json
    leads, classifications = data["leads"], data["classifications"]
    enrichments, outreach = data["enrichments"], data["outreach"]
    result = {}
    for lead in leads:
        cls = classifications.get(lead["lead_id"], {})
        enr = enrichments.get(lead["lead_id"], {})
        result[lead["lead_id"]] = pipeline.rule_based_qc(
            lead, cls, enr, outreach.get(lead["lead_id"]), lead.get("priority")
        )
    return jsonify(result)


@app.route("/api/pipeline/run", methods=["POST"])
def api_run_full():
    """Runs every stage in one call, on either the uploaded leads or the
    bundled sample dataset. Used as a single-shot entry point and by
    scripts/run_and_export.py for scheduled/automated runs."""
    body = request.get_json(silent=True) or {}
    leads = body.get("leads") or RAW_LEADS
    return jsonify(pipeline.run_full_pipeline(leads))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
