"""
Flask route/integration tests for app.py (the API boundary the pipeline unit
tests don't cover).

Covers:
  A. POST /api/outreach/regenerate with the exact frontend payload
     {"lead_id": "L001"} — AI generation mocked, only the requested lead's
     outreach changes, context is built from current server state.
  B. Invalid regenerate payloads return 4xx JSON (never an HTML error page).
  C. POST /api/export/csv reflects the exact current state POSTed by the UI
     (modified outreach, contacted/removed/flagged rows).
  D. Malformed CSV-export payloads return clean JSON errors.
  E. /api/workspace/result + /api/workspace/state hydration round-trip.
"""
import copy
import csv
import io
import json
import os
import re
import subprocess
import sys
import unittest

import pipeline  # noqa: E402  (after sys.path setup)
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402

# A small, deterministic stand-in for a completed pipeline run (the payload
# the UI would POST to /api/workspace/result after a real run).
RESULT_SEED = {
    "primaries": [
        {
            "lead_id": "L001", "name": "Asha Kapoor", "city": "Berlin",
            "education": "BSc Nursing", "experience_years": 3,
            "german_level": "A2", "goal": "Move from A2 to B1",
            "source": "Instagram", "conversation": "I want to start classes soon",
            "priority": {"band": "High", "score": 88, "priority_reason": "Strong intent"},
        },
        {
            "lead_id": "L002", "name": "Ravi Nair", "city": "Munich",
            "education": "BBA", "experience_years": 1,
            "german_level": "A1", "goal": "Learn German for work",
            "source": "Website", "conversation": "Just exploring options",
            "priority": {"band": "Medium", "score": 61, "priority_reason": "Exploratory"},
        },
        {
            "lead_id": "L003", "name": "Tomas Weber", "city": "Hamburg",
            "education": "MBA", "experience_years": 9,
            "german_level": "C1", "goal": "Already fluent",
            "source": "LinkedIn", "conversation": "Just browsing",
            "priority": {"band": "N/A", "score": None, "priority_reason": ""},
        },
    ],
    "duplicate_groups": [
        {"primary": "L001", "exact": ["L008"], "fuzzy": ["L028"], "all": ["L001", "L008", "L028"]},
    ],
    "lead_audit": [
        {"lead_id": "L008", "status": "merged", "primary_id": "L001",
         "reason": "Duplicate phone number", "requires_human_review": False},
        {"lead_id": "L028", "status": "flagged", "primary_id": "L001",
         "reason": "Shares phone with L001 but has conflicting details",
         "requires_human_review": True},
    ],
    "classifications": {
        "L001": {"relevant": True, "confidence": 0.92, "reason": "Matches ICP"},
        "L002": {"relevant": True, "confidence": 0.71, "reason": "Borderline intent"},
        "L003": {"relevant": False, "confidence": 0.31, "reason": "Already fluent, no need"},
    },
    "enrichments": {
        "L001": {"profile": "Nurse, 3 yrs", "intent": "high", "need": "B1 in 6 months",
                 "objection": "Price", "objection_severity": "medium", "urgency": "high",
                 "missing_info": "budget", "next_action": "Send B1 plan"},
        "L002": {"profile": "Graduate, 1 yr", "intent": "low", "need": "A1 course",
                 "objection": "", "objection_severity": "low", "urgency": "low",
                 "missing_info": "", "next_action": "Nurture"},
    },
    "outreach": {
        "L001": "ORIGINAL DRAFT FOR L001",
        "L002": "ORIGINAL DRAFT FOR L002",
    },
    "qc": {"L001": [], "L002": [], "L003": []},
    "ai_review": {
        "L001": {"lead_id": "L001", "status": "pass", "confidence": 0.96,
                 "issues": [], "recommended_action": "No action needed."},
        "L002": {"lead_id": "L002", "status": "review", "confidence": 0.62,
                 "issues": ["Urgency not supported by the conversation."],
                 "recommended_action": "Verify urgency before contacting."},
    },
}


def _reset_state(result=None):
    """Point the app's current state at a private copy of the seed (or empty)."""
    app_module._CURRENT_RESULT["primaries"] = []
    app_module._CURRENT_RESULT["duplicate_groups"] = []
    app_module._CURRENT_RESULT["lead_audit"] = []
    app_module._CURRENT_RESULT["classifications"] = {}
    app_module._CURRENT_RESULT["enrichments"] = {}
    app_module._CURRENT_RESULT["outreach"] = {}
    app_module._CURRENT_RESULT["qc"] = {}
    if result is not None:
        app_module._store_pipeline_result(copy.deepcopy(result))


class AppRouteTestBase(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    # -- helpers ------------------------------------------------------------

    def seed_state(self):
        _reset_state(RESULT_SEED)

    def post_json(self, url, payload):
        return self.client.post(
            url, data=json.dumps(payload), content_type="application/json"
        )


class TestRegenerate(AppRouteTestBase):
    """Test A + B: /api/outreach/regenerate contract."""

    def setUp(self):
        super().setUp()
        self.seed_state()

    def test_a_regenerate_accepts_frontend_payload_and_returns_outreach(self):
        """The frontend sends {"lead_id": "L001"} — that exact shape must work."""
        with mock.patch("pipeline.generate_outreach") as mock_gen, \
             mock.patch("pipeline.review_leads") as mock_review:
            mock_gen.return_value = {"L001": "REGENERATED DRAFT FOR L001"}
            mock_review.return_value = {"L001": {"lead_id": "L001", "status": "pass",
                                                 "confidence": 0.95, "issues": [],
                                                 "recommended_action": "No action needed."}}
            resp = self.post_json("/api/outreach/regenerate", {"lead_id": "L001"})

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.is_json)
        data = resp.get_json()
        self.assertEqual(data["lead_id"], "L001")
        self.assertEqual(data["outreach"], "REGENERATED DRAFT FOR L001")
        self.assertNotIn("Missing required field(s): lead", json.dumps(data))

        # The existing generation logic is reused, with the lead resolved from
        # current server state (not from the request) plus its enrichment.
        mock_gen.assert_called_once()
        items = mock_gen.call_args[0][0]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["lead"]["lead_id"], "L001")
        self.assertEqual(items[0]["lead"]["name"], "Asha Kapoor")
        self.assertEqual(items[0]["enrichment"]["urgency"], "high")

    def test_a_only_requested_lead_is_affected(self):
        with mock.patch("pipeline.generate_outreach") as mock_gen, \
             mock.patch("pipeline.review_leads") as mock_review:
            mock_gen.return_value = {"L001": "REGENERATED DRAFT FOR L001"}
            mock_review.return_value = {"L001": {"lead_id": "L001", "status": "pass",
                                                 "confidence": 0.9, "issues": [],
                                                 "recommended_action": "No action needed."}}
            resp = self.post_json("/api/outreach/regenerate", {"lead_id": "L001"})
        self.assertEqual(resp.status_code, 200)

        outreach = app_module._CURRENT_RESULT["outreach"]
        self.assertEqual(outreach["L001"], "REGENERATED DRAFT FOR L001")   # updated
        self.assertEqual(outreach["L002"], "ORIGINAL DRAFT FOR L002")      # untouched
        # No other part of the current state is modified.
        self.assertEqual(app_module._CURRENT_RESULT["classifications"],
                         RESULT_SEED["classifications"])
        self.assertEqual(app_module._CURRENT_RESULT["enrichments"],
                         RESULT_SEED["enrichments"])
        self.assertEqual(app_module._CURRENT_RESULT["primaries"],
                         RESULT_SEED["primaries"])
        # Rule QC is re-run for the regenerated lead (spec: rerun the existing
        # rule-based QC); other leads' QC is untouched.
        expected_flags = pipeline.rule_based_qc(
            RESULT_SEED["primaries"][0],
            RESULT_SEED["classifications"]["L001"],
            RESULT_SEED["enrichments"]["L001"],
            "REGENERATED DRAFT FOR L001",
            RESULT_SEED["primaries"][0].get("priority"),
        )
        self.assertEqual(app_module._CURRENT_RESULT["qc"]["L001"], expected_flags)
        self.assertEqual(app_module._CURRENT_RESULT["qc"]["L002"], RESULT_SEED["qc"]["L002"])
        self.assertEqual(app_module._CURRENT_RESULT["qc"]["L003"], RESULT_SEED["qc"]["L003"])
        # AI review is re-run for the regenerated lead only; other reviews untouched
        review = app_module._CURRENT_RESULT["ai_review"]["L001"]
        self.assertEqual(review["lead_id"], "L001")
        self.assertEqual(review["status"], "pass")
        self.assertEqual(app_module._CURRENT_RESULT["ai_review"]["L002"],
                         RESULT_SEED["ai_review"]["L002"])

    def test_b_empty_payload_is_4xx_json(self):
        resp = self.post_json("/api/outreach/regenerate", {})
        self.assertEqual(resp.status_code, 400)
        self.assertTrue(resp.is_json)
        data = resp.get_json()
        self.assertIn("error", data)
        self.assertIn("lead_id", data["error"])
        # A clean JSON error, not an HTML Flask error page.
        self.assertNotIn("<", resp.get_data(as_text=True))

    def test_b_non_object_body_is_4xx_json(self):
        resp = self.client.post(
            "/api/outreach/regenerate", data="not json", content_type="application/json"
        )
        self.assertEqual(resp.status_code, 400)
        self.assertTrue(resp.is_json)
        self.assertIn("error", resp.get_json())

    def test_b_unknown_lead_id_is_404_json(self):
        resp = self.post_json("/api/outreach/regenerate", {"lead_id": "L999"})
        self.assertEqual(resp.status_code, 404)
        self.assertTrue(resp.is_json)
        self.assertIn("L999", resp.get_json()["error"])

    def test_a_generation_failure_preserves_old_outreach(self):
        with mock.patch("pipeline.generate_outreach",
                        side_effect=RuntimeError("Gemini unavailable")):
            resp = self.post_json("/api/outreach/regenerate", {"lead_id": "L001"})
        self.assertEqual(resp.status_code, 502)
        self.assertTrue(resp.is_json)
        self.assertIn("error", resp.get_json())
        # The old draft must still be the current state after a failed attempt.
        self.assertEqual(app_module._CURRENT_RESULT["outreach"]["L001"],
                         "ORIGINAL DRAFT FOR L001")

    def test_a_regenerate_response_includes_qc_and_ai_review(self):
        """Regeneration re-runs rule QC + AI review for that lead only."""
        with mock.patch("pipeline.generate_outreach") as mock_gen, \
             mock.patch("pipeline.review_leads") as mock_review:
            mock_gen.return_value = {"L001": "BRAND NEW DRAFT"}
            mock_review.return_value = {"L001": {"lead_id": "L001", "status": "pass",
                                                 "confidence": 0.9, "issues": [],
                                                 "recommended_action": "No action needed."}}
            resp = self.post_json("/api/outreach/regenerate", {"lead_id": "L001"})

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["outreach"], "BRAND NEW DRAFT")
        self.assertIsInstance(data["qc_flags"], list)
        self.assertEqual(data["ai_review"]["lead_id"], "L001")
        self.assertIn(data["ai_review"]["status"], ("pass", "review"))
        mock_review.assert_called_once()
        # The review item for the regenerated lead uses the NEW outreach
        reviewed_item = mock_review.call_args[0][0][0]
        self.assertEqual(reviewed_item["outreach"], "BRAND NEW DRAFT")

    def test_a_regenerate_survives_ai_review_failure(self):
        """AI review is advisory: a review crash must not lose the new draft."""
        with mock.patch("pipeline.generate_outreach") as mock_gen, \
             mock.patch("pipeline.review_leads", side_effect=RuntimeError("review down")):
            mock_gen.return_value = {"L001": "DRAFT DESPITE REVIEW FAILURE"}
            resp = self.post_json("/api/outreach/regenerate", {"lead_id": "L001"})

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["outreach"], "DRAFT DESPITE REVIEW FAILURE")
        self.assertEqual(data["ai_review"]["status"], "review")  # conservative fallback
        self.assertEqual(app_module._CURRENT_RESULT["outreach"]["L001"],
                         "DRAFT DESPITE REVIEW FAILURE")


class TestCsvExport(AppRouteTestBase):
    """Test C + D: /api/export/csv reflects the posted current state."""

    def current_ui_state(self):
        """What the frontend would POST: its live STATE + contacted map,
        with a regenerated L001 outreach applied."""
        return {
            "primaries": copy.deepcopy(RESULT_SEED["primaries"]),
            "lead_audit": copy.deepcopy(RESULT_SEED["lead_audit"]),
            "classifications": copy.deepcopy(RESULT_SEED["classifications"]),
            "enrichments": copy.deepcopy(RESULT_SEED["enrichments"]),
            "outreach": {
                "L001": "UPDATED TEST OUTREACH",
                "L002": "ORIGINAL DRAFT FOR L002",
            },
            "qc": copy.deepcopy(RESULT_SEED["qc"]),
            "ai_review": copy.deepcopy(RESULT_SEED["ai_review"]),
            "contacted": {"L002": {"timestamp": "2026-09-25T10:00:00Z"}},
        }

    def export_csv(self, state):
        return self.post_json("/api/export/csv", {"state": state})

    def csv_rows(self, resp):
        return list(csv.reader(io.StringIO(resp.get_data(as_text=True))))

    def test_c_csv_success_content_type_and_headers(self):
        resp = self.export_csv(self.current_ui_state())
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.get_data(as_text=True).startswith("Lead,Name"))
        self.assertEqual(resp.headers["Content-Disposition"],
                         "attachment; filename=skillcase_lead_manifest.csv")
        rows = self.csv_rows(resp)
        self.assertEqual(rows[0], app_module.CSV_EXPORT_HEADERS)

    def test_c_csv_contains_lead_data_and_modified_outreach(self):
        resp = self.export_csv(self.current_ui_state())
        rows = self.csv_rows(resp)
        by_id = {r[0]: r for r in rows[1:]}

        # The regenerated outreach from the posted state appears verbatim.
        self.assertIn("L001", by_id)
        self.assertEqual(by_id["L001"][15], "UPDATED TEST OUTREACH")
        self.assertEqual(by_id["L001"][1], "Asha Kapoor")
        self.assertEqual(by_id["L001"][2], "Yes")          # Relevant
        self.assertEqual(by_id["L001"][12], "High")        # Priority band
        # The old cached draft must NOT appear anywhere.
        self.assertNotIn("ORIGINAL DRAFT FOR L001", resp.get_data(as_text=True))

    def test_c_csv_includes_ai_review_column(self):
        resp = self.export_csv(self.current_ui_state())
        self.assertIn("AI Review", app_module.CSV_EXPORT_HEADERS)
        rows = self.csv_rows(resp)
        by_id = {r[0]: r for r in rows[1:]}

        # L001 passes: clean AI Review cell. L002 is flagged: its issues appear.
        self.assertEqual(by_id["L001"][17], "Pass")
        self.assertIn("Urgency not supported", by_id["L002"][17])

        # L003 is not-relevant, so Removed takes precedence (mirrors the UI:
        # removed leads never enter Needs Human Review). Prove that an
        # AI-review flag ALONE puts a clean relevant lead (L001) into
        # Needs Human Review.
        state = self.current_ui_state()
        state["ai_review"]["L001"] = {"lead_id": "L001", "status": "review",
                                      "confidence": 0.55,
                                      "issues": ["Intent not grounded."],
                                      "recommended_action": "Verify intent."}
        resp3 = self.export_csv(state)
        by_id3 = {r[0]: r for r in self.csv_rows(resp3)[1:]}
        self.assertEqual(by_id3["L001"][18], "Needs Human Review")

    def test_c_csv_reflects_contacted_removed_and_flagged_state(self):
        resp = self.export_csv(self.current_ui_state())
        rows = self.csv_rows(resp)
        by_id = {r[0]: r for r in rows[1:]}

        # Status column: contacted, removed, active
        self.assertEqual(by_id["L002"][18], "Contacted")
        self.assertEqual(by_id["L003"][18], "Removed — Not Relevant")
        self.assertEqual(by_id["L001"][18], "Active")

        # Merged duplicate and flagged duplicate audit records are represented.
        self.assertEqual(by_id["L008"][17], "Removed — Duplicate (merged)")
        self.assertEqual(by_id["L008"][18], "L001")        # Primary Lead
        self.assertIn("Flagged duplicate", by_id["L028"][17])
        self.assertEqual(by_id["L028"][18], "L001")

    def test_d_malformed_payload_returns_json_error(self):
        cases = [
            None,                                            # no body at all
            "not json",                                      # unparseable
            {},                                              # missing "state"
            {"state": "nope"},                               # wrong type
            {"state": {"primaries": "nope"}},                # wrong primaries type
            {"state": {"outreach": []}},                     # wrong outreach type
        ]
        for payload in cases:
            if payload is None:
                resp = self.client.post("/api/export/csv")
            elif isinstance(payload, str):
                resp = self.client.post("/api/export/csv", data=payload,
                                        content_type="application/json")
            else:
                resp = self.post_json("/api/export/csv", payload)
            self.assertEqual(resp.status_code, 400, f"payload={payload!r}")
            self.assertTrue(resp.is_json, f"payload={payload!r}")
            self.assertIn("error", resp.get_json(), f"payload={payload!r}")
            self.assertNotIn("<html", resp.get_data(as_text=True).lower())

    def test_d_empty_state_yields_header_only_csv(self):
        resp = self.export_csv({"primaries": [], "lead_audit": []})
        self.assertEqual(resp.status_code, 200)
        rows = self.csv_rows(resp)
        self.assertEqual(rows[0], app_module.CSV_EXPORT_HEADERS)
        self.assertEqual(len(rows), 1)


class TestWorkspaceState(AppRouteTestBase):
    """Test E: hydration endpoints (/api/workspace/result, /api/workspace/state)."""

    def test_e_store_then_hydrate_round_trip(self):
        resp = self.post_json("/api/workspace/result", {"result": RESULT_SEED})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.is_json)
        self.assertTrue(resp.get_json()["ok"])

        resp = self.client.get("/api/workspace/state")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.is_json)
        result = resp.get_json()["result"]

        # Required state fields exist and all records are represented per the
        # current tab semantics: primaries drive All/Contacted; the audit
        # carries the merged duplicate (Removed) and the flagged duplicate
        # (Needs Human Review).
        for key in ("primaries", "lead_audit", "duplicate_groups",
                    "classifications", "enrichments", "outreach", "qc", "ai_review"):
            self.assertIn(key, result)
        self.assertEqual(len(result["primaries"]), 3)
        self.assertIn("L001", result["outreach"])
        self.assertEqual(result["ai_review"]["L001"]["status"], "pass")
        self.assertEqual(result["ai_review"]["L002"]["status"], "review")
        statuses = {a["lead_id"]: a["status"] for a in result["lead_audit"]}
        self.assertEqual(statuses, {"L008": "merged", "L028": "flagged"})

    def test_e_malformed_result_payload_is_400_json(self):
        resp = self.post_json("/api/workspace/result", {"result": {"no": "primaries"}})
        self.assertEqual(resp.status_code, 400)
        self.assertTrue(resp.is_json)
        self.assertIn("error", resp.get_json())

    def test_e_no_completed_run_returns_204(self):
        _reset_state(None)
        resp = self.client.get("/api/workspace/state")
        self.assertEqual(resp.status_code, 204)
        self.assertEqual(resp.get_data(as_text=True), "")


class TestAiReviewSurfacingPolicy(AppRouteTestBase):
    """UI surfacing policy: AI Review is a backend QC mechanism.

    PASS results are stored but never surfaced; only 'review' results are
    surfaced (concise warning, no confidence score). These tests pin the
    client-side rendering contract implemented by aiReviewHTML() in
    static/index.html.
    """

    PASS_REVIEW = {"lead_id": "L001", "status": "pass", "confidence": 0.98,
                   "issues": [], "recommended_action": "No action needed."}
    FLAG_REVIEW = {"lead_id": "L007", "status": "review", "confidence": 0.89,
                   "issues": ["Outreach introduces a timeline that was not stated by the lead."],
                   "recommended_action": "Review outreach before sending."}

    def render_ai_block(self, review):
        """Evaluate aiReviewHTML() against a JS STATE with one lead (L001)."""
        html_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "static", "index.html",
        )
        with open(html_path) as f:
            page = f.read()
        m = re.search(r"function aiReviewHTML\(p\) \{.*?\n\}", page, re.S)
        self.assertTrue(m, "aiReviewHTML() not found in static/index.html")
        state = json.dumps({"ai_review": {"L001": review}})
        script = (
            "const STATE = " + state + ";"
            "const p = { lead_id: 'L001' };"
            "const escapeHTML = (v) => String(v ?? '').replace(/&/g, '&amp;');"
            + m.group(0) +
            "; console.log(aiReviewHTML(p))"
        )
        proc = subprocess.run(
            ["node", "-e", script], capture_output=True, text=True, timeout=10
        )
        if proc.returncode != 0:
            self.fail(f"aiReviewHTML eval failed: {proc.stderr}")
        return proc.stdout.strip()

    def test_pass_review_is_not_surfaced_and_backend_still_stores_it(self):
        # UI: PASS renders nothing at all (no block, no confidence, no
        # "No action needed", no empty container).
        self.assertEqual(self.render_ai_block(self.PASS_REVIEW), "")

        # Backend: the result is still stored for QC/audit purposes.
        self.post_json("/api/workspace/result", {"result": RESULT_SEED})
        state = self.client.get("/api/workspace/state").get_json()["result"]
        self.assertEqual(state["ai_review"]["L001"], RESULT_SEED["ai_review"]["L001"])
        self.assertEqual(state["ai_review"]["L001"]["status"], "pass")

    def test_flagged_review_is_surfaced_without_confidence(self):
        html = self.render_ai_block(self.FLAG_REVIEW)

        # Surfaced: concise warning + the specific issue + action.
        self.assertIn("Review needed", html)
        self.assertIn("Outreach introduces a timeline", html)
        self.assertIn("Review outreach before sending.", html)

        # Never surfaced: the AI Review label, status word or confidence.
        self.assertNotIn("AI Review", html)
        self.assertNotIn("PASS", html)
        self.assertNotIn("REVIEW", html.replace("Review needed", ""))
        self.assertNotIn("89", html)  # confidence percentage hidden

        # Backend: flagged result stored too.
        self.post_json("/api/workspace/result", {"result": RESULT_SEED})
        state = self.client.get("/api/workspace/state").get_json()["result"]
        self.assertEqual(state["ai_review"]["L002"]["status"], "review")

    def test_flagged_review_moves_lead_into_needs_human_review(self):
        """An AI-review flag feeds the existing human-review queue."""
        self.post_json("/api/workspace/result", {"result": RESULT_SEED})
        state = self.client.get("/api/workspace/state").get_json()["result"]
        # L002 carries a 'review' AI verdict in the seed.
        self.assertEqual(state["ai_review"]["L002"]["status"], "review")
        # ...and its export status is Needs Human Review (queue membership).
        resp = self.post_json("/api/export/csv", {"state": {
            "primaries": RESULT_SEED["primaries"],
            "lead_audit": RESULT_SEED["lead_audit"],
            "classifications": RESULT_SEED["classifications"],
            "enrichments": RESULT_SEED["enrichments"],
            "outreach": RESULT_SEED["outreach"],
            "qc": RESULT_SEED["qc"],
            "ai_review": RESULT_SEED["ai_review"],
            "contacted": {},
        }})
        rows = {r[0]: r for r in list(csv.reader(io.StringIO(resp.get_data(as_text=True))))[1:]}
        self.assertEqual(rows["L002"][18], "Needs Human Review")
        self.assertEqual(rows["L001"][18], "Active")  # PASS leads stay active


if __name__ == "__main__":
    unittest.main()
