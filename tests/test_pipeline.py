"""
Unit tests for Skillcase lead qualification and prioritization pipeline.

Covers: priority scoring, dedupe accounting, QC rules, messy-field safety,
Gemini-layer robustness (retries, malformed output, schema fallback),
batching/echo-validation, and a full offline 30-lead pipeline run.
"""
import copy
import json
import os
import sys
import unittest
from unittest import mock

import pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_client_response(parsed):
    return {"parsed": parsed, "text": None}


class TestPriorityScoring(unittest.TestCase):

    def test_confidence_excluded_from_score(self):
        """Classification confidence must have zero effect on priority score."""
        lead = {
            "lead_id": "T01",
            "german_level": "B2",
            "experience_years": 3,
            "source": "Instagram",
        }

        cls_high_conf = {"relevant": True, "confidence": 0.95}
        cls_low_conf = {"relevant": True, "confidence": 0.20}

        enr = {
            "intent_level": "medium",
            "urgency": "medium",
            "engagement_level": "mocked",
        }

        res_high = pipeline.compute_priority(lead, cls_high_conf, enr)
        res_low = pipeline.compute_priority(lead, cls_low_conf, enr)

        self.assertEqual(res_high["score"], res_low["score"])

    def test_priority_formula(self):
        """Priority should follow Intent 40% + Urgency 25% + Fit 20% + Engagement 15%."""

        lead = {
            "lead_id": "T02",
            "german_level": "B2",
            "experience_years": 3,
        }

        cls = {
            "relevant": True,
            "confidence": 0.9,
        }

        enr = {
            "intent_level": "high",
            "urgency": "high",
            "engagement_level": "high",
        }

        result = pipeline.compute_priority(lead, cls, enr)

        # B2 = 80
        # 3 years experience = 80
        # Fit = (80 + 80) / 2 = 80
        #
        # 100*0.40 + 100*0.25 + 80*0.20 + 100*0.15
        # = 40 + 25 + 16 + 15
        # = 96
        self.assertEqual(result["score"], 96.0)

    def test_priority_components(self):
        """Priority result should expose the four scoring components."""

        lead = {
            "lead_id": "T03",
            "german_level": "A2",
            "experience_years": 1,
        }

        cls = {
            "relevant": True,
            "confidence": 0.0,  # regression guard: confidence must stay numeric
        }

        enr = {
            "intent_level": "medium",
            "urgency": "low",
            "engagement_level": "high",
        }

        result = pipeline.compute_priority(lead, cls, enr)

        self.assertEqual(result["components"]["intent"], 60)
        self.assertEqual(result["components"]["urgency"], 20)
        self.assertEqual(result["components"]["engagement"], 100)

        # A2 = 40
        # 1 year experience = 40
        # Fit = 40
        self.assertEqual(result["components"]["fit"], 40.0)

    def test_priority_bands(self):
        """Bands should be High >=80, Medium 50-79, Low <50."""

        cls = {
            "relevant": True,
            "confidence": 0.9,
        }

        # High:
        # high intent = 100
        # high urgency = 100
        # B2 + 5 years = fit 90
        # high engagement = 100
        #
        # 40 + 25 + 18 + 15 = 98
        high = pipeline.compute_priority(
            {
                "german_level": "B2",
                "experience_years": 5,
            },
            cls,
            {
                "intent_level": "high",
                "urgency": "high",
                "engagement_level": "high",
            },
        )

        self.assertEqual(high["band"], "High")
        self.assertGreaterEqual(high["score"], 80)

        # Medium:
        # medium intent = 60
        # test placeholder 60
        # B2 + 3 years = fit 80
        # medium engagement = 60
        #
        # 24 + 15 + 16 + 9 = 64
        medium = pipeline.compute_priority(
            {
                "german_level": "B2",
                "experience_years": 3,
            },
            cls,
            {
                "intent_level": "medium",
                "urgency": "medium",
                "engagement_level": "medium",
            },
        )

        self.assertEqual(medium["score"], 64.0)
        self.assertEqual(medium["band"], "Medium")

        # Low:
        # low intent = 25
        # low urgency = 20
        # A1 + 0 experience = fit 10
        # low engagement = 20
        #
        # 10 + 5 + 2 + 3 = 20
        low = pipeline.compute_priority(
            {
                "g German": "A1",  # messy key: must not affect scoring
                "german_level": "A1",
                "experience_years": 0,
            },
            cls,
            {
                "intent_level": "low",
                "urgency": "low",
                "engriority": "low",  # messy key: ignored
                "engagement_level": "low",
            },
        )

        self.assertEqual(low["score"], 20.0)
        self.assertEqual(low["band"], "Low")

    def test_invalid_values_fallback_safely(self):
        """Invalid AI labels should fall back without crashing."""

        lead = {
            "lead_id": "T04",
            "german_level": "B2",
            "experience_years": 3,
        }

        cls = {
            "relevant": True,
            "confidence": 0.9,
        }

        enr = {
            "intent_level": "something_invalid",
            "urgency": "something_invalid",
            "enr-engagement_level": "nothing",
            "engagement_level": "something_invalid",
        }

        result = pipeline.compute_priority(lead, cls, enr)

        # Invalid intent -> unclear = 0
        # Invalid urgency -> low = 20
        # B2 + 3 years -> fit 80
        # Invalid engagement -> low = 20
        #
        # 0 + 5 + 16 + 3 = 24
        self.assertEqual(result["score"], 24.0)
        self.assertEqual(result["band"], "Low")

    def test_irrelevant_lead_not_scored(self):
        """Irrelevant leads should not receive a priority score."""

        lead = {
            "lead_id": "T05",
            "german_level": "B2",
            "experience_years": 3,
        }

        cls = {
            "relevant": False,
            "confidence": 0.95,
        }

        enr = {
            "intent_level": "high",
            "urgency": "high",
            "engagement_level": "high",
        }

        result = pipeline.compute_priority(lead, cls, enr)

        self.assertIsNone(result["score"])
        self.assertEqual(result["band"], "N/A")

    def test_priority_reason_generation(self):
        """Priority reason should describe the actual scoring components."""

        lead = {
            "lead_id": "T06",
            "german_level": "B2",
            "experience_years": 3,
        }

        cls = {
            "relevant": True,
            "confidence": 0.9,
        }

        enr = {
            "intent_level": "high",
            "urgency": "high",
            "engagement_level": "high",
        }

        result = pipeline.compute_priority(lead, cls, enr)

        reason = result["priority_reason"]

        self.assertIn("Intent: High", reason)
        self.assertIn("Urgency: High", reason)
        self.assertIn("Fit: 80.0/100", reason)
        self.assertIn("Engagement: High", reason)


class TestLeadPreservationAndDuplicates(unittest.TestCase):
    def setUp(self):
        leads_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "leads.json")
        with open(leads_path) as f:
            self.raw_leads = json.load(f)

    def test_all_30_original_leads_accounted_for(self):
        """Deduplication must account for all 30 original lead IDs in lead_audit."""
        self.assertEqual(len(self.raw_leads), 30)
        cleaned = pipeline.clean_and_dedupe(self.raw_leads)

        primaries = cleaned["primaries"]
        lead_audit = cleaned["lead_audit"]

        # Active primary batch has 27 records
        self.assertEqual(len(primaries), 27)

        # Audit contains all 30 original records
        self.assertEqual(len(lead_audit), 30)

        original_ids = {l["lead_id"] for l in self.raw_leads}
        audited_ids = {a["lead_id"] for a in lead_audit}
        self.assertEqual(original_ids, audited_ids)

    def test_specific_duplicate_cases(self):
        """Verify L001, L008, L021, and L028 relationships."""
        cleaned = pipeline.clean_and_dedupe(self.raw_leads)
        audit_by_id = {a["lead_id"]: a for a in cleaned["lead_audit"]}

        # L001 is primary retained
        self.assertEqual(audit_by_id["L001"]["status"], "retained")
        self.assertEqual(audit_by_id["L001"]["primary_id"], "L001")
        self.assertEqual(audit_by_id["L001"]["duplicate_relationship"], "primary")

        # L008 is exact duplicate of L001 (auto-merged)
        self.assertEqual(audit_by_id["L008"]["status"], "merged")
        self.assertNotIn("priority", audit_by_id["L008"])  # merged records are not scored
        self.assertEqual(audit_by_id["L008"]["primary_id"], "L001")
        self.assertEqual(audit_by_id["L008"]["duplicate_relationship"], "exact_duplicate")
        self.assertFalse(audit_by_id["L008"]["requires_human_review"])

        # L028 is conflicting duplicate of L001 (flagged for review)
        self.assertEqual(audit_by_id["L028"]["status"], "flagged")
        self.assertEqual(audit_by_id["L028"]["primary_id"], "L001")
        self.assertEqual(audit_by_id["L028"]["duplicate_relationship"], "conflicting_duplicate")
        self.assertTrue(audit_by_id["L028"]["requires_human_review"])

        # L004 is primary retained
        self.assertEqual(audit_by_id["L004"]["status"], "retained")
        self.assertEqual(audit_by_id["L004"]["primary_id"], "L004")

        # L021 is exact duplicate of L004 (auto-merged)
        self.assertEqual(audit_by_id["L021"]["status"], "merged")
        self.assertEqual(audit_by_id["L021"]["primary_id"], "L004")
        self.assertEqual(audit_by_id["L021"]["duplicate_relationship"], "exact_duplicate")
        self.assertFalse(audit_by_id["L021"]["requires_human_review"])

    def test_missing_and_messy_fields_do_not_crash(self):
        """Missing name/phone/education/experience/goal must not raise."""
        messy = [
            {"lead_id": "X1", "name": "", "phone": "", "email": None, "city": None,
             "education": None, "experience": None, "goal": None, "german_level": "",
             "source": None, "last_contacted": None, "conversation": None, "notes": None},
            {},  # no lead_id -> synthesized
            {"lead_id": "X3", "experience": "18 months", "name": "Test Lead"},
            {"lead_id": "X4", "phone": "+91 98765 43210", "conversation": "", "german_level": "b1"},
        ]
        cleaned = pipeline.clean_and_dedupe(messy)
        self.assertEqual(len(cleaned["lead_audit"]), 4)
        ids = {a["lead_id"] for a in cleaned["lead_audit"]}
        self.assertIn("RAW-002", ids)
        # 18 months = 1.5 years, parsed from a non-"years" format
        x3 = next(p for p in cleaned["primaries"] if p["lead_id"] == "X3")
        self.assertEqual(x3["experience_years"], 1.5)
        # phone spaces stripped before dedupe grouping
        self.assertTrue(all(c["phone"].count(" ") == 0 for c in cleaned["primaries"]))

    def test_two_phoneless_leads_are_not_merged_together(self):
        """Leads without any phone must never be merged with each other."""
        messy = [
            {"lead_id": "P1", "phone": "", "name": "A", "email": "a@x.com"},
            {"lead_id": "P2", "phone": "", "name": "B", "email": "b@x.com"},
        ]
        cleaned = pipeline.clean_and_dedupe(messy)
        self.assertEqual(len(cleaned["primaries"]), 2)


class TestQCValidation(unittest.TestCase):
    def test_missing_or_invalid_urgency_flag(self):
        lead = {"lead_id": "T05", "education": "BSc Nursing"}
        cls = {"relevant": True, "confidence": 0.9}

        # Missing urgency
        flags_missing = pipeline.rule_based_qc(lead, cls, {"urgency": ""}, None)
        self.assertTrue(any("Missing or invalid urgency" in f for f in flags_missing))

        # Invalid urgency
        flags_invalid = pipeline.rule_based_qc(lead, cls, {"urgency": "super_urgent"}, None)
        self.assertTrue(any("Missing or invalid urgency" in f for f in flags_invalid))

        # Valid urgency
        flags_valid = pipeline.rule_based_qc(lead, cls, {"urgency": "high"}, None)
        self.assertFalse(any("Missing or invalid urgency" in f for f in flags_valid))

    def test_missing_or_invalid_objection_severity_flag(self):
        lead = {"lead_id": "T06", "education": "BSc Nursing"}
        cls = {"relevant": True, "confidence": 0.9}

        # Price objection with missing severity
        flags_missing_sev = pipeline.rule_based_qc(
            lead, cls, {"urgency": "medium", "objection_category": "price", "objection_severity": ""}, None
        )
        self.assertTrue(any("Missing or invalid objection severity" in f for f in flags_missing_sev))

        # Price objection with invalid severity
        flags_inv_sev = pipeline.rule_based_qc(
            lead, cls, {"urgency": "medium", "objection_category": "price", "objection_severity": "extreme"}, None
        )
        self.assertTrue(any("Missing or invalid objection severity" in f for f in flags_inv_sev))

        # Severity specified when category is none
        flags_sev_no_cat = pipeline.rule_based_qc(
            lead, cls, {"urgency": "medium", "objection_category": "none", "objection_severity": "moderate"}, None
        )
        self.assertTrue(any("specified but objection category is none" in f for f in flags_sev_no_cat))

    def test_low_confidence_and_duplicate_conflict_flags(self):
        lead = {"lead_id": "T07", "education": "BSc Nursing", "_dup_fuzzy": ["T07_B"]}
        cls = {"relevant": True, "confidence": 0.45}
        enr = {"urgency": "low", "objection_category": "none"}

        flags = pipeline.rule_based_qc(lead, cls, enr, None)
        self.assertTrue(any("Low classification confidence (0.45)" in f for f in flags))
        self.assertTrue(any("conflicting details (T07_B)" in f for f in flags))

    def test_formula_mismatch_flag(self):
        lead = {
            "lead_id": "T08",
            "german_level": "B2",
            "experience_years": 3,
            "source": "Instagram",
        }

        cls = {
            "relevant": True,
            "confidence": 0.9,
        }

        enr = {
            "intent_level": "high",
            "urgency": "high",
            "engagement_level": "high",
            "objection_category": "none",
            "objection_severity": "none",
        }

        # Correct score generated by the current formula.
        expected = pipeline.compute_priority(
            lead,
            cls,
            enr,
        )

        # Tampered score should trigger a score mismatch.
        tampered_priority = {
            "score": 99.0,
            "band": expected["band"],
        }

        flags = pipeline.rule_based_qc(
            lead,
            cls,
            enr,
            None,
            priority=tampered_priority,
        )

        self.assertTrue(
            any("Priority score mismatch" in f for f in flags)
        )

        # Correct score + deliberately wrong band should trigger
        # a priority-band mismatch.
        wrong_band = "Low" if expected["band"] != "Low" else "High"

        tampered_band = {
            "score": expected["score"],
            "band": wrong_band,
        }

        flags_band = pipeline.rule_based_qc(
            lead,
            cls,
            enr,
            None,
            priority=tampered_band,
        )

        self.assertTrue(
            any("Priority band mismatch" in f for f in flags_band)
        )

    def test_relevant_lead_without_enrichment_is_flagged(self):
        """Regression: an absent enrichment must surface as a QC flag
        (previously the lead silently scored low with empty enrichment)."""
        lead = {"lead_id": "T09", "education": "BSc Nursing"}
        cls = {"relevant": True, "confidence": 0.9}

        flags = pipeline.rule_based_qc(lead, cls, {}, None)
        self.assertTrue(any("No enrichment returned" in f for f in flags))

    def test_unclassified_lead_is_flagged(self):
        """Regression: missing/malformed classification must not silently
        count as not-relevant without review."""
        lead = {"lead_id": "T10", "education": "BSc Nursing"}
        flags = pipeline.rule_based_qc(lead, {}, {"urgency": "high"}, None)
        self.assertTrue(any("Classification missing or malformed" in f for f in flags))

    def test_outreach_claim_and_hygiene_flags(self):
        lead = {"lead_id": "T11", "education": "BSc Nursing"}
        cls = {"relevant": True, "confidence": 0.9}
        enr = {"urgency": "high", "objection_category": "none", "objection_severity": "none"}

        # Guarantee-style claim
        flags_claim = pipeline.rule_based_qc(
            lead, cls, enr, "We can guarantee a job in Germany after training.")
        self.assertTrue(any("guarantee" in f for f in flags_claim))

        # Placement-rate style claim
        flags_placement = pipeline.rule_based_qc(
            lead, cls, enr, "Our placement rate is 95%.")
        self.assertTrue(any("unsupported placement/salary/visa claim" in f for f in flags_placement))

        # Too many questions
        flags_questions = pipeline.rule_based_qc(
            lead, cls, enr, "When can you talk? Are you free Monday? What about Tuesday? Or Wednesday?")
        self.assertTrue(any("asks 4 questions" in f for f in flags_questions))

        # Excessive length
        long_msg = " ".join(["word"] * 95)
        flags_length = pipeline.rule_based_qc(lead, cls, enr, long_msg)
        self.assertTrue(any("Outreach draft is long (95 words)" in f for f in flags_length))

        # Generic filler
        flags_generic = pipeline.rule_based_qc(
            lead, cls, enr, "We'd love to help you on your journey to Germany.")
        self.assertTrue(any("generic filler" in f for f in flags_generic))

        # Clean outreach raises none of the outreach flags
        clean = "Thanks for asking about the B2 requirement, Priya. I can walk you through what the Goethe exam involves and what studying looks like. Would a 10-minute call this week help?"
        flags_clean = pipeline.rule_based_qc(lead, cls, enr, clean)
        self.assertFalse([f for f in flags_clean if "Outreach" in f or "outreach" in f])


class TestGeminiLayer(unittest.TestCase):
    """Tests for the Gemini request/retry layer (network-free, mocked)."""

    def setUp(self):
        pipeline._client = None

    def tearDown(self):
        pipeline._client = None

    def test_missing_api_key_raises_clean_error(self):
        env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as ctx:
                pipeline.get_client()
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_api_error_message_redacts_key(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "supersecret123"}):
            exc = RuntimeError("boom supersecret123 leaked")
            msg = pipeline._safe_api_error_message(exc)
            self.assertNotIn("supersecret123", msg)
            self.assertIn("[redacted]", msg)

    def test_no_retry_on_deterministic_400(self):
        """A 400 must surface immediately (after schema fallback), not hang retries."""
        calls = {"n": 0}

        def boom(prompt, max_tokens=0, schema=None):
            calls["n"] += 1
            raise RuntimeError("Gemini API error (400): bad request")

        with mock.patch.object(pipeline, "_ask_for_json_array", boom):
            with self.assertRaises(RuntimeError):
                pipeline._ask_for_json_array_with_retry("p", schema=pipeline.CLASSIFY_SCHEMA)
        self.assertEqual(calls["n"], 1)

    def test_retries_on_429_then_succeeds(self):
        responses = [
            RuntimeError("Gemini API error (429): rate limited"),
            [{"lead_id": "L001", "relevant": True, "reason": "ok", "confidence": 0.9}],
        ]

        def flaky(prompt, max_tokens=0, schema=None):
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        sleeps = []
        with mock.patch.object(pipeline, "_ask_for_json_array", side_effect=flaky), \
             mock.patch.object(pipeline.time, "sleep", side_effect=sleeps.append):
            result = pipeline._ask_for_json_array_with_retry("p", schema=pipeline.CLASSIFY_SCHEMA)

        self.assertEqual(len(result), 1)
        self.assertEqual(len(sleeps), 1)  # exactly one backoff between the two attempts
        self.assertGreater(sleeps[0], 0)

    def test_retries_on_malformed_json_then_succeeds(self):
        responses = [
            ValueError("No JSON array found in model response."),
            [{"lead_id": "L001", "relevant": False, "reason": "ok", "confidence": 0.9}],
        ]

        def flaky(prompt, max_tokens=0, schema=None):
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        with mock.patch.object(pipeline, "_ask_for_json_array", side_effect=flaky), \
             mock.patch.object(pipeline.time, "sleep", lambda s: None):
            result = pipeline._ask_for_json_array_with_retry("p")

        self.assertEqual(len(result), 1)

    def test_gives_up_after_max_attempts(self):
        def always_429(prompt, max_tokens=0, schema=None):
            raise RuntimeError("Gemini API error (429): rate limited")

        with mock.patch.object(pipeline, "_ask_for_json_array", side_effect=always_429), \
             mock.patch.object(pipeline.time, "sleep", lambda s: None):
            with self.assertRaises(RuntimeError):
                pipeline._ask_for_json_array_with_retry("p")

    def test_parse_json_array_handles_fences_and_wrappers(self):
        self.assertEqual(pipeline._parse_json_array('```json\n[{"a":1}]\n```'), [{"a": 1}])
        self.assertEqual(pipeline._parse_json_array('{"items": [{"a": 1}]}'), [{"a": 1}])
        with self.assertRaises(ValueError):
            pipeline._parse_json_array("no json here")
        with self.assertRaises(ValueError):
            pipeline._parse_json_array("")

    def test_salvage_truncated_array_keeps_complete_records(self):
        text = '[{"lead_id":"L001","outreach":"hi"},{"lead_id":"L002","outre'
        salvaged = pipeline._salvage_truncated_array(text)
        self.assertEqual(salvaged, [{"lead_id": "L001", "outreach": "hi"}])
        self.assertIsNone(pipeline._salvage_truncated_array("not json at all"))

    def test_schema_fallback_on_400(self):
        """A 400 against response_json_schema must fall back to plain JSON mode."""
        seen_kwargs = []

        class FakeModels:
            def generate_content(self, model, contents, config):
                seen_kwargs.append(config)
                if "response_json_schema" in config:
                    raise pipeline.genai_errors.APIError(400, {"error": {"message": "response_schema unsupported"}})
                resp = mock.Mock()
                resp.parsed = [{"lead_id": "L001", "relevant": True, "reason": "ok", "confidence": 0.9}]
                resp.text = None
                return resp

        class FakeClient:
            models = FakeModels()

        pipeline._client = FakeClient()
        try:
            result = pipeline._ask_for_json_array("p", schema=pipeline.CLASSIFY_SCHEMA)
        finally:
            pipeline._client = None
        self.assertEqual(len(result), 1)
        self.assertEqual(len(seen_kwargs), 2)
        self.assertIn("response_json_schema", seen_kwargs[0])
        self.assertNotIn("response_json_schema", seen_kwargs[1])

    def test_classify_batching_and_missing_echo_backfilled(self):
        """Batching splits large inputs and backfills leads the model skipped."""
        leads = [{"lead_id": f"L{i:03d}"} for i in range(1, 46)]  # 45 leads -> 3 batches
        captured = []

        def fake_call(prompt, max_tokens=0, schema=None):
            captured.append(prompt)
            ids = {f"L{i:03d}" for i in range(1, 46) if f'L{i:03d}' in prompt}
            batch_ids = [lid for lid in [f"L{i:03d}" for i in range(1, 46)] if f'"{lid}"' in prompt]
            # Model returns only the FIRST lead of the batch (skips the rest)
            return [{"lead_id": batch_ids[0], "relevant": True, "reason": "ok", "confidence": 0.9}]

        with mock.patch.object(pipeline, "_ask_for_json_array_with_retry", side_effect=fake_call):
            result = pipeline.classify_leads(leads)

        self.assertEqual(len(captured), 3)          # 15-lead batches
        self.assertEqual(len(result), 45)           # every lead accounted for
        self.assertEqual(result["L001"]["relevant"], True)
        # Backfilled record is conservative and reviewable
        self.assertEqual(result["L002"]["relevant"], False)
        self.assertEqual(result["L002"]["confidence"], 0.0)
        self.assertIn("human review", result["L002"]["reason"])

    def test_enrich_and_outreach_placeholders_when_model_skips(self):
        leads = [{"lead_id": "L001"}, {"lead_id": "L002"}]

        with mock.patch.object(pipeline, "_ask_for_json_array_with_retry", return_value=[]):
            enr = pipeline.enrich_leads(leads)
            outreach = pipeline.generate_outreach([{"lead": l, "enrichment": {}} for l in leads])

        self.assertEqual(set(enr.keys()), {"L001", "L002"})
        self.assertEqual(enr["L001"]["intent_level"], "unclear")
        self.assertEqual(outreach["L001"], "")


class TestFullPipelineOffline(unittest.TestCase):
    """Full 30-lead pipeline run with the Gemini layer mocked out."""

    def setUp(self):
        leads_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "leads.json")
        with open(leads_path) as f:
            self.raw_leads = json.load(f)

    def test_full_pipeline_accounts_for_every_lead(self):
        n_raw = len(self.raw_leads)

        def fake_classify(leads):
            return {
                l["lead_id"]: {"lead_id": l["lead_id"], "relevant": l["lead_id"] != "L002",
                               "reason": "nursing background", "confidence": 0.92}
                for l in leads
            }

        captured_enrich_leads = []

        def fake_enrich(leads):
            captured_enrich_leads.extend(leads)
            return {
                l["lead_id"]: {
                    "lead_id": l["lead_id"], "profile": "nurse", "intent": "work in Germany",
                    "intent_level": "high", "need": "B2 prep", "objection": "cost",
                    "objection_category": "price", "objection_severity": "mild",
                    "urgency": "high", "engagement_level": "high", "missing_info": "None",
                    "opportunity": "B2 course", "next_action": "schedule call",
                }
                for l in leads
            }

        def fake_outreach(items):
            return {i["lead"]["lead_id"]: f"Hi {i['lead']['name']}, about your question..." for i in items}

        def fake_review(items):
            return {
                i["lead"]["lead_id"]: {
                    "lead_id": i["lead"]["lead_id"], "status": "pass", "confidence": 0.95,
                    "issues": [], "recommended_action": "No action needed.",
                }
                for i in items
            }

        with mock.patch.object(pipeline, "classify_leads", side_effect=fake_classify), \
             mock.patch.object(pipeline, "enrich_leads", side_effect=fake_enrich), \
             mock.patch.object(pipeline, "generate_outreach", side_effect=fake_outreach), \
             mock.patch.object(pipeline, "review_leads", side_effect=fake_review):
            result = pipeline.run_full_pipeline(self.raw_leads)

        self.assertEqual(result["summary"]["raw_leads"], n_raw)
        self.assertEqual(result["summary"]["primaries"], 27)
        self.assertEqual(len(result["classifications"]), 27)
        # AI review runs for relevant leads only and lands in the result
        n_relevant = sum(1 for c in result["classifications"].values() if c["relevant"])
        self.assertEqual(len(result["ai_review"]), n_relevant)
        self.assertEqual(result["summary"]["ai_review_passed"], n_relevant)
        # Enrichment must run for relevant leads only
        n_relevant = sum(1 for c in result["classifications"].values() if c["relevant"])
        self.assertEqual(len(captured_enrich_leads), n_relevant)
        self.assertEqual(result["summary"]["enriched"], n_relevant)
        self.assertEqual(result["summary"]["with_outreach"], n_relevant)
        self.assertEqual(len(result["qc"]), n_raw)  # all 30 traceable
        self.assertIn("runtime_seconds", result["summary"])

        # Every primary lead has a priority dict
        for p in result["primaries"]:
            self.assertIn("priority", p)

        # Irrelevant lead got no score
        self.assertIsNone(next(p for p in result["primaries"] if p["lead_id"] == "L002")["priority"]["score"])

    def test_stream_yields_stage_sequence(self):
        events = []
        with mock.patch.object(pipeline, "classify_leads", return_value={}), \
             mock.patch.object(pipeline, "enrich_leads", return_value={}), \
             mock.patch.object(pipeline, "generate_outreach", return_value={}), \
             mock.patch.object(pipeline, "review_leads", return_value={}):
            for stage, status, message, payload in pipeline.run_full_pipeline_stream(self.raw_leads):
                events.append(stage)

        self.assertEqual(
            events,
            ["clean", "clean", "classify", "classify", "enrich", "enrich",
             "prioritize", "prioritize", "outreach", "outreach", "qc", "qc",
             "ai_review", "ai_review", "done"],
        )


class TestAiReview(unittest.TestCase):
    """AI Review stage: semantic second-pass reviewer (Gemini mocked out)."""

    def _lead(self, lead_id="L001", conversation="I want to start B1 classes next month"):
        return {
            "lead_id": lead_id,
            "name": "Priya Sharma",
            "city": "Delhi",
            "education": "BSc Nursing",
            "experience_years": 3,
            "german_level": "A2",
            "goal": "Work as a nurse in Germany",
            "source": "Instagram",
            "conversation": conversation,
        }

    def _item(self, lead=None, outreach="Hi Priya, you mentioned starting B1 classes next month — shall we set up a plan?", qc_flags=None):
        lead = lead or self._lead()
        return {
            "lead": lead,
            "classification": {"relevant": True, "confidence": 0.9, "reason": "nurse targeting Germany"},
            "enrichment": {
                "profile": "Nurse, 3 yrs experience", "intent": "start B1 classes",
                "intent_level": "high", "need": "B1 course", "objection": "Not stated",
                "urgency": "high", "engagement_level": "high",
                "missing_info": "None", "next_action": "Schedule a call",
            },
            "priority": {"band": "High", "score": 88, "priority_reason": "high intent and urgency"},
            "outreach": outreach,
            "qc_flags": qc_flags or [],
        }

    def test_review_pass_result(self):
        parsed = [{
            "lead_id": "L001", "status": "pass", "confidence": 0.95,
            "issues": [], "recommended_action": "No action needed.",
        }]
        with mock.patch.object(pipeline, "_ask_for_json_array_with_retry", return_value=parsed) as ask:
            result = pipeline.review_leads([self._item()])

        self.assertEqual(result["L001"]["status"], "pass")
        self.assertEqual(result["L001"]["issues"], [])
        self.assertEqual(result["L001"]["recommended_action"], "No action needed.")
        # The reviewer receives the original conversation as source of truth
        payload = ask.call_args[0][0]
        self.assertIn("I want to start B1 classes next month", payload)

    def test_review_review_result(self):
        parsed = [{
            "lead_id": "L001", "status": "review", "confidence": 0.89,
            "issues": ["Outreach introduces a timeline that was not stated by the lead."],
            "recommended_action": "Review outreach before sending.",
        }]
        with mock.patch.object(pipeline, "_ask_for_json_array_with_retry", return_value=parsed):
            result = pipeline.review_leads([self._item()])

        self.assertEqual(result["L001"]["status"], "review")
        self.assertEqual(result["L001"]["issues"],
                         ["Outreach introduces a timeline that was not stated by the lead."])
        self.assertEqual(result["L001"]["recommended_action"], "Review outreach before sending.")

    def test_review_flags_unsupported_outreach_claim(self):
        """The prompt directs the reviewer to flag invented outcomes, and the
        reviewer's verdict must survive normalization unchanged."""
        lead = self._lead(conversation="Just asking generally, no plans yet")
        item = self._item(lead=lead, outreach="We guarantee you a nursing job in Germany within 6 months!")
        parsed = [{
            "lead_id": lead["lead_id"], "status": "review", "confidence": 0.91,
            "issues": ["Outreach promises a guaranteed job and a fixed timeline, neither of which the lead stated."],
            "recommended_action": "Rewrite outreach before sending.",
        }]
        with mock.patch.object(pipeline, "_ask_for_json_array_with_retry", return_value=parsed) as ask:
            result = pipeline.review_leads([item])

        self.assertEqual(result[lead["lead_id"]]["status"], "review")
        self.assertTrue(any("guarantee" in i.lower() for i in result[lead["lead_id"]]["issues"]))
        # The outreach itself is passed through for context, never rewritten
        self.assertIn("We guarantee you a nursing job", ask.call_args[0][0])

    def test_review_normalizes_malformed_model_output(self):
        """Drift (casing, bad confidence, pass-with-issues) is clamped, never crashes."""
        parsed = [
            {"lead_id": "L001", "status": "PASS", "confidence": 7, "issues": []},
            {"lead_id": "L002", "status": "pass", "confidence": 0.9,
             "issues": ["contradicts the conversation"]},
            {"lead_id": "L003", "status": "review", "confidence": "oops"},
        ]
        leads = [self._lead("L001"), self._lead("L002"), self._lead("L003")]
        with mock.patch.object(pipeline, "_ask_for_json_array_with_retry", return_value=parsed):
            result = pipeline.review_leads([self._item(l) for l in leads])

        self.assertEqual(result["L001"]["status"], "pass")
        self.assertEqual(result["L001"]["confidence"], 1.0)  # clamped to [0, 1]
        self.assertEqual(result["L002"]["status"], "review")  # pass + issues -> review
        self.assertEqual(result["L003"]["status"], "review")
        self.assertTrue(result["L003"]["issues"])  # review without issues gets one

    def test_review_placeholder_when_model_skips_lead(self):
        with mock.patch.object(pipeline, "_ask_for_json_array_with_retry", return_value=[]):
            result = pipeline.review_leads([self._item(self._lead("L001"))])

        self.assertEqual(result["L001"]["status"], "review")  # never silently passes
        self.assertTrue(result["L001"]["issues"])

    def test_review_empty_input_is_noop(self):
        self.assertEqual(pipeline.review_leads([]), {})

    def test_review_never_rewrites_inputs(self):
        """The reviewer only reports; enrichment/outreach dicts are untouched."""
        item = self._item()
        original_enrichment = copy.deepcopy(item["enrichment"])
        original_outreach = item["outreach"]
        parsed = [{
            "lead_id": "L001", "status": "review", "confidence": 0.8,
            "issues": ["objection not grounded"],
            "recommended_action": "Review.",
        }]
        with mock.patch.object(pipeline, "_ask_for_json_array_with_retry", return_value=parsed):
            pipeline.review_leads([item])

        self.assertEqual(item["enrichment"], original_enrichment)
        self.assertEqual(item["outreach"], original_outreach)


if __name__ == "__main__":
    unittest.main()
