"""
Unit tests for Skillcase lead qualification and prioritization pipeline.
"""
import json
import os
import unittest

import pipeline


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
        enr = {"urgency": "medium", "objection_category": "none", "objection_severity": "none"}

        res_high = pipeline.compute_priority(lead, cls_high_conf, enr)
        res_low = pipeline.compute_priority(lead, cls_low_conf, enr)

        # B2 (6) + 3 exp + 0 source + 1 urgency - 0 obj = 10.0
        self.assertEqual(res_high["score"], 10.0)
        self.assertEqual(res_low["score"], 10.0)
        self.assertEqual(res_high["score"], res_low["score"])

    def test_urgency_bonus_and_fallback(self):
        """Test high, medium, low urgency bonuses and fallback for missing/invalid urgency."""
        lead = {"lead_id": "T02", "german_level": "A2", "experience_years": 2, "source": "Website"}
        cls = {"relevant": True, "confidence": 0.8}

        # A2 (2) + 2 exp = 4 base

        # High urgency (+2) -> 6.0
        res_high = pipeline.compute_priority(lead, cls, {"urgency": "high", "objection_category": "none"})
        self.assertEqual(res_high["score"], 6.0)

        # Medium urgency (+1) -> 5.0
        res_med = pipeline.compute_priority(lead, cls, {"urgency": "medium", "objection_category": "none"})
        self.assertEqual(res_med["score"], 5.0)

        # Low urgency (+0) -> 4.0
        res_low = pipeline.compute_priority(lead, cls, {"urgency": "low", "objection_category": "none"})
        self.assertEqual(res_low["score"], 4.0)

        # Missing urgency -> defaults to low (+0) -> 4.0
        res_none = pipeline.compute_priority(lead, cls, {"urgency": None, "objection_category": "none"})
        self.assertEqual(res_none["score"], 4.0)

        # Invalid urgency -> defaults to low (+0) -> 4.0
        res_invalid = pipeline.compute_priority(lead, cls, {"urgency": "asap_urgent", "objection_category": "none"})
        self.assertEqual(res_invalid["score"], 4.0)

    def test_objection_severity_multipliers(self):
        """Test strong, moderate, mild, none multipliers and defaults."""
        lead = {"lead_id": "T03", "german_level": "B2", "experience_years": 4, "source": "Referral"}
        cls = {"relevant": True, "confidence": 0.9}
        # B2 (6) + 4 exp + 1 referral + 0 urgency = 11.0 base
        # Price base = 2.0

        # Strong (1.5x) -> penalty 3.0 -> 8.0
        res_strong = pipeline.compute_priority(
            lead, cls, {"urgency": "low", "objection_category": "price", "objection_severity": "strong"}
        )
        self.assertEqual(res_strong["score"], 8.0)

        # Moderate (1.0x) -> penalty 2.0 -> 9.0
        res_mod = pipeline.compute_priority(
            lead, cls, {"urgency": "low", "objection_category": "price", "objection_severity": "moderate"}
        )
        self.assertEqual(res_mod["score"], 9.0)

        # Mild (0.5x) -> penalty 1.0 -> 10.0
        res_mild = pipeline.compute_priority(
            lead, cls, {"urgency": "low", "objection_category": "price", "objection_severity": "mild"}
        )
        self.assertEqual(res_mild["score"], 10.0)

        # None (0.0x) -> penalty 0.0 -> 11.0
        res_none = pipeline.compute_priority(
            lead, cls, {"urgency": "low", "objection_category": "price", "objection_severity": "none"}
        )
        self.assertEqual(res_none["score"], 11.0)

        # Missing severity for price objection -> defaults to moderate (1.0x -> 2.0 penalty) -> 9.0
        res_missing_sev = pipeline.compute_priority(
            lead, cls, {"urgency": "low", "objection_category": "price", "objection_severity": None}
        )
        self.assertEqual(res_missing_sev["score"], 9.0)

        # Invalid severity -> defaults to moderate (1.0x -> 2.0 penalty) -> 9.0
        res_invalid_sev = pipeline.compute_priority(
            lead, cls, {"urgency": "low", "objection_category": "price", "objection_severity": "super_blocker"}
        )
        self.assertEqual(res_invalid_sev["score"], 9.0)

    def test_priority_bands(self):
        """Bands: High >= 8, Medium >= 4, Low < 4."""
        cls = {"relevant": True, "confidence": 0.8}

        # Score 8.0 -> High
        res_8 = pipeline.compute_priority(
            {"german_level": "B2", "experience_years": 2, "source": "Instagram"},
            cls, {"urgency": "low", "objection_category": "none"}
        )
        self.assertEqual(res_8["score"], 8.0)
        self.assertEqual(res_8["band"], "High")

        # Score 7.9 -> Medium
        res_7_9 = pipeline.compute_priority(
            {"german_level": "B2", "experience_years": 2, "source": "WhatsApp"},  # 6 + 2 + 0.5 = 8.5
            cls, {"urgency": "low", "objection_category": "timeline", "objection_severity": "moderate"}  # -0.5 -> 8.0
        )
        # Let's craft exact 4.0: A2 (2) + 2 exp = 4.0 -> Medium
        res_4 = pipeline.compute_priority(
            {"german_level": "A2", "experience_years": 2, "source": "Instagram"},
            cls, {"urgency": "low", "objection_category": "none"}
        )
        self.assertEqual(res_4["score"], 4.0)
        self.assertEqual(res_4["band"], "Medium")

        # Score 3.0 -> Low
        res_3 = pipeline.compute_priority(
            {"german_level": "A2", "experience_years": 1, "source": "Instagram"},
            cls, {"urgency": "low", "objection_category": "none"}
        )
        self.assertEqual(res_3["score"], 3.0)
        self.assertEqual(res_3["band"], "Low")

    def test_priority_reason_generation(self):
        """Priority reason must be human-readable and reflect actual lead data."""
        lead = {
            "lead_id": "T04",
            "german_level": "B2",
            "experience_years": 3,
            "source": "Referral",
        }
        cls = {"relevant": True, "confidence": 0.9}
        enr = {
            "urgency": "high",
            "objection_category": "price",
            "objection_severity": "moderate",
        }
        res = pipeline.compute_priority(lead, cls, enr)
        reason = res["priority_reason"]
        self.assertIn("High priority", reason)
        self.assertIn("German", reason)
        self.assertIn("experience", reason)
        self.assertIn("price objection reduced the score", reason)

        # Irrelevant lead reason
        res_irr = pipeline.compute_priority(lead, {"relevant": False}, enr)
        self.assertEqual(res_irr["band"], "N/A")
        self.assertIn("Not scored", res_irr["priority_reason"])


class TestLeadPreservationAndDuplicates(unittest.TestCase):
    def setUp(self):
        leads_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "leads.json")
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
        lead = {"lead_id": "T08", "german_level": "B2", "experience_years": 3, "source": "Instagram"}
        cls = {"relevant": True, "confidence": 0.9}
        enr = {"urgency": "high", "objection_category": "none"}
        # Expected score: B2 (6) + 3 exp + 2 urgency = 11.0, band: High

        # Tampered score (99.0 instead of 11.0)
        tampered_priority = {"score": 99.0, "band": "High"}
        flags = pipeline.rule_based_qc(lead, cls, enr, None, priority=tampered_priority)
        self.assertTrue(any("Priority score mismatch" in f for f in flags))

        # Tampered band
        tampered_band = {"score": 11.0, "band": "Low"}
        flags_band = pipeline.rule_based_qc(lead, cls, enr, None, priority=tampered_band)
        self.assertTrue(any("Priority band mismatch" in f for f in flags_band))


if __name__ == "__main__":
    unittest.main()
