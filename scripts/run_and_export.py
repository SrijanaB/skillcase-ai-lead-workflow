#!/usr/bin/env python3
"""
Runs the full lead pipeline end-to-end with no UI and writes a CSV.

This is the piece of the workflow meant to run automatically -- e.g. on
a cron schedule, or triggered by a webhook when a new lead lands in a
Google Sheet or CRM. It shares the exact same pipeline.py used by the
web app, so results are identical either way.

Usage:
    GEMINI_API_KEY=your-key python scripts/run_and_export.py
    python scripts/run_and_export.py --out today.csv

Example cron entry (runs every morning at 7am):
    0 7 * * * cd /path/to/skillcase-app && /usr/bin/python3 scripts/run_and_export.py --out exports/$(date +%F).csv
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import pipeline  # noqa: E402

FIELDNAMES = [
    "Lead", "Name", "Relevant", "Reason", "Confidence", "Urgency", "Intent", "Profile",
    "Need", "Objection", "Objection Severity", "Missing Information", "Priority", "Priority Reason",
    "Next Action", "Outreach", "QC Flags",
]


def build_rows(result: dict) -> list[dict]:
    rows = []
    for lead in result["primaries"]:
        lid = lead["lead_id"]
        cls = result["classifications"].get(lid, {})
        enr = result["enrichments"].get(lid, {})
        outreach = result["outreach"].get(lid, "")
        qc = result["qc"].get(lid, [])
        priority = lead.get("priority", {})
        rows.append({
            "Lead": lid,
            "Name": lead["name"],
            "Relevant": "Yes" if cls.get("relevant") else "No",
            "Reason": cls.get("reason", ""),
            "Confidence": cls.get("confidence", ""),
            "Urgency": enr.get("urgency", ""),
            "Intent": enr.get("intent", ""),
            "Profile": enr.get("profile", ""),
            "Need": enr.get("need", ""),
            "Objection": enr.get("objection", ""),
            "Objection Severity": enr.get("objection_severity", ""),
            "Missing Information": enr.get("missing_info", ""),
            "Priority": priority.get("band", "N/A"),
            "Priority Reason": priority.get("priority_reason", ""),
            "Next Action": enr.get("next_action", ""),
            "Outreach": outreach,
            "QC Flags": " | ".join(qc),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="skillcase_lead_manifest.csv", help="output CSV path")
    parser.add_argument("--leads", default=None, help="path to a leads JSON file (defaults to data/leads.json)")
    parser.add_argument("--json-out", default=None, help="optionally also dump the full raw result as JSON")
    args = parser.parse_args()

    leads_path = args.leads or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "leads.json")
    with open(leads_path) as f:
        raw_leads = json.load(f)

    print(f"Running pipeline on {len(raw_leads)} raw leads...")
    result = pipeline.run_full_pipeline(raw_leads)

    n_dupes = sum(g["all"].__len__() - 1 for g in result["duplicate_groups"])
    n_relevant = sum(1 for c in result["classifications"].values() if c.get("relevant"))
    n_flagged = sum(1 for flags in result["qc"].values() if flags)
    print(f"  {n_dupes} duplicate record(s) resolved")
    print(f"  {n_relevant}/{len(result['primaries'])} leads classified relevant")
    print(f"  {n_flagged} lead(s) carry at least one QC flag")

    rows = build_rows(result)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
