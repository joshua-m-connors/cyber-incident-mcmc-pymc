#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2025 Joshua M. Connors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---------------------------------------------------------------------------
# MITRE ATT&CK Technique Relevance Template Builder
# ---------------------------------------------------------------------------
# Creates technique_relevance.csv and evidence files to mark which ATT&CK
# techniques are relevant to specified campaigns or procedures.
# ---------------------------------------------------------------------------
"""
Build technique_relevance.csv from the MITRE ATT&CK Enterprise JSON bundle.

Purpose
-------
Creates a checklist CSV listing ATT&CK tactics (Initial Access → Impact),
their associated techniques/sub-techniques, MITRE Technique IDs, and a "Relevant"
column users can mark with X/x.

Enhancements
------------
• Automatically detects the enterprise-attack.json file if not specified.
• Supports auto-populating techniques based on:
    - --procedure "<NAME>"  (e.g., APT29, FIN7, PlugX)
    - --campaign CXXXX      (e.g., C0017, C0029)
• Users can combine multiple --procedure and --campaign arguments.
• Outputs CSV and evidence JSON files in the daily output directory.
• Outputs a JSON sidecar summarizing which techniques were auto-marked.

Intended for use by:
  - mitre_control_strength_dashboard.py
  - cyber_incident_pymc.py
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from collections import defaultdict

# Canonical ATT&CK Enterprise tactic order
TACTIC_ORDER = [
    "initial-access",
    "execution",
    "persistence",
    "privilege-escalation",
    "defense-evasion",
    "credential-access",
    "discovery",
    "lateral-movement",
    "collection",
    "command-and-control",
    "exfiltration",
    "impact",
]

TACTIC_LABELS = {
    "initial-access": "Initial Access",
    "execution": "Execution",
    "persistence": "Persistence",
    "privilege-escalation": "Privilege Escalation",
    "defense-evasion": "Defense Evasion",
    "credential-access": "Credential Access",
    "discovery": "Discovery",
    "lateral-movement": "Lateral Movement",
    "collection": "Collection",
    "command-and-control": "Command and Control",
    "exfiltration": "Exfiltration",
    "impact": "Impact",
}


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def find_local_enterprise_json():
    """Search the current working directory for a plausible ATT&CK bundle."""
    candidates = [
        "enterprise-attack.json",
        "enterprise_attack.json",
        "mitre-enterprise-attack.json",
        "mitre_attack_enterprise.json",
    ]
    for fname in candidates:
        if os.path.exists(fname):
            return fname
    return None


def extract_mitre_id(external_refs):
    """Extract MITRE ATT&CK external ID (e.g., T1059 or C0017)."""
    if not isinstance(external_refs, list):
        return None
    for ref in external_refs:
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def normalize_tactic_name(phase_name):
    """Normalize ATT&CK phase names."""
    return str(phase_name).strip().lower()


def load_attack_patterns(bundle):
    """Return list of (technique_id, technique_name, tactic_slug)."""
    rows = []
    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        name = obj.get("name")
        ext_id = extract_mitre_id(obj.get("external_references"))
        kcp = obj.get("kill_chain_phases", [])
        if not name or not kcp:
            continue
        for phase in kcp:
            if phase.get("kill_chain_name") != "mitre-attack":
                continue
            tactic_slug = normalize_tactic_name(phase.get("phase_name"))
            if tactic_slug not in TACTIC_ORDER:
                continue
            rows.append((ext_id, name, tactic_slug))
    return rows


def build_rows_by_tactic(technique_tuples, dedupe_names=False, sort_techniques=True):
    """Group (technique_id, technique_name, tactic_slug) → tactic_slug → list."""
    by_tactic = defaultdict(list)
    for tid, tname, tactic in technique_tuples:
        by_tactic[tactic].append((tid, tname))

    if dedupe_names:
        for tactic, items in by_tactic.items():
            seen = set()
            deduped = []
            for tid, tname in items:
                key = tname.lower()
                if key not in seen:
                    deduped.append((tid, tname))
                    seen.add(key)
            by_tactic[tactic] = deduped

    if sort_techniques:
        for tactic in by_tactic:
            by_tactic[tactic].sort(key=lambda x: x[1].lower())

    return by_tactic


# ---------------------------------------------------------------------
# Relationship-based Mapping for Procedures and Campaigns
# ---------------------------------------------------------------------

def find_object_by_name_or_id(bundle, name_or_id, valid_types):
    """Find an ATT&CK object by name (case-insensitive) or exact id."""
    matches = []
    for obj in bundle.get("objects", []):
        if obj.get("type") not in valid_types:
            continue
        if obj.get("id") == name_or_id:
            matches.append(obj)
        elif name_or_id.lower() in obj.get("name", "").lower():
            matches.append(obj)
    return matches


def find_campaign_by_external_id(bundle, cid):
    """Find a campaign object whose external_id == 'Cxxxx'."""
    for obj in bundle.get("objects", []):
        if obj.get("type") != "campaign":
            continue
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack" and ref.get("external_id") == cid:
                return obj
    return None


def collect_techniques_for_sources(bundle, source_ids):
    """Collect attack-pattern IDs linked via 'uses' relationships from given sources."""
    technique_ids = set()
    technique_descriptions = []

    # Map attack-pattern IDs to external IDs and names for fast lookup
    id_to_tech = {}
    for obj in bundle.get("objects", []):
        if obj.get("type") == "attack-pattern":
            mitre_id = extract_mitre_id(obj.get("external_references"))
            id_to_tech[obj.get("id")] = (mitre_id, obj.get("name"))

    for rel in bundle.get("objects", []):
        if rel.get("type") != "relationship":
            continue
        if rel.get("relationship_type") != "uses":
            continue
        if rel.get("source_ref") in source_ids and rel.get("target_ref", "").startswith("attack-pattern--"):
            target = rel["target_ref"]
            if target in id_to_tech:
                tid, tname = id_to_tech[target]
                if tid:
                    technique_ids.add(tid)
                    technique_descriptions.append({
                        "technique_id": tid,
                        "technique_name": tname,
                        "relationship_description": rel.get("description", "")
                    })
    return technique_ids, technique_descriptions


# ---------------------------------------------------------------------
# CSV Writer
# ---------------------------------------------------------------------

def write_csv(by_tactic, out_path, auto_mark_ids=None, mark_all="none"):
    """Write CSV with Tactic, Technique ID, Technique, Relevant."""
    fieldnames = ["Tactic", "Technique ID", "Technique", "Relevant"]
    mark_value = "X" if mark_all == "all" else ""

    auto_mark_ids = auto_mark_ids or set()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for tactic_slug in TACTIC_ORDER:
            techniques = by_tactic.get(tactic_slug, [])
            if not techniques:
                continue
            tactic_label = TACTIC_LABELS.get(tactic_slug, tactic_slug.title())
            for tid, tname in techniques:
                relevant = "X" if tid in auto_mark_ids else mark_value
                writer.writerow({
                    "Tactic": tactic_label,
                    "Technique ID": tid or "",
                    "Technique": tname,
                    "Relevant": relevant
                })


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build technique_relevance.csv from MITRE ATT&CK Enterprise JSON."
    )
    parser.add_argument(
        "--enterprise-json",
        help="Path to MITRE Enterprise JSON bundle (auto-detected if not provided).",
    )
    parser.add_argument(
        "--mark-all",
        choices=["none", "all"],
        default="none",
        help="Mark all techniques as relevant ('all') or leave blank ('none'). Default: none",
    )
    parser.add_argument(
        "--procedure",
        action="append",
        help="Threat actor / malware / tool name (e.g., 'APT29', 'FIN7'). May be used multiple times.",
    )
    parser.add_argument(
        "--campaign",
        action="append",
        help="Campaign ID (e.g., C0017). May be used multiple times.",
    )
    parser.add_argument(
        "--dedupe-names",
        action="store_true",
        help="Dedupe identical Technique names within a tactic (rare).",
    )
    parser.add_argument(
        "--sort-techniques",
        action="store_true",
        default=True,
        help="Sort techniques alphabetically within each tactic (default: on).",
    )

    args = parser.parse_args()

    # Auto-detect ATT&CK Enterprise JSON
    enterprise_path = args.enterprise_json or find_local_enterprise_json()
    if not enterprise_path:
        print("❌ ERROR: No MITRE ATT&CK Enterprise JSON file found.")
        sys.exit(1)

    # Load bundle
    try:
        with open(enterprise_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        print(f"📦 Loaded MITRE ATT&CK Enterprise JSON: {enterprise_path}")
    except Exception as e:
        print(f"ERROR: Could not read JSON bundle: {e}", file=sys.stderr)
        sys.exit(1)

    # Build base data (all techniques)
    techniques = load_attack_patterns(bundle)
    grouped = build_rows_by_tactic(
        technique_tuples=techniques,
        dedupe_names=args.dedupe_names,
        sort_techniques=args.sort_techniques,
    )

    # Collect techniques based on procedure and campaign arguments
    # a9fbaa806d527ffe1cc1aa3b2a9ba944567a4a60
    auto_mark_ids = set()
    evidence_records = []

    # Procedure-based mapping
    if args.procedure:
        for pname in args.procedure:
            matches = find_object_by_name_or_id(
                bundle, pname, valid_types=["intrusion-set", "malware", "tool"]
            )
            if not matches:
                print(f"⚠️  WARNING: No matching procedure found for '{pname}'")
                continue
            ids = [m["id"] for m in matches]
            tids, descs = collect_techniques_for_sources(bundle, ids)
            auto_mark_ids.update(tids)
            evidence_records.append({"procedure": pname, "techniques": descs})

    # Campaign-based mapping
    if args.campaign:
        for cid in args.campaign:
            if not re.match(r"^C\d{4,5}$", cid, re.IGNORECASE):
                print(f"⚠️  WARNING: Invalid campaign ID format '{cid}' (expected Cxxxx).")
                continue
            campaign_obj = find_campaign_by_external_id(bundle, cid.upper())
            if not campaign_obj:
                print(f"⚠️  WARNING: Campaign '{cid}' not found in dataset.")
                continue
            tids, descs = collect_techniques_for_sources(bundle, [campaign_obj["id"]])
            auto_mark_ids.update(tids)
            evidence_records.append({"campaign": cid.upper(), "techniques": descs})

    # ------------------------------------------------------------------
    # Output handling – always write to daily timestamped output folder
    # ------------------------------------------------------------------
    today_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(os.getcwd(), f"output_{today_str}")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "technique_relevance.csv")
    evidence_path = os.path.join(output_dir, "technique_relevance_evidence.json")

    try:
        write_csv(grouped, csv_path, auto_mark_ids=auto_mark_ids, mark_all=args.mark_all)
        print(f"\n📁 Output directory: {output_dir}")
        print(f"✅ Wrote: {csv_path}")
        print("   Columns: Tactic, Technique ID, Technique, Relevant")
        print(f"   Techniques auto-marked: {len(auto_mark_ids)}")

        if args.procedure or args.campaign:
            with open(evidence_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "procedures_used": args.procedure or [],
                        "campaigns_used": args.campaign or [],
                        "auto_marked_count": len(auto_mark_ids),
                        "evidence": evidence_records,
                    },
                    f,
                    indent=2,
                )
            print(f"📝 Wrote evidence file: {evidence_path}")

    except Exception as e:
        print(f"ERROR: Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
