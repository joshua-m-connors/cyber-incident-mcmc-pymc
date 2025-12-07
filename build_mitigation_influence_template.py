#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================================
MITRE ATT&CK Mitigation Influence Template Builder (Instructional Version)
=====================================================================================
This script constructs a structured CSV file (`mitigation_control_strengths.csv`) that
quantifies how each MITRE ATT&CK mitigation influences techniques and tactics. It is
intended to help analysts understand how mitigations map to ATT&CK data, and to seed
control strength modeling in later FAIR–MITRE ATT&CK quantitative risk simulations.

--------------------------------------------
KEY TASKS PERFORMED BY THIS SCRIPT
--------------------------------------------
1. Loads a MITRE ATT&CK STIX bundle (typically `enterprise-attack.json`).
2. Extracts mitigations, attack techniques, and relationships (which mitigates which).
3. Computes:
   • Number of techniques mitigated by each mitigation.
   • Number of tactics covered (unique MITRE tactics linked to those techniques).
   • A normalized weight (0–1) indicating relative influence or coverage.
4. Assigns default control strength ranges (Control_Min / Control_Max) to each mitigation.
   • These are rough seed values used in later dashboards and modeling scripts.
   • Certain mitigations (like “Audit”) are underweighted intentionally.
5. Outputs a CSV template and a simple build log.

--------------------------------------------
HOW TO USE / MODIFY
--------------------------------------------
- Ensure `enterprise-attack.json` (MITRE ATT&CK bundle) is in the same directory.
- Run this script directly to produce `mitigation_control_strengths.csv`.
- The default output directory for logs is auto-created as `output_YYYY-MM-DD/`.

--------------------------------------------
USER-TUNABLE PARAMETERS
--------------------------------------------
- `default_ranges`: Default min/max control strength seeds (in percent).
- Analysts can later edit the generated CSV to:
  • Override Control_Min / Control_Max.
  • Add more nuanced dependency and group health information for the model.

--------------------------------------------
OUTPUT FILES
--------------------------------------------
- `mitigation_control_strengths.csv` : Generated CSV for downstream modeling.
- `mitigation_template_build_log_<timestamp>.txt` : Basic log of build results.
=====================================================================================
"""

import os
import json
import random
import pandas as pd
from datetime import datetime

# ─────────────────────────────
# Simple ANSI color helpers
# ─────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

# ─────────────────────────────
# Paths and defaults
# ─────────────────────────────
DATASET_PATH = "enterprise-attack.json"

def make_output_dir(prefix: str = "output") -> str:
    """Create daily output directory."""
    base = os.getcwd()
    today = datetime.now().strftime("%Y-%m-%d")
    out = os.path.join(base, f"{prefix}_{today}")
    os.makedirs(out, exist_ok=True)
    return out

OUTPUT_DIR = make_output_dir("output")

# CSV output is placed in the base folder (not output dir) since users will
# typically open and edit it for calibration.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV  = os.path.join(BASE_DIR, "mitigation_control_strengths.csv")

# ──────────────────────────────────────────────────────────────────────────────
def _load_stix_objects(dataset_path: str):
    """
    Load STIX objects from a MITRE ATT&CK bundle.

    Returns:
      techniques: dict[id -> attack-pattern object]
      mitigations: dict[id -> course-of-action object]
      relationships: list[relationship objects of type "mitigates"]
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    techniques = {}
    mitigations = {}
    relationships = []

    for obj in data.get("objects", []):
        t = obj.get("type")
        if t == "attack-pattern":
            techniques[obj["id"]] = obj
        elif t == "course-of-action":
            mitigations[obj["id"]] = obj
        elif t == "relationship" and obj.get("relationship_type") == "mitigates":
            relationships.append(obj)

    return techniques, mitigations, relationships

def _technique_tactics_map(techniques: dict) -> dict:
    """
    Build a mapping: technique_id -> set of tactics (strings).
    """
    tech_to_tactics = {}
    for tid, tobj in techniques.items():
        phases = tobj.get("kill_chain_phases", [])
        tactic_names = set()
        for ref in phases:
            tactic = ref.get("phase_name", "").replace("-", " ").title()
            if tactic:
                tactic_names.add(tactic)
        tech_to_tactics[tid] = tactic_names
    return tech_to_tactics

def _mitigation_technique_map(relationships: list) -> dict:
    """
    Build a mapping: mitigation_id -> set of technique_ids mitigated.
    """
    mit_to_techs = {}
    for rel in relationships:
        src = rel.get("source_ref")
        tgt = rel.get("target_ref")
        if not src or not tgt:
            continue
        mit_to_techs.setdefault(src, set()).add(tgt)
    return mit_to_techs

def _mitigation_external_id(mit_obj: dict) -> str:
    """
    Extract MITRE external_id (like 'M1047') from a mitigation object.
    Returns an empty string if not found.
    """
    for ref in mit_obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", "").strip()
    return ""

# ──────────────────────────────────────────────────────────────────────────────
def build_mitigation_template(dataset_path: str = DATASET_PATH) -> None:
    """
    Build the mitigation influence template and write to CSV.

    The CSV will include:
      - Mitigation_ID
      - Mitigation_Name
      - Techniques_Mitigated
      - Tactics_Covered
      - Weight
      - Control_Min, Control_Max
      - Control_Type (MITRE vs SUPPORT)
      - Dependency_Group, Group_Health_Min, Group_Health_Max
      - Requires, Requires_Influence_Min, Requires_Influence_Max
    """
    print(f"{GREEN}Loading ATT&CK dataset:{RESET} {dataset_path}")

    techniques, mitigations, relationships = _load_stix_objects(dataset_path)
    tech_to_tactics = _technique_tactics_map(techniques)
    mit_to_techs = _mitigation_technique_map(relationships)

    print(f"{GREEN}Techniques loaded:{RESET}   {len(techniques)}")
    print(f"{GREEN}Mitigations loaded:{RESET}  {len(mitigations)}")
    print(f"{GREEN}Relationships loaded:{RESET} {len(relationships)}")

    # Default control strength ranges (in percent) used as seed values
    default_ranges = [
        (30, 60),
        (40, 70),
        (50, 80),
    ]

    # Compute a simple influence metric: number of techniques mitigated
    max_count = 0
    for mid in mitigations.keys():
        linked_techs = mit_to_techs.get(mid, set())
        if len(linked_techs) > max_count:
            max_count = len(linked_techs)

    if max_count == 0:
        max_count = 1  # avoid divide-by-zero

    rows = []
    for mid, mobj in mitigations.items():
        name = mobj.get("name", "Unknown")
        m_id = _mitigation_external_id(mobj)

        # Skip irrelevant or placeholder mitigations
        if not m_id or name.strip().lower() == "do not mitigate":
            continue

        # Compute counts for reporting
        linked_techs = mit_to_techs.get(mid, set())
        techniques_mitigated = len(linked_techs)

        # Identify all unique tactics those techniques belong to
        tactics_linked = set()
        for t in linked_techs:
            tactics_linked |= tech_to_tactics.get(t, set())
        tactics_covered = len(tactics_linked)

        # Normalized weight: higher for mitigations covering more techniques
        weight = round(techniques_mitigated / max_count, 3) if max_count else 0.0

        # Assign a base control strength seed range (lo, hi)
        lo, hi = random.choice(default_ranges)

        # Special case: lower default strength for generic mitigations like “Audit”
        if name.strip().lower() == "audit":
            lo, hi = int(lo * 0.5), int(hi * 0.5)

        # Append computed record
        rows.append({
            "Mitigation_ID": m_id,
            "Mitigation_Name": name,
            "Techniques_Mitigated": techniques_mitigated,
            "Tactics_Covered": tactics_covered,
            "Weight": weight,
            "Control_Min": lo,
            "Control_Max": hi,
            # New dependency-related fields (analyst-editable in the CSV)
            # Control_Type: "MITRE" for native ATT&CK mitigations, "SUPPORT" for non-MITRE / governance controls
            "Control_Type": "MITRE",
            # Dependency_Group: optional logical stack name (e.g., "IAM_Stack1"); blank means no grouping
            "Dependency_Group": "",
            # Group_Health_Min/Max: optional group health range in [0..1]; blank means neutral health (1.0)
            "Group_Health_Min": "",
            "Group_Health_Max": "",
            # Requires: optional space-separated list of Mitigation_ID values this control depends on
            "Requires": "",
            # Requires_Influence_Min/Max: optional influence weight range in [0..1]; blank → full dependence (1.0)
            "Requires_Influence_Min": "",
            "Requires_Influence_Max": "",
        })

    # Convert list to DataFrame and sort by influence metrics
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(by=["Weight", "Techniques_Mitigated"], ascending=[False, False], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Write CSV
    df.to_csv(OUT_CSV, index=False)
    print(f"{GREEN}Template CSV written to:{RESET} {OUT_CSV}")

    # Write a simple build log
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(OUTPUT_DIR, f"mitigation_template_build_log_{ts}.txt")
    with open(log_path, "w", encoding="utf-8") as rep:
        rep.write("MITRE ATT&CK Mitigation Influence Template Build Log\n")
        rep.write("===================================================\n\n")
        rep.write(f"Dataset: {dataset_path}\n")
        rep.write(f"Timestamp: {ts}\n\n")
        rep.write(f"Techniques: {len(techniques)}\n")
        rep.write(f"Mitigations: {len(mitigations)}\n")
        rep.write(f"Mitigation relationships: {len(relationships)}\n")
        rep.write(f"Template saved: {OUT_CSV}\n")
    print(f"{GREEN}📝 Log saved to:{RESET} {log_path}")

# ──────────────────────────────────────────────────────────────────────────────
def main():
    try:
        build_mitigation_template(DATASET_PATH)
    except FileNotFoundError:
        print(f"{RED}❌ Could not find dataset file:{RESET} {DATASET_PATH}")
    except json.JSONDecodeError as e:
        print(f"{RED}❌ JSON parse error in dataset:{RESET} {e}")
    except Exception as e:
        print(f"{RED}❌ Unhandled error:{RESET} {e}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
