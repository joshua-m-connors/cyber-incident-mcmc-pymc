#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================================
MITRE ATT&CK Mitigation Influence Template Builder (Instructional Version)
=====================================================================================
This script constructs a structured CSV file (`mitigation_influence_template.csv`) that
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
- Run this script directly to produce `mitigation_influence_template.csv`.
- The default output directory for logs is auto-created as `output_YYYY-MM-DD/`.

--------------------------------------------
USER-TUNABLE PARAMETERS
--------------------------------------------
- `default_ranges`: Default min/max control strength seeds (in percent).
  Adjust if you want a wider or narrower spread of initial strengths.
- Random seed (set via `random.seed(42)`) controls deterministic behavior.

--------------------------------------------
OUTPUT FILES
--------------------------------------------
- `mitigation_influence_template.csv` : Generated CSV for downstream modeling.
- `mitigation_template_build_log_<timestamp>.txt` : Basic log of build results.
=====================================================================================
"""

import os
import json
import random
import pandas as pd
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# ANSI color codes for pretty console output (optional aesthetic only)
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

# ──────────────────────────────────────────────────────────────────────────────
# Output directory setup
# This creates a date-labeled folder for log/report output, ensuring results
# stay organized by execution date.
def make_output_dir(prefix: str = "output") -> str:
    """Create (if needed) and return the output directory path."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"{GREEN}📁 Output directory:{RESET} {out_dir}")
    return out_dir

OUTPUT_DIR = make_output_dir("output")

# CSV output is placed in the base folder (not output dir) since users will
# typically open and edit it for calibration.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_CSV  = os.path.join(BASE_DIR, "mitigation_influence_template.csv")

# ──────────────────────────────────────────────────────────────────────────────
def _load_stix_objects(dataset_path: str):
    """Load the MITRE ATT&CK STIX bundle and return parsed objects.

    Args:
        dataset_path (str): Path to ATT&CK JSON file (e.g., enterprise-attack.json).

    Returns:
        tuple(dict, dict, list):
            - techniques: dict of attack-pattern objects
            - mitigations: dict of course-of-action objects
            - relationships: list of mitigation relationships
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # STIX bundles typically contain an "objects" list
    objs = data["objects"] if isinstance(data, dict) and "objects" in data else data

    # Separate by object type
    techniques    = {o["id"]: o for o in objs if o.get("type") == "attack-pattern"}
    mitigations   = {o["id"]: o for o in objs if o.get("type") == "course-of-action"}
    relationships = [
        o for o in objs
        if o.get("type") == "relationship" and o.get("relationship_type") == "mitigates"
    ]

    print(f"{GREEN}✅ Loaded {len(mitigations)} mitigations from dataset{RESET}")
    return techniques, mitigations, relationships

# ──────────────────────────────────────────────────────────────────────────────
def _mitigation_external_id(mobj: dict) -> str | None:
    """Extract the mitigation’s external MITRE ID (e.g., M1031)."""
    for r in mobj.get("external_references", []) or []:
        ext = r.get("external_id", "")
        if isinstance(ext, str) and ext.startswith("M"):
            return ext
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Random seed ensures reproducibility of random range selections.
random.seed(42)

def _build_template(techniques: dict, mitigations: dict, relationships: list[dict]) -> pd.DataFrame:
    """Constructs the mitigation influence template.

    Steps:
      1. Map each mitigation to the techniques it mitigates.
      2. Map each technique to its associated MITRE tactics.
      3. Count techniques and tactics per mitigation.
      4. Compute normalized weights (based on relative technique coverage).
      5. Assign randomized default control strength ranges.

    Returns:
        pd.DataFrame: Mitigation influence template ready for CSV export.
    """

    # Build mapping: mitigation → techniques
    from collections import defaultdict
    mit_to_techs = defaultdict(set)
    for rel in relationships:
        src, tgt = rel.get("source_ref", ""), rel.get("target_ref", "")
        if src.startswith("course-of-action") and tgt.startswith("attack-pattern"):
            mit_to_techs[src].add(tgt)

    # Build mapping: technique → tactics (phases)
    tech_to_tactics: dict[str, set[str]] = {}
    for tid, tobj in techniques.items():
        phases = tobj.get("kill_chain_phases", []) or []
        tactics = [p.get("phase_name", "").replace("-", " ").title()
                   for p in phases if p.get("phase_name")]
        tech_to_tactics[tid] = set(tactics)

    # Count techniques per mitigation to normalize influence weights
    tech_counts = {mid: len(ts) for mid, ts in mit_to_techs.items()}
    max_count = max(tech_counts.values()) if tech_counts else 1

    # USER TIP: Adjust default_ranges if you want to bias initial control values.
    # These are percentage-based placeholders (later capped in dashboards).
    default_ranges = [(30, 70), (35, 65), (40, 60)]

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
        })

    # Convert list to DataFrame and sort by influence metrics
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(by=["Weight", "Techniques_Mitigated", "Tactics_Covered"],
                       ascending=[False, False, False], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

# ──────────────────────────────────────────────────────────────────────────────
def main():
    """Main execution: load dataset, compute template, and save results."""
    DATASET_PATH = "enterprise-attack.json"

    try:
        # Load data and build influence matrix
        techniques, mitigations, relationships = _load_stix_objects(DATASET_PATH)
        df = _build_template(techniques, mitigations, relationships)

        if df.empty:
            print(f"{YELLOW}⚠️ No mitigations available to write. Check dataset/filters.{RESET}")
        else:
            # Save to CSV (editable template)
            df.to_csv(OUT_CSV, index=False)
            print(f"{GREEN}✅ Saved {len(df)} mitigations to:{RESET} {OUT_CSV}")
            print(f"{BLUE}📊 Weight range:{RESET} {df['Weight'].min():.3f} → {df['Weight'].max():.3f}")
            print(f"{CYAN}🏆 Top mitigation:{RESET} {df.iloc[0]['Mitigation_ID']} ({df.iloc[0]['Weight']:.3f})  "
                  f"— Lowest: {df.iloc[-1]['Mitigation_ID']} ({df.iloc[-1]['Weight']:.3f})")

        # Save basic log file summarizing build run
        log_path = os.path.join(
            OUTPUT_DIR, f"mitigation_template_build_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        )
        with open(log_path, "w", encoding="utf-8") as rep:
            rep.write(f"MITRE dataset: {DATASET_PATH}\n")
            rep.write(f"Mitigations loaded: {len(mitigations)}\n")
            rep.write(f"Mitigation relationships: {len(relationships)}\n")
            rep.write(f"Template saved: {OUT_CSV}\n")
        print(f"{GREEN}📝 Log saved to:{RESET} {log_path}")

    except FileNotFoundError as e:
        print(f"{RED}❌ File not found:{RESET} {e}")
    except json.JSONDecodeError as e:
        print(f"{RED}❌ JSON parse error in dataset:{RESET} {e}")
    except Exception as e:
        print(f"{RED}❌ Unhandled error:{RESET} {e}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
