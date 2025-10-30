#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================================
MITRE ATT&CK Control Strength Dashboard (Instructional Version)
=====================================================================================
This script visualizes the relative control strengths of mitigations across MITRE ATT&CK
tactics. It combines data from the MITRE ATT&CK STIX bundle and a CSV file of control
strength ranges to produce a weighted view of security coverage.

--------------------------------------------
KEY FUNCTIONS
--------------------------------------------
1. **Data Loading:** Extracts attack techniques, mitigations, and relationships from the
   MITRE ATT&CK dataset.
2. **Data Merging:** Merges control strength data (from CSV) with MITRE mitigations.
3. **Weighting Logic:** Adjusts mitigation influence based on type (e.g., Decision Support 
   or Variance Management (see FAIR-CAM) controls like "Audit" or "User Training" receive 
   less weight).
4. **Visualization:** Creates a Plotly grouped bar chart showing minimum and maximum
   control strength for each MITRE tactic.
5. **Optional Debug Mode:** Verifies that the normalized influence values sum to ~100%
   per tactic.

--------------------------------------------
USER-TUNABLE PARAMETERS
--------------------------------------------
- `DISCOUNT_CONTROLS`: A list of mitigation names to underweight (e.g., user training).
- `DEBUG_INFLUENCE_SUM`: If True, prints per-tactic influence totals for diagnostic use.
- `CSV_PATH`: Default input CSV of mitigation control strength values.
- `OUTPUT_DIR`: Automatically created directory for output charts.

--------------------------------------------
OUTPUTS
--------------------------------------------
- **HTML dashboard** saved to: `output_YYYY-MM-DD/mitre_tactic_strengths_<timestamp>.html`
- **Console table** of average strength per tactic.
- **Optional debug messages** (if `DEBUG_INFLUENCE_SUM=True`).

--------------------------------------------
PURPOSE
--------------------------------------------
This tool is typically used as an intermediate diagnostic and visualization step before
running quantitative simulations (e.g., PyMC-based FAIR risk models). It allows users to
see which tactics have stronger or weaker defensive coverage based on weighted averages
of mitigation effectiveness.
=====================================================================================
"""

import os
import sys
import json
import random
import datetime
import pandas as pd
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────
QUIET_MODE = not sys.argv[0].endswith("mitre_control_strength_dashboard.py")
DEBUG_INFLUENCE_SUM = False  # ← Set to True to print influence normalization checks

# NOTE: You can modify this list to discount the influence of certain controls
# that are broad or less direct (e.g., training, auditing, or intelligence programs).
DISCOUNT_CONTROLS = [
    "audit",
    "vulnerability scanning",
    "user training",
    "threat intelligence program",
    "application developer guidance",
]

# Default input dataset and CSV (you can override these if needed)
DATASET_PATH = "enterprise-attack.json"
CSV_PATH = "mitigation_control_strengths.csv"

# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY SETUP
# ──────────────────────────────────────────────────────────────────────────────
today = datetime.date.today().strftime("%Y-%m-%d")
OUTPUT_DIR = os.path.join(os.getcwd(), f"output_{today}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
def log(message):
    """Helper function for controlled console printing."""
    if not QUIET_MODE:
        print(message)

# ──────────────────────────────────────────────────────────────────────────────
def _load_stix_objects(dataset_path):
    """Load STIX data from a MITRE ATT&CK JSON bundle.

    Args:
        dataset_path (str): Path to ATT&CK bundle (e.g., enterprise-attack.json)
    Returns:
        tuple(dict, dict, list): (techniques, mitigations, relationships)
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    techniques, mitigations, relationships = {}, {}, []

    for obj in data["objects"]:
        if obj["type"] == "attack-pattern":
            techniques[obj["id"]] = obj
        elif obj["type"] == "course-of-action":
            mitigations[obj["id"]] = obj
        elif obj["type"] == "relationship" and obj.get("relationship_type") == "mitigates":
            relationships.append(obj)

    return techniques, mitigations, relationships

# ──────────────────────────────────────────────────────────────────────────────
def get_mitre_tactic_strengths(dataset_path=DATASET_PATH, csv_path=CSV_PATH,
                               seed=42, build_figure=True):
    """Main function: compute weighted control strengths and optionally build dashboard.

    Args:
        dataset_path (str): Path to MITRE ATT&CK JSON dataset.
        csv_path (str): Path to mitigation control strength CSV.
        seed (int): Random seed for deterministic weighting.
        build_figure (bool): If True, create a Plotly dashboard.

    Returns:
        tuple(pd.DataFrame, plotly.Figure | None):
            - DataFrame of averaged control strengths by tactic.
            - Plotly figure (if generated).
    """
    random.seed(seed)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # STEP 1: Load MITRE dataset objects (techniques, mitigations, relationships)
    try:
        techniques, mitigations, relationships = _load_stix_objects(dataset_path)
        log(f"\033[92m✅ Loaded {len(mitigations)} mitigations from dataset\033[0m")
    except Exception as e:
        log(f"\033[93m⚠️ Failed to load MITRE dataset: {e}\033[0m")
        return pd.DataFrame(), None

    # STEP 2: Build tactic → mitigations mapping
    tactic_map = {}
    for rel in relationships:
        src, tgt = rel.get("source_ref"), rel.get("target_ref")
        if src in mitigations and tgt in techniques:
            tech = techniques[tgt]
            for ref in tech.get("kill_chain_phases", []):
                tactic = ref.get("phase_name", "").replace("-", " ").title()
                if tactic:
                    tactic_map.setdefault(tactic, []).append(src)

    # STEP 3: Load control strength data from CSV (if available)
    try:
        csv = pd.read_csv(csv_path)
        csv["Mitigation_ID"] = csv["Mitigation_ID"].astype(str).str.strip().str.lower()

        # Build quick lookup dictionary for min/max control strengths
        csv_map = {}
        for mid, lo, hi in zip(csv["Mitigation_ID"], csv["Control_Min"], csv["Control_Max"]):
            csv_map[mid] = (float(lo), float(hi))
            if mid.startswith("m") and len(mid) <= 6:
                csv_map[mid.replace(" ", "").lower()] = (float(lo), float(hi))

        log(f"\033[92m✅ Loaded {len(csv_map)} mitigation strengths from {csv_path}\033[0m")
    except Exception as e:
        log(f"\033[93m⚠️ Could not load mitigation strengths ({e})\033[0m")
        csv_map = {}

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4: Compute average min/max strengths per tactic
    rows, summary_rows = [], []

    for tactic, mit_ids in tactic_map.items():
        if not mit_ids:
            rows.append((tactic, 0.0, 0.0, "No linked mitigations"))
            summary_rows.append((tactic, 0.0, 0.0, 0))
            continue

        weighted = []

        # Iterate over each mitigation linked to this tactic
        for mid in mit_ids:
            mit = mitigations[mid]

            # Extract external MITRE ID (e.g., M1030)
            ext_id = None
            for ref in mit.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    ext_id = ref.get("external_id", "").strip().lower()
                    break

            # Retrieve control strength range (default to 30–70 if missing)
            lo, hi = (
                csv_map.get(mid.lower())
                or (csv_map.get(ext_id) if ext_id else None)
                or (30.0, 70.0)
            )

            name = mit["name"]

            # Apply weighting factor (discounting generic mitigations)
            name_lower = name.lower()
            if "do not mitigate" in name_lower:
                lo, hi = 0.0, 0.0
                weight_factor = 1.0
            elif any(ctrl in name_lower for ctrl in DISCOUNT_CONTROLS):
                weight_factor = 0.5
            else:
                weight_factor = 1.0

            weighted.append((lo, hi, name, weight_factor))

        # Weighted averages (normalized by total influence weight)
        total_weight = max(1e-9, sum(w[3] for w in weighted))
        avg_min = sum(w[0] * w[3] for w in weighted) / total_weight
        avg_max = sum(w[1] * w[3] for w in weighted) / total_weight

        # Construct hover text showing per-mitigation influence
        hover_dict = {}
        for lo, hi, name, weight_factor in weighted:
            if name not in hover_dict:
                hover_dict[name] = {"lo": lo, "hi": hi, "count": 1, "weight": weight_factor}
            else:
                hover_dict[name]["lo"] = (hover_dict[name]["lo"] + lo) / 2
                hover_dict[name]["hi"] = (hover_dict[name]["hi"] + hi) / 2
                hover_dict[name]["count"] += 1
                hover_dict[name]["weight"] += weight_factor

        # Normalize influence percentages (for hover display)
        total_weighted_refs = sum(v["weight"] for v in hover_dict.values())
        details = [
            f"{name}: {v['lo']:.1f}–{v['hi']:.1f}% (influence {100*v['weight']/total_weighted_refs:.1f}%)"
            for name, v in sorted(hover_dict.items())
        ]
        hover_text = "<br>".join(details)

        # Optional debug verification
        if DEBUG_INFLUENCE_SUM:
            influence_sum = sum(100 * v["weight"] / total_weighted_refs for v in hover_dict.values())
            log(f"[DEBUG] {tactic}: Total influence = {influence_sum:.2f}%")

        rows.append((tactic, avg_min, avg_max, hover_text))
        summary_rows.append((tactic, avg_min, avg_max, len(weighted)))

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5: Convert results to DataFrames and order by ATT&CK flow
    df = pd.DataFrame(rows, columns=["Tactic", "MinStrength", "MaxStrength", "MitigationList"])
    summary_df = pd.DataFrame(summary_rows, columns=["Tactic", "MinStrength", "MaxStrength", "MitigationCount"])

    order = [
        "Initial Access", "Execution", "Persistence", "Privilege Escalation",
        "Defense Evasion", "Credential Access", "Discovery", "Lateral Movement",
        "Collection", "Command And Control", "Exfiltration", "Impact"
    ]

    for d in (df, summary_df):
        d["Tactic"] = pd.Categorical(d["Tactic"], categories=order, ordered=True)
        d.dropna(subset=["Tactic"], inplace=True)
        d.sort_values("Tactic", inplace=True)
        d.reset_index(drop=True, inplace=True)

    # Print tabular summary to console
    if not QUIET_MODE:
        log("\n--- Weighted Control Strengths (ordered by MITRE flow) ---")
        log(summary_df.to_string(index=False))

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 6: Plot visualization (grouped bar chart)
    fig = None
    if build_figure:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["Tactic"],
            y=df["MinStrength"],
            name="Min Strength (%)",
            marker_color="skyblue",
            hovertext=df["MitigationList"],
            hoverinfo="text+y"
        ))
        fig.add_trace(go.Bar(
            x=df["Tactic"],
            y=df["MaxStrength"],
            name="Max Strength (%)",
            marker_color="steelblue",
            hovertext=df["MitigationList"],
            hoverinfo="text+y"
        ))

        # Layout customization for readability
        fig.update_layout(
            title="Weighted MITRE ATT&CK Control Strengths by Tactic",
            xaxis_title="Tactic",
            yaxis_title="Control Strength (%)",
            barmode="group",
            template="plotly_white",
            height=650,
            hoverlabel=dict(font_size=11, font_family="Courier New")
        )

        # Optional enhanced hover clarity
        fig.update_traces(
            hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.1f}%<br>%{customdata}<extra></extra>",
            customdata=df["MitigationList"],
            hoverlabel=dict(namelength=-1, font_size=11, font_family="Courier New")
        )

        # Save dashboard to HTML
        fig_path = os.path.join(OUTPUT_DIR, f"mitre_tactic_strengths_{ts}.html")
        fig.write_html(fig_path)
        log(f"\033[92m✅ Chart saved → {fig_path}\033[0m")

        if not QUIET_MODE:
            fig.show()

    return df, fig

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, fig = get_mitre_tactic_strengths(build_figure=True)
