#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2025 Joshua Connors
"""
MITRE ATT&CK Control Strength Dashboard — Stable Hover Edition
- Multiline hover via per-point hovertemplate (array of strings)
- Plain-text bullet list (•) with <br> separators (no HTML containers)
- Caps hover items to prevent off-screen overflow (top 20 + "… and N more")
- Unified hover; Min bar hover disabled
"""

from __future__ import annotations
import os
import json
import random
import datetime
import argparse
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go

# ----------------------- Config -----------------------
DISCOUNT_CONTROLS: List[str] = [
    "audit",
    "vulnerability scanning",
    "user training",
    "threat intelligence program",
    "application developer guidance",
]

DATASET_PATH = "enterprise-attack.json"
CSV_PATH = "mitigation_control_strengths.csv"
USE_RELEVANCE_CSV = True
TECHNIQUE_RELEVANCE_FILE = "technique_relevance.csv"
MAX_HOVER_ITEMS = 20  # cap number of mitigation lines shown per tactic

TODAY_STR = datetime.date.today().strftime("%Y-%m-%d")
OUTPUT_DIR = os.path.join(os.getcwd(), f"output_{TODAY_STR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg: str, quiet: bool = False):
    if not quiet:
        print(msg)

def load_stix_objects(dataset_path: str):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    techniques, mitigations, relationships = {}, {}, []
    for obj in data.get("objects", []):
        t = obj.get("type")
        if t == "attack-pattern":
            techniques[obj["id"]] = obj
        elif t == "course-of-action":
            mitigations[obj["id"]] = obj
        elif t == "relationship" and obj.get("relationship_type") == "mitigates":
            relationships.append(obj)
    return techniques, mitigations, relationships

def build_tactic_map_full(techniques, mitigations, relationships):
    tactic_map = {}
    for rel in relationships:
        src, tgt = rel.get("source_ref"), rel.get("target_ref")
        if src in mitigations and tgt in techniques:
            tech = techniques[tgt]
            for ref in tech.get("kill_chain_phases", []):
                tactic = ref.get("phase_name", "").replace("-", " ").title()
                if tactic:
                    tactic_map.setdefault(tactic, []).append(src)
    return tactic_map

def build_tactic_map_filtered(techniques, mitigations, relationships, technique_ids_keep: Set[str], tactics_keep: Set[str]):
    tactic_map = {}
    for rel in relationships:
        src, tgt = rel.get("source_ref"), rel.get("target_ref")
        if src in mitigations and tgt in techniques:
            tech = techniques[tgt]
            ext_tid = next((r.get("external_id", "").strip()
                            for r in tech.get("external_references", [])
                            if r.get("source_name") == "mitre-attack"), None)
            if not ext_tid or ext_tid not in technique_ids_keep:
                continue
            for ref in tech.get("kill_chain_phases", []):
                tactic = ref.get("phase_name", "").replace("-", " ").title()
                if tactic and tactic in tactics_keep:
                    tactic_map.setdefault(tactic, []).append(src)
    return {t: m for t, m in tactic_map.items() if m}

def load_relevance_filter(file_path: str):
    df = pd.read_csv(file_path)
    norm = {c: c.lower().replace(" ", "").replace("_", "") for c in df.columns}
    tactic_col = next((o for o, n in norm.items() if n == "tactic"), None)
    tech_col = next((o for o, n in norm.items() if n in {"techniqueid", "technique"}), None)
    mark_col = next((o for o, n in norm.items() if n in {"relevant", "include", "selected"}), None)
    if not tactic_col or not tech_col:
        raise ValueError("technique_relevance.csv must include 'Tactic' and 'Technique ID'.")
    if not mark_col:
        raise ValueError("Missing relevance column (e.g., 'Relevant').")
    mask = df[mark_col].astype(str).str.strip().str.upper().eq("X")
    kept = df.loc[mask, [tactic_col, tech_col]].copy()
    kept[tactic_col] = kept[tactic_col].astype(str).str.strip()
    kept[tech_col] = kept[tech_col].astype(str).str.strip()
    return set(kept[tech_col].dropna().unique()), set(kept[tactic_col].dropna().unique())

def compute_tactic_strengths(tactic_map, mitigations, strengths_map, discount_controls=None):
    discount_controls = discount_controls or []
    detail_rows, summary_rows = [], []
    for tactic, mit_ids in tactic_map.items():
        if not mit_ids:
            detail_rows.append((tactic, 0.0, 0.0, []))
            summary_rows.append((tactic, 0.0, 0.0, 0))
            continue

        # collect (name, lo, hi, w) per relationship
        entries = []
        for mid in mit_ids:
            mit = mitigations[mid]
            name = mit.get("name", "Unknown Mitigation")
            name_lower = name.lower()
            ext_id = next((r.get("external_id", "").strip().lower()
                           for r in mit.get("external_references", [])
                           if r.get("source_name") == "mitre-attack"), None)
            lo, hi = strengths_map.get(mid.lower()) or strengths_map.get(ext_id) or (30.0, 70.0)
            if "do not mitigate" in name_lower:
                lo, hi, w = 0.0, 0.0, 1.0
            elif any(k in name_lower for k in discount_controls):
                w = 0.5
            else:
                w = 1.0
            entries.append((name, lo, hi, w))

        # aggregate duplicates by name
        agg = defaultdict(lambda: [0.0, 0.0, 0.0])
        for name, lo, hi, w in entries:
            agg[name][0] += lo * w
            agg[name][1] += hi * w
            agg[name][2] += w

        weighted_unique = []
        for name, (sum_lo, sum_hi, sum_w) in agg.items():
            avg_lo = sum_lo / max(1e-9, sum_w)
            avg_hi = sum_hi / max(1e-9, sum_w)
            weighted_unique.append((avg_lo, avg_hi, name, sum_w))

        # tactic-level averages
        total_w = max(1e-9, sum(w for *_, w in weighted_unique))
        avg_min = sum(lo * w for lo, hi, n, w in weighted_unique) / total_w
        avg_max = sum(hi * w for lo, hi, n, w in weighted_unique) / total_w

        # build mitigation lines sorted by influence desc
        items_sorted = sorted(weighted_unique, key=lambda t: t[3], reverse=True)
        lines = [f"{n}: {lo:.1f}–{hi:.1f}% (influence {(w/total_w)*100.0:.1f}%)"
                 for lo, hi, n, w in items_sorted]

        detail_rows.append((tactic, avg_min, avg_max, lines))
        summary_rows.append((tactic, avg_min, avg_max, len(items_sorted)))

    detail_df = pd.DataFrame(detail_rows, columns=["Tactic", "MinStrength", "MaxStrength", "MitigationLines"])
    summary_df = pd.DataFrame(summary_rows, columns=["Tactic", "MinStrength", "MaxStrength", "MitigationCount"])

    order = [
        "Initial Access", "Execution", "Persistence", "Privilege Escalation",
        "Defense Evasion", "Credential Access", "Discovery", "Lateral Movement",
        "Collection", "Command And Control", "Exfiltration", "Impact"
    ]
    for d in (detail_df, summary_df):
        d["Tactic"] = pd.Categorical(d["Tactic"], categories=order, ordered=True)
        d.dropna(subset=["Tactic"], inplace=True)
        d.sort_values("Tactic", inplace=True)
        d.reset_index(drop=True, inplace=True)

    return detail_df, summary_df

def get_mitre_tactic_strengths(dataset_path: str = DATASET_PATH,
                               csv_path: str = CSV_PATH,
                               seed: int = 42,
                               build_figure: bool = True,
                               use_relevance: bool = False,
                               relevance_file: str = TECHNIQUE_RELEVANCE_FILE,
                               quiet: bool = False):
    random.seed(seed)
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    try:
        techniques, mitigations, relationships = load_stix_objects(dataset_path)
        log(f"✅ Loaded {len(mitigations)} mitigations and {len(relationships)} relationships.", quiet)
    except Exception as e:
        log(f"⚠️ Failed to load dataset: {e}", quiet)
        return pd.DataFrame(), pd.DataFrame(), {}, {}, {}

    if use_relevance and os.path.exists(relevance_file):
        try:
            tech_keep, tact_keep = load_relevance_filter(relevance_file)
            tactic_map = build_tactic_map_filtered(techniques, mitigations, relationships, tech_keep, tact_keep)
            log(f"✅ FILTERED mode — {len(tech_keep)} techniques, {len(tact_keep)} tactics.", quiet)
        except Exception as e:
            log(f"⚠️ Filter load error ({e}) — reverting to FULL mode.", quiet)
            tactic_map = build_tactic_map_full(techniques, mitigations, relationships)
            use_relevance = False
    else:
        tactic_map = build_tactic_map_full(techniques, mitigations, relationships)
        log("⚙️ FULL mode (no filtering).", quiet)

    if use_relevance and not tactic_map:
        log("⚠️ No tactics retained after filtering.", quiet)
        return pd.DataFrame(), pd.DataFrame(), {}, {}, {}

    try:
        csv = pd.read_csv(csv_path)
        csv["Mitigation_ID"] = csv["Mitigation_ID"].astype(str).str.strip().str.lower()
        csv_map = {mid: (float(lo), float(hi))
                   for mid, lo, hi in zip(csv["Mitigation_ID"], csv["Control_Min"], csv["Control_Max"])}
        log(f"✅ Loaded {len(csv_map)} mitigation strengths.", quiet)
    except Exception as e:
        log(f"⚠️ Could not load strengths ({e}) — defaulting to 30–70%.", quiet)
        csv_map = {}

    detail_df, summary_df = compute_tactic_strengths(tactic_map, mitigations, csv_map, DISCOUNT_CONTROLS)

    # impact-reduction controls
    impact_reduction_controls = {}
    for mit in mitigations.values():
        name_lower = mit.get("name", "").lower()
        if "data backup" in name_lower or "encrypt sensitive information" in name_lower:
            ext_id = next((r.get("external_id", "").strip().lower()
                           for r in mit.get("external_references", [])
                           if r.get("source_name") == "mitre-attack"), None)
            lo, hi = csv_map.get(ext_id) or (30.0, 70.0)
            impact_reduction_controls[mit["name"]] = {
                "min_strength": lo,
                "max_strength": hi,
                "mean_strength": (lo + hi) / 2
            }

    control_strength_map = {
        row["Tactic"]: {
            "min_strength": row["MinStrength"],
            "max_strength": row["MaxStrength"],
            "mean_strength": (row["MinStrength"] + row["MaxStrength"]) / 2,
            "mitigation_count": int(row["MitigationCount"])
        }
        for _, row in summary_df.iterrows()
    }
    relevance_metadata = {"mode": "Filtered" if use_relevance else "Full",
                          "included_tactics": list(summary_df["Tactic"]),
                          "timestamp": ts}

    # ---------- Visualization ----------
    if build_figure and not detail_df.empty:
        # Build per-point hovertemplate strings (fully formatted; no HTML containers)
        hover_templates = []
        for _, row in detail_df.iterrows():
            lines = row["MitigationLines"]
            extra = ""
            if len(lines) > MAX_HOVER_ITEMS:
                extra_count = len(lines) - MAX_HOVER_ITEMS
                lines = lines[:MAX_HOVER_ITEMS]
                extra = f"<br>… and {extra_count} more"
            bullets = "<br>".join(f"• {ln}" for ln in lines)
            hover_templates.append(
                "Tactic: %{x}<br>"
                f"Min Strength: {row['MinStrength']:.1f}%<br>"
                f"Max Strength: {row['MaxStrength']:.1f}%<br><br>"
                f"Mitigations:<br>{bullets}{extra}<extra></extra>"
            )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=detail_df["Tactic"],
            y=detail_df["MinStrength"],
            name="Min Strength (%)",
            marker_color="skyblue",
            hoverinfo="skip"
        ))
        fig.add_trace(go.Bar(
            x=detail_df["Tactic"],
            y=detail_df["MaxStrength"],
            name="Max Strength (%)",
            marker_color="steelblue",
            hovertemplate=hover_templates  # per-point fully formatted text
        ))

        mode_suffix = " (Filtered)" if use_relevance else " (Full)"
        fig.update_layout(
            title=f"Weighted MITRE ATT&CK Control Strengths by Tactic{mode_suffix}",
            xaxis_title="Tactic",
            yaxis_title="Control Strength (%)",
            barmode="group",
            template="plotly_white",
            height=650,
            hovermode="x unified",
            hoverlabel=dict(namelength=-1, font_size=11)  # keep labels compact
        )
        html_path = os.path.join(OUTPUT_DIR, f"mitre_tactic_strengths_{ts}.html")
        fig.write_html(html_path)
        log(f"✅ Chart saved → {html_path}", quiet)

    # Save summary CSV
    summary_out = summary_df.copy()
    summary_out.insert(0, "Mode", relevance_metadata["mode"])
    summary_out["Timestamp"] = ts
    out_csv = os.path.join(OUTPUT_DIR, f"filtered_summary_{ts}.csv")
    summary_out.to_csv(out_csv, index=False)
    log(f"✅ Summary CSV saved → {out_csv}", quiet)

    # --- Apply approved gating logic for impact reduction controls ---
    # Encryption mitigation only applies if Exfiltration tactic is in scope
    if 'Exfiltration' not in summary_df['Tactic'].values and 'Encrypt Sensitive Information' in impact_reduction_controls:
        impact_reduction_controls['Encrypt Sensitive Information']['min_strength'] = 0.0
        impact_reduction_controls['Encrypt Sensitive Information']['max_strength'] = 0.0
        impact_reduction_controls['Encrypt Sensitive Information']['mean_strength'] = 0.0

    # Data Backup mitigation only applies if one or more in-scope Impact techniques map to Data Backup
    has_backup_in_impact = False
    try:
        impact_rows = detail_df[detail_df['Tactic'] == 'Impact']
        if not impact_rows.empty:
            lines = impact_rows.iloc[0]['MitigationLines']
            if isinstance(lines, (list, tuple)):
                has_backup_in_impact = any('data backup' in str(ln).lower() for ln in lines)
            else:
                has_backup_in_impact = 'data backup' in str(lines).lower()
    except Exception:
        has_backup_in_impact = False

    if not has_backup_in_impact and 'Data Backup' in impact_reduction_controls:
        impact_reduction_controls['Data Backup']['min_strength'] = 0.0
        impact_reduction_controls['Data Backup']['max_strength'] = 0.0
        impact_reduction_controls['Data Backup']['mean_strength'] = 0.0

    return detail_df, summary_df, control_strength_map, relevance_metadata, impact_reduction_controls

def parse_args():
    p = argparse.ArgumentParser(description="MITRE ATT&CK Control Strength Dashboard")
    p.add_argument("--dataset", "-d", default=DATASET_PATH)
    p.add_argument("--strengths", "-s", default=CSV_PATH)
    p.add_argument("--use-relevance", "-r", action="store_true")
    p.add_argument("--relevance-file", "-f", default=TECHNIQUE_RELEVANCE_FILE)
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--show-figure", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    detail_df, summary_df, control_map, meta, impact_controls = get_mitre_tactic_strengths(
        dataset_path=args.dataset,
        csv_path=args.strengths,
        seed=42,
        build_figure=not args.no_figure,
        use_relevance=args.use_relevance or USE_RELEVANCE_CSV,
        relevance_file=args.relevance_file,
        quiet=args.quiet,
    )

    if args.show_figure and not args.no_figure and not detail_df.empty:
        try:
            html_file = os.path.join(OUTPUT_DIR, f"mitre_tactic_strengths_{meta['timestamp']}.html")
            if os.path.exists(html_file):
                import webbrowser
                webbrowser.open_new_tab(f"file://{html_file}")
        except Exception:
            pass
