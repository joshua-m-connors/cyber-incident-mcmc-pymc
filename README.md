# Cyber Risk Simulation Framework  
### *(FAIR + MITRE ATT&CK Integrated Quantitative Model)*

This program operationalizes **Factor Analysis of Information Risk (FAIR)** principles using the **MITRE ATT&CK** framework to produce quantitative cyber risk estimates.  

It leverages:
- MITRE ATT&CK data for **threat realism and control mapping**.  
- FAIR for **loss magnitude and frequency modeling**.  
- **Bayesian inference (PyMC)** and **Monte Carlo simulation** for uncertainty propagation.  
- A complete data pipeline from ATT&CK → control strengths → simulated loss distributions.  

For those interested in a deeper understanding of the model and to see what I have done to put it into action (think Agentic AI) see this [Wiki article](https://github.com/joshua-m-connors/cyber-incident-mcmc-pymc/wiki/FAIR-%E2%80%90-MITRE-ATT&CK-Model-Deep-Dive).

---

## 🧭 Framework Overview

| **Script** | **Function** |
|-------------|--------------|
| `build_mitigation_influence_template.py` | Builds a baseline **mitigation influence matrix**, quantifying which ATT&CK mitigations cover which techniques/tactics and generating a seed control-strength CSV. |
| `build_technique_relevance_template.py` | Creates a **tactic/technique relevance checklist** that can be pre-populated based on MITRE **procedures (e.g., APT29)** or **campaigns (e.g., C0017)** for focused threat modeling. |
| `mitre_control_strength_dashboard.py` | Aggregates mitigation-level control strengths into **tactic-level weighted averages**, applying relevance filters and producing an interactive **Plotly dashboard**. |
| `cyber_incident_pymc.py` | Executes the **Bayesian FAIR–MITRE simulation** using PyMC and Monte Carlo techniques, producing quantitative results such as **AAL**, **SLE**, and **loss exceedance curves**. |
| `cyber_incident_pymc.ipynb` | This is a legacy Jupyter Notebooks version of the main MCMC modeling script. Keeping it because it has some interesting visualizations breaking down the threat actor attack chain progression (e.g. retires, fallbacks, furthest point reached, etc.). |

---

## ⚙️ Full Workflow

### 1. Acquire MITRE ATT&CK Enterprise Dataset

Download the latest ATT&CK Enterprise bundle:

```bash
wget https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json
```

Ensure it resides in the same working directory as the scripts.

---

### 2. Generate the Mitigation Influence Template

Constructs a foundational view of all ATT&CK mitigations and their influence.

```bash
python3 build_mitigation_influence_template.py
```

Outputs:
- **`mitigation_influence_template.csv`** — baseline mitigation/control strength seed values.  
- **`/output_YYYY-MM-DD/mitigation_template_build_log_*.txt`** — run log.  

You can manually refine the `Control_Min` / `Control_Max` columns to reflect assessed control maturity.

---

### 3. Build Technique Relevance Template (Optional)

Scopes the model to specific **threat actors or campaigns**, marking relevant techniques automatically.

```bash
python3 build_technique_relevance_template.py --procedure "APT29" --campaign C0017
```

Outputs:
- `/output_YYYY-MM-DD/technique_relevance.csv` — Tactic/Technique checklist with “Relevant” marks.  
- `/output_YYYY-MM-DD/technique_relevance_evidence.json` — JSON evidence of auto-selections.  

You may open `technique_relevance.csv` to manually adjust relevance (mark additional `X`s).

---

### 4. Generate Control Strength Dashboard

Aggregates and visualizes the strength of mitigations per MITRE tactic.  
Optionally filters by relevance from the previous step.

```bash
python3 mitre_control_strength_dashboard.py
```

Features:
- Weighted mean/min/max of mitigations per tactic.  
- Discounts generic controls (Audit, User Training, etc.).  
- Caps hover lists (top 20 mitigations per tactic + “... and N more”).  
- Auto-detects and applies `technique_relevance.csv` if present.  
- Applies **impact mitigation gating logic**:
  - “Encrypt Sensitive Information” only applies if *Exfiltration* is in scope.  
  - “Data Backup” only applies if *Impact* includes backup-related techniques.  

Outputs:
- `/output_YYYY-MM-DD/mitre_tactic_strengths_*.html` — interactive dashboard.  
- `/output_YYYY-MM-DD/filtered_summary_*.csv` — per-tactic weighted control strengths.  
- These feed directly into the risk model script.

---

### 5. Run the FAIR–MITRE Bayesian Risk Model

Combines the control strengths, threat relevance, and FAIR-based loss modeling.

```bash
python3 cyber_incident_pymc.py --print-control-strengths
```

Core capabilities:
- Bayesian inference for **attack frequency (λ)**.  
- **Beta-distributed per-tactic success probabilities**, driven by control strengths.  
- Simulation of multi-stage attacker progressions (retries, detection, fallbacks).  
- Stochastic **adaptability** and **threat capability** per attacker.  
- FAIR-style losses: lognormal core + Pareto tails for Legal and Reputation.  
- Dynamic **impact reduction** for Backup and Encryption mitigations.  

Outputs:
- **CSV files** (results and summaries).  
- **2×2 dashboard**, **log-scale ALE histogram**, and **Loss Exceedance Curve (LEC)** plots.  
- Optional control-strength parameter CSV if `--print-control-strengths` is specified.

---

## 🧩 Command-Line Options

### `build_mitigation_influence_template.py`

| Option | Description |
|--------|--------------|
| *(none)* | Generates the baseline mitigation influence CSV and build log. |
| `--dataset PATH` | (Optional) Path to `enterprise-attack.json` (default: current directory). |

---

### `build_technique_relevance_template.py`

| Option | Description | Example |
|--------|--------------|----------|
| `--procedure NAME` | Auto-mark techniques used by an ATT&CK procedure. | `--procedure "APT29"` |
| `--campaign Cxxxx` | Auto-mark techniques used in a campaign. | `--campaign C0017` |
| `--mark-all all` | Mark all techniques as relevant. | `--mark-all all` |
| `--dedupe-names` | Remove duplicate technique names within a tactic. | `--dedupe-names` |

---

### `mitre_control_strength_dashboard.py`

| Option | Description | Example |
|--------|--------------|----------|
| `--dataset PATH` | MITRE ATT&CK JSON dataset. | `--dataset enterprise-attack.json` |
| `--strengths PATH` | Mitigation control strength CSV. | `--strengths mitigation_control_strengths.csv` |
| `--use-relevance` | Enables filtering by relevance CSV. | `--use-relevance` |
| `--relevance-file PATH` | Alternate relevance file path. | `--relevance-file ./output_2025-11-05/technique_relevance.csv` |
| `--no-figure` | Suppress chart generation. | `--no-figure` |
| `--show-figure` | Automatically open the generated HTML dashboard. | `--show-figure` |

---

### `cyber_incident_pymc.py`

| Option | Description | Example |
|--------|--------------|----------|
| `--dataset PATH` | MITRE ATT&CK JSON path. | `--dataset enterprise-attack.json` |
| `--csv PATH` | Control strength CSV. | `--csv mitigation_control_strengths.csv` |
| `--no-adapt-stochastic` | Disables stochastic adaptability (fixed learning). | `--no-adapt-stochastic` |
| `--no-stochastic-impact` | Use mean values for impact reduction controls. | `--no-stochastic-impact` |
| `--print-control-strengths` | Prints/exports per-tactic control parameter table. | `--print-control-strengths` |
| `--no-plot` | Headless mode (saves figures only). | `--no-plot` |

---

## 📁 Output File Summary

| File | Description |
|------|--------------|
| **`mitigation_influence_template.csv`** | Seed file listing each ATT&CK mitigation, its coverage, and default control strength range. |
| **`technique_relevance.csv`** | Tactic–technique matrix allowing marking of relevant items per campaign or procedure. |
| **`filtered_summary_*.csv`** | Weighted tactic-level control strength summary (from dashboard). |
| **`mitre_tactic_strengths_*.html`** | Interactive HTML visualization of control strengths. |
| **`tactic_control_strengths_*.csv`** | Diagnostic export of per-tactic control parameters used in simulation. |
| **`cyber_risk_simulation_results_*.csv`** | Detailed posterior results (λ, success probability, annual losses, etc.). |
| **`cyber_risk_simulation_summary_*.csv`** | Summary of AAL, credible intervals, incident frequency, and SLE. |
| **`dashboard_2x2_*.png`** | Posterior distributions (λ, success probability, incidents, losses). |
| **`ale_log_chart_*.png`** | Log-scale Annual Loss histogram with percentile markers. |
| **`loss_exceedance_curve_*.png`** | Log-scale loss exceedance curve (P50, P90, P95, P99). |

---

## 🧮 Model Highlights

- **Subset-aware simulation:** only includes tactics/techniques marked as relevant.  
- **Threat capability:** randomizes per-attacker success scaling.  
- **Adaptability:** logistic per-retry learning curve for adaptive attackers.  
- **Detection/fallback:** simulates realistic defensive re-engagement.  
- **Impact mitigation:** Backup and Encryption dynamically reduce modeled losses.  
- **FAIR-aligned losses:** multi-category lognormal + Pareto-tails for extreme events.  
- **Outputs credible intervals** for AAL, incidents/year, and loss-per-incident.  
- **Validates AAL ≈ Frequency × SLE.**

---

## 🧠 Recommended Practices

1. **Calibrate control strengths:** refine `mitigation_influence_template.csv` with SME input.  
2. **Scope the analysis:** use the relevance CSV to align with specific campaigns or actors.  
3. **Run sensitivity analysis:** vary adaptability, fallback, and detection parameters.  
4. **Compare benchmarks:** validate AAL/SLE outputs against internal loss data or peer estimates.  
5. **Version outputs:** retain daily `/output_YYYY-MM-DD/` folders for reproducibility and audit trail.  

---

© 2025 — FAIR–MITRE ATT&CK Quantitative Cyber Risk Framework  
Author: *Joshua M. Connors*  
License: *Open Use for Non-Commercial Research and Risk Analysis*  
