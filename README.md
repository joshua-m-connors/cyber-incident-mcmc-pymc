# Cyber Risk Simulation Framework  
### *(FAIR + MITRE ATT&CK Integrated Quantitative Model)*

This framework integrates **Factor Analysis of Information Risk (FAIR)** principles with the **MITRE ATT&CK** framework to estimate cyber risk using **Bayesian inference**, **Monte Carlo simulation**, and **weighted control strength analytics**.  

It provides an end-to-end process to generate control strength data, calibrate it to organizational relevance, and quantify **annualized loss exposure (AAL)**, **incident frequency**, and **Single Loss Expectancy (SLE)**.

---

## 🧭 Overview

| Script | Purpose |
|--------|----------|
| `build_mitigation_influence_template.py` | Creates a baseline mitigation template (`mitigation_influence_template.csv`) derived from the MITRE ATT&CK dataset, seeding initial control strengths and influence weights. |
| `build_technique_relevance_template.py` | Generates `technique_relevance.csv`, which lets analysts mark relevant MITRE techniques (by Tactic) for specific **threat actors** or **campaigns** (e.g., APT29, C0017). |
| `mitre_control_strength_dashboard.py` | Aggregates mitigation-level strengths into tactic-level weighted ranges and produces an interactive Plotly dashboard (per MITRE tactic). Includes logic to gate “impact-reduction” mitigations (Backups, Encryption). |
| `cyber_incident_pymc.py` | Runs the **FAIR-MITRE hybrid risk model**, combining Bayesian inference (PyMC) with stochastic attacker simulations, adaptability logic, and FAIR loss distributions to estimate AAL, credible intervals, and exceedance curves. |

---

## ⚙️ Workflow Summary

### 1. Get the MITRE ATT&CK dataset  
Download and place the **Enterprise ATT&CK JSON** file into your working directory:

```bash
wget https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json
```

---

### 2. Build the mitigation influence template  
Generates a baseline of all MITRE mitigations, their coverage, and initial strength ranges:

```bash
python3 build_mitigation_influence_template.py
```

Output:
- `mitigation_influence_template.csv` — editable baseline of all mitigations
- Log file saved under `/output_YYYY-MM-DD/`

You can manually tune `Control_Min` and `Control_Max` in this CSV (e.g., based on internal maturity).

---

### 3. (Optional) Scope the model to relevant threat campaigns or actors  
Automatically marks techniques used by one or more specific **procedures** (e.g., APT29, FIN7) or **campaigns** (e.g., C0017):

```bash
python3 build_technique_relevance_template.py --procedure "APT29" --campaign C0017
```

Output:
- `/output_YYYY-MM-DD/technique_relevance.csv` — all tactics/techniques with “Relevant” column pre-marked  
- `/output_YYYY-MM-DD/technique_relevance_evidence.json` — audit evidence of what was auto-selected  

This file can later be manually refined to mark additional techniques.

---

### 4. Generate the control strength dashboard  
Aggregates mitigations → tactics and visualizes their weighted control strength ranges:

```bash
python3 mitre_control_strength_dashboard.py
```

Features:
- Weighted average of mitigation strengths within each MITRE tactic.  
- Discounts generic controls (e.g., “Audit”, “User Training”).  
- Caps hover list to top 20 mitigations (with “… and N more”).  
- Auto-detects and applies `technique_relevance.csv` if present.  
- Automatically zeroes out impact-reduction mitigations **not in scope**:
  - “Encrypt Sensitive Information” → only if *Exfiltration* tactic is in scope.
  - “Data Backup” → only if mapped to one or more *Impact* techniques.

Outputs:
- Interactive HTML dashboard: `/output_YYYY-MM-DD/mitre_tactic_strengths_*.html`
- Summary CSV: `/output_YYYY-MM-DD/filtered_summary_*.csv`
- (Used directly by `cyber_incident_pymc.py`)

---

### 5. Run the FAIR-MITRE Bayesian Risk Model

```bash
python3 cyber_incident_pymc.py --print-control-strengths
```

This performs:
- Bayesian inference of **attack frequency (λ)** using a lognormal prior.  
- **Per-tactic success probability** sampling from Beta distributions informed by MITRE control strengths.  
- **Monte Carlo simulation** of attacker progressions with retries, detection, and fallbacks.  
- Stochastic **adaptability and threat capability** per attacker chain.  
- FAIR-style loss modeling (lognormal bodies + bounded Pareto tails).  
- Optional stochastic sampling of impact-reduction mitigations (Backups, Encryption).  

Outputs:
- Summary and detailed CSVs under `/output_YYYY-MM-DD/`
- Diagnostic dashboard PNGs (2×2 view, log histogram, exceedance curve)
- Optional control-parameter CSV if `--print-control-strengths` used

---

## 🧩 Command-Line Options

### `cyber_incident_pymc.py`

| Flag | Description | Example |
|------|--------------|----------|
| `--dataset PATH` | MITRE ATT&CK STIX bundle path. | `--dataset enterprise-attack.json` |
| `--csv PATH` | Control strength file (defaults to `mitigation_control_strengths.csv`). | `--csv ./mitigation_control_strengths.csv` |
| `--print-control-strengths` | Prints and exports per-tactic control parameters used. | `--print-control-strengths` |
| `--no-adapt-stochastic` | Disables stochastic adaptability; uses fixed mean adaptability factor. | `--no-adapt-stochastic` |
| `--no-stochastic-impact` | Disables stochastic sampling of impact-reduction controls. | `--no-stochastic-impact` |
| `--no-plot` | Suppresses chart display (still saves images). | `--no-plot` |

---

### `mitre_control_strength_dashboard.py`

| Flag | Description | Example |
|------|--------------|----------|
| `--dataset PATH` | MITRE ATT&CK JSON dataset path. | `--dataset enterprise-attack.json` |
| `--strengths PATH` | Mitigation control strength CSV. | `--strengths ./mitigation_control_strengths.csv` |
| `--use-relevance` | Enables relevance filtering (auto-detects if file present). | `--use-relevance` |
| `--relevance-file PATH` | Specify alternate relevance CSV file. | `--relevance-file ./output_2025-10-31/technique_relevance.csv` |
| `--no-figure` | Run headless (no chart output). | `--no-figure` |
| `--show-figure` | Opens the generated HTML dashboard automatically. | `--show-figure` |

---

### `build_technique_relevance_template.py`

| Flag | Description | Example |
|------|--------------|----------|
| `--procedure NAME` | Include all techniques used by this threat actor, malware, or tool. | `--procedure "APT29"` |
| `--campaign Cxxxx` | Include all techniques used in this MITRE campaign. | `--campaign C0017` |
| `--mark-all all` | Mark all techniques as relevant. | `--mark-all all` |
| `--dedupe-names` | Deduplicate duplicate technique names within each tactic. | `--dedupe-names` |

---

## 📁 Output Structure

| File | Description |
|------|--------------|
| `/output_YYYY-MM-DD/mitre_tactic_strengths_*.html` | Interactive tactic-level control strength dashboard. |
| `/output_YYYY-MM-DD/filtered_summary_*.csv` | Weighted summary of control strengths by tactic. |
| `/output_YYYY-MM-DD/tactic_control_strengths_*.csv` | Diagnostic export of control parameters used in simulation. |
| `/output_YYYY-MM-DD/cyber_risk_simulation_results_*.csv` | Detailed per-draw results: λ, success chain, annual loss, incidents. |
| `/output_YYYY-MM-DD/cyber_risk_simulation_summary_*.csv` | Summary metrics including credible intervals for AAL, incident frequency, and SLE. |
| `/output_YYYY-MM-DD/dashboard_2x2_*.png` | Posterior distributions of λ, success prob., incident count, and annual loss. |
| `/output_YYYY-MM-DD/ale_log_chart_*.png` | Log-scale histogram of annual loss with percentile markers. |
| `/output_YYYY-MM-DD/loss_exceedance_curve_*.png` | Log-scale loss exceedance curve (P50, P90, P95, P99 markers). |

---

## 🧮 Model Highlights

- **Subset-aware modeling** – Only tactics and mitigations marked as relevant are modeled.  
- **Threat capability** – Randomized per attacker, controlling base success probability.  
- **Adaptability** – Logistic update per retry, simulating learning during repeated attempts.  
- **Detection and fallback** – Attack chains can reset to prior stages, realistically extending attack paths.  
- **Impact reduction controls** – “Data Backup” and “Encrypt Sensitive Information” dynamically reduce losses when in scope.  
- **FAIR-aligned losses** – Category-level lognormal distributions with Pareto tails for Legal and Reputation losses.  
- **Comprehensive summary output** – AAL, SLE, credible intervals, and validation of `AAL ≈ Frequency × SLE`.  
- **Visualization suite** – Auto-generated dashboard, ALE log histogram, and loss exceedance curve.

---

## 🧠 Recommended Practices

1. **Calibrate**: Review control strength ranges periodically using subject-matter input.  
2. **Scope Carefully**: Use `build_technique_relevance_template.py` to focus analysis on specific threat campaigns.  
3. **Validate Results**: Compare AAL and SLE to real or benchmark incident cost data.  
4. **Sensitivity Testing**: Run variations with different adaptability ranges or fallback probabilities.  
5. **Version Control**: Keep generated CSVs and evidence JSONs under source control for reproducibility.  

---

© 2025 — Quantitative Cyber Risk Analysis Framework (FAIR + MITRE ATT&CK)
