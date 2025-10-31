# Cyber Risk Simulation Framework (FAIR + MITRE ATT&CK Integrated)

This framework integrates **quantitative cyber risk modeling** using the **FAIR taxonomy** with the **MITRE ATT&CK** control framework.  
It combines Bayesian inference, Monte Carlo simulation, and weighted control strength analytics for comprehensive enterprise cyber risk estimation.

---

## 🧭 Overview

This repository contains three primary Python scripts that work together to produce an annualized loss distribution (AAL) and diagnostic dashboards.

| Script | Purpose |
|--------|----------|
| `build_mitigation_influence_template.py` | Generates the default mitigation template (`mitigation_influence_template.csv`) from the MITRE ATT&CK dataset. |
| `mitre_control_strength_dashboard.py` | Calculates per-tactic weighted control strength ranges using the MITRE ATT&CK mappings and user-updated mitigation strengths. |
| `cyber_incident_pymc.py` | Runs the Bayesian/Monte Carlo cyber loss model using FAIR-based loss categories and the aggregated MITRE control data. |
| `cyber_incident_pymc.ipynb` | This is a Jupyter Notebooks implemetation of a legacy version of the script. It does not utilize the control strengths captured in `mitigation_influence_template.csv` |

---

## Prerequisite

You will need the following software to run these scripts:
- Python3
- PyMC (a Python library for performing Markov Chain Monte Carlo simulation)
- Jupyter Notebooks (optional - only if you want to run the one Jupyter Notebooks `.ipynb` script)

---

## ⚙️ Workflow Summary

1. **Get the MITRE ATT&CK dataset**  
   Download the Enterprise ATT&CK JSON dataset into your project folder:  
   ```bash
   wget https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json
   ```
   or manually download from:  
   [https://github.com/mitre/cti/tree/master/enterprise-attack](https://github.com/mitre/cti/tree/master/enterprise-attack)

2. **Generate the mitigation influence template**  
   Run:
   ```bash
   python3 build_mitigation_influence_template.py
   ```
   This creates `mitigation_influence_template.csv` in the project’s base directory.

3. **Update mitigation strengths**  
   Open the generated CSV in Excel or another editor.  
   - Adjust **Control_Min** and **Control_Max** for each mitigation (in %).  
   - Save as: `mitigation_control_strengths.csv`

4. **Run the control strength analyzer**  
   ```bash
   python3 mitre_control_strength_dashboard.py
   ```
   This will:
   - Load your updated control strengths  
   - Weight mitigations by occurrence within each tactic  
   - Output an interactive Plotly dashboard  
   - Save all results to `/output_YYYY-MM-DD/`

5. **Run the risk simulation**  
   ```bash
   python3 cyber_incident_pymc.py
   ```
   The model will:
   - Load the per-tactic control strength ranges  
   - Simulate attacker success probabilities  
   - Apply Bayesian inference to event frequencies and FAIR-based losses  
   - Generate 2×2 charts, ALE distributions, and exceedance curves  
   - Export results to `/output_YYYY-MM-DD/`
   - If you have actual observational data (i.e. 3 successful incidents over the last 5 years) those can be entered using the variables `observed_total_incidents` and `observed_years`

   **New summary outputs include:**
   - Mean and 95% credible interval for successful incidents per year  
   - Mean and 95% credible interval for **loss per successful incident** (Single Loss Expectancy, or SLE)  
   - Derived check: **AAL ≈ Frequency × SLE**

---

## 🧩 Command-Line Flags

### `cyber_incident_pymc.py`

| Flag | Description | Example |
|------|--------------|----------|
| `--dataset PATH` | Path to MITRE ATT&CK dataset JSON (e.g., `enterprise-attack.json`). | `--dataset enterprise-attack.json` |
| `--csv PATH` | Optional override for `mitigation_control_strengths.csv`. | `--csv ./mitigation_control_strengths.csv` |
| `--print-control-strengths` | Print and export the per-tactic control strength parameters used. | `--print-control-strengths` |
| `--no-plot` | Skip all charts and visualizations (headless/CI use). | `--no-plot` |
| `--seed INT` | Set the random seed for reproducibility. | `--seed 42` |

### `mitre_control_strength_dashboard.py`

| Flag | Description | Example |
|------|--------------|----------|
| `--dataset PATH` | Path to the MITRE ATT&CK dataset JSON. | `--dataset enterprise-attack.json` |
| `--csv PATH` | Specify a control strength file to import. | `--csv ./mitigation_control_strengths.csv` |
| `--no-plot` | Skip the interactive Plotly chart generation. | `--no-plot` |

### `build_mitigation_influence_template.py`

| Flag | Description | Example |
|------|--------------|----------|
| `--dataset PATH` | Path to the MITRE ATT&CK dataset JSON file. | `--dataset enterprise-attack.json` |
| `--output PATH` | Optional override for CSV export path. | `--output ./templates/` |

---

## 📁 Output Structure

Each run automatically creates a date-labeled folder (e.g., `output_2025-10-30/`) containing:

| File | Description |
|------|--------------|
| `mitre_tactic_strengths_*.html` | Interactive Plotly dashboard of tactic-level control strengths. |
| `cyber_risk_simulation_results_*.csv` | Detailed per-run Monte Carlo sample results (λ, end-to-end success, annual loss, incidents). |
| `cyber_risk_simulation_summary_*.csv` | **Expanded summary statistics** including:<br>• Mean/Median AAL and 95% credible interval<br>• Mean incidents/year and 95% CI<br>• Mean loss per successful incident (SLE) and 95% CI<br>• Derived AAL check (Mean_Incidents × Mean_Loss_Per_Incident). |
| `dashboard_2x2_*.png` | Four-panel visualization of posterior λ, success probability, incident count, and annual loss. |
| `ale_log_chart_*.png` | Log-scaled annual loss histogram with percentile markers. |
| `loss_exceedance_curve_*.png` | Log-scale loss exceedance curve with annotated percentiles. |
| `tactic_control_strengths_*.csv` | Optional diagnostic export of per-tactic control parameters. |

---

## 🧮 Model Highlights

- Bayesian inference for **attack frequency (λ)** and **per-stage success** using **PyMC**.  
- Markov-style attacker simulation with retries, detection, and fallback logic.  
- FAIR taxonomy loss modeling with **lognormal bodies** and **bounded Pareto tails**.  
- Reports both **event frequency** and **Single Loss Expectancy (SLE)** — including mean and 95% credible intervals for each.  
- Provides **AAL decomposition validation:** AAL ≈ Frequency × SLE.  
- All summary metrics printed to console are also written to the daily summary CSV.  
- Outputs 2×2 dashboards, log-scale ALE distributions, and exceedance curves for intuitive interpretation.

---

## 🧩 File Renaming Guidance

If you rename any of the main scripts:
- Update any cross-script imports (`from mitre_control_strength_dashboard import ...` etc.).
- Ensure each script can still locate `enterprise-attack.json` and `mitigation_control_strengths.csv` in the same directory or via the `--dataset`/`--csv` flags.

---

## 🧠 Recommended Use

1. Update `mitigation_influence_template.csv` regularly when MITRE releases new dataset versions.  
2. Rerun `mitre_control_strength_dashboard.py` to update weighted control strengths.  
3. Use `--print-control-strengths` in the simulation script to confirm control weighting alignment.  
4. Review both frequency and severity credible intervals to ensure loss calibration remains realistic.  
5. Use the **AAL decomposition** and **summary CSV metrics** to validate risk quantification transparency.

---

© 2025 — Quantitative Cyber Risk Analysis Framework (FAIR + MITRE ATT&CK)
