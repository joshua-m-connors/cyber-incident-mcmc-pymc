# Cyber Risk Simulation Framework  
### *(FAIR + MITRE ATT&CK Integrated Quantitative Model)*

This program operationalizes **Factor Analysis of Information Risk (FAIR)** principles using the **MITRE ATT&CK** framework to produce quantitative cyber risk estimates.  

It leverages:
- MITRE ATT&CK data for **threat realism and control mapping**.  
- FAIR for **loss magnitude and frequency modeling**.  
- **Bayesian inference (PyMC)** and **Monte Carlo simulation** for uncertainty propagation.  
- A complete data pipeline from ATT&CK → control strengths → simulated loss distributions.  

For those interested in a deeper understanding of the model and to see what I have done to put it into action (think Agentic AI) see this [Wiki article](https://github.com/joshua-m-connors/cyber-incident-mcmc-pymc/wiki/FAIR-%E2%80%90-MITRE-ATT&CK-Model-Deep-Dive).

If you are interested in an implementation using R see this repository: https://github.com/joshua-m-connors/cyber-incident-mcmc-cmdstanr

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

## Build Mitigation Influence Template

Run:

```
python3 build_mitigation_influence_template.py
```

This produces:

- `mitigation_influence_template.csv`  
- A daily log file in `output_YYYY-MM-DD`

It contains default Control_Min and Control_Max values that can be manually calibrated.

### **Added: Dependent Controls and Control Groups**

The mitigation template now includes optional fields that allow analysts to model structural dependencies and shared health conditions across controls. These fields are part of `mitigation_control_strengths.csv` and are editable.

| Column | Description |
|--------|-------------|
| `Control_Type` | Indicates whether a mitigation is a native MITRE control (MITRE) or a supporting governance control (SUPPORT). |
| `Dependency_Group` | Logical grouping for controls that rely on a shared underlying platform or process, such as IAM foundation or network policy stack. |
| `Group_Health_Min` and `Group_Health_Max` | Analyst defined range in [0..1]. A single value is sampled for each group per run. |
| `Requires` | Space separated list of Mitigation_ID values that this control depends on. |
| `Requires_Influence_Min` and `Requires_Influence_Max` | Defines the strength of dependency influence. A value in the sampled range determines how upstream controls affect the final effective strength. |

These fields allow for realistic modeling of interdependent defensive controls within the analytical pipeline.

---

## Build Technique Relevance Template (Optional)

Scopes the model to specific campaigns or actors.

```
python3 build_technique_relevance_template.py --procedure "APT29" --campaign C0017
```

This produces:

- `technique_relevance.csv`
- Evidence JSON for auditability

---

## MITRE Control Strength Dashboard

```
python3 mitre_control_strength_dashboard.py
```

It computes weighted, tactic level control strengths, optionally filtered by the relevance template. It outputs:

- Interactive HTML dashboard  
- Filtered summary CSV  
- Impact reduction control adjustments

### **Added: Dependency and Control Group Health Engine**

Before aggregating mitigations into tactic level strengths, the dashboard evaluates all dependency and group health logic declared in `mitigation_control_strengths.csv`. This takes place in the `_compute_effective_mitigation_strengths()` function.

Key behaviors:

1. **Group Health Sampling**  
   For any set of controls sharing a `Dependency_Group`, a single health scalar is drawn from the analyst defined `Group_Health_Min` and `Group_Health_Max` range and applied to all controls in that group.

2. **Propagation of Control Dependencies**  
   Controls that have entries in the `Requires` column have their effective strength reduced if dependent controls are weak. An influence factor is sampled from `Requires_Influence_Min` and `Requires_Influence_Max`.  
   The formula applied is:  
   ```
   effective = baseline × ((1 - k) + k × dependency_strength)
   ```

3. **Cycle Detection**  
   Any cyclic dependency is detected and resolved by falling back to group adjusted baseline values.

4. **Result**  
   The dashboard outputs dependency aware, group adjusted effective strengths for all mitigations. These are then aggregated into per tactic minimum, maximum, and mean strengths used directly by the risk simulation.

---

## Run the FAIR MITRE Bayesian Risk Model

```
python3 cyber_incident_pymc.py --print-control-strengths
```

This produces:

- Posterior results for lambda, success probability, and annual losses  
- Annual loss histograms and exceedance curve  
- A diagnostic CSV of tactic level control strengths  

### **Added: Consumption of Dependency Adjusted Tactic Strengths**

The simulation now uses only the dependency adjusted and group adjusted tactic strengths produced by the dashboard. These values are already:

- Relevance filtered  
- Group health scaled  
- Dependency influenced  
- Discount adjusted  

The model converts each tactic's adjusted `(MinStrength, MaxStrength)` into a Beta distribution that governs attacker success probabilities at each stage. All Monte Carlo progression logic, detection, retries, and FAIR loss estimation rely on these effective strengths.

This makes the simulation structurally dependency aware throughout the entire analytical pipeline.

---

## Output Structure

| File | Description |
|------|--------------|
| `mitre_tactic_strengths_*.html` | Interactive control strength dashboard. |
| `filtered_summary_*.csv` | Weighted summaries of tactic strengths. |
| `tactic_control_strengths_*.csv` | Dependency adjusted tactic control values used in simulation. |
| `cyber_risk_simulation_results_*.csv` | Posterior results including lambda and losses. |
| `cyber_risk_simulation_summary_*.csv` | Summary metrics including credible intervals. |
| `dashboard_2x2_*.png` | Posterior distribution visualizations. |
| `ale_log_chart_*.png` | Log scale annual loss histogram. |
| `loss_exceedance_curve_*.png` | Loss exceedance curve. |

---

## Model Highlights

- Subset aware modeling using relevance filtering  
- Threat capability and adaptability modeling  
- Multi stage attack progression with detection and fallback  
- Impact reduction via backup and encryption mitigations  
- FAIR aligned category and tail loss modeling  
- Dependency aware simulation of control strength effects  
- Full reproducibility through daily output directories  

---

## Recommended Practices

1. Calibrate mitigation ranges regularly  
2. Use relevance templates to focus on specific actors  
3. Review dependency and group definitions to ensure realistic modeling  
4. Validate modeled AAL and SLE against internal data  
5. Version all CSV inputs to preserve analytic lineage  

---

FAIR–MITRE ATT&CK Quantitative Cyber Risk Framework

Copyright 2025 Joshua M. Connors

Licensed under the Apache License, Version 2.0.

This software incorporates public data from the MITRE ATT&CK® framework.
