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
# FAIR–MITRE Bayesian Cyber Incident Model
# ---------------------------------------------------------------------------
# Quantifies cyber risk using Bayesian inference (PyMC) and FAIR-style loss
# distributions, informed by MITRE ATT&CK control strengths and relevance.
# ---------------------------------------------------------------------------
"""
=====================================================================================
Cyber Incident Risk Model with MITRE ATT&CK + FAIR (Subset-Aware, Impact-Reduced)
=====================================================================================
PURPOSE
-------------------------------------------------------------------------------------
This script quantifies cyber risk by combining:
  • MITRE ATT&CK tactic-level defensive control strengths (from the dashboard)
  • A Bayesian model (PyMC) for attack attempt frequency and per-stage success
  • FAIR-style per-incident losses (with heavy-tailed legal/reputation components)

ENHANCEMENTS IN THIS VERSION
-------------------------------------------------------------------------------------
1) Subset-aware modeling:
   - If available, calls mitre_control_strength_dashboard.get_mitre_tactic_strengths()
     to obtain an ordered subset of tactics and aggregated control strengths.
   - The model builds priors and simulates progression ONLY across those tactics.

2) Dashboard fallback:
   - If the dashboard is unavailable or fails, the script reverts to the legacy
     12-tactic SME map embedded in this file.

3) Impact-side reductions:
   - Incorporates two special mitigations provided by the dashboard:
       "Data Backup" (reduces Productivity & ResponseContainment losses)
       "Encrypt Sensitive Information" (reduces RegulatoryLegal & ReputationCompetitive)
   - Default behavior samples their strengths per posterior draw from their [min,max] range.
   - Toggleable via top-level variable and CLI flag.

CODE STYLE
-------------------------------------------------------------------------------------
- Highly commented for maintainability and clarity.
- No breaking changes to outputs except being subset-aware automatically.
=====================================================================================
"""

import os
import sys
import math
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------- Optional dependency (PyMC) ----------
try:
    import pymc as pm
    HAVE_PYMC = True
except Exception:
    HAVE_PYMC = False

# ---------- Dashboard integration ----------
try:
    from mitre_control_strength_dashboard import get_mitre_tactic_strengths
    HAVE_MITRE_ANALYZER = True
except Exception as e:
    print(f"⚠️ MITRE analyzer not available: {e}")
    HAVE_MITRE_ANALYZER = False

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

# =============================================================================
# Variables: GLOBAL CONFIG — priors, runtime, plotting
# =============================================================================
# Frequency prior (attempts/year), elicited via 90% CI → lognormal
# These are parameters that are similar to TEF in FAIR
# (These are fairly broad defaults; adjust as needed.)
CI_MIN_FREQ = 4
CI_MAX_FREQ = 24
Z_90 = 1.645

# =============================================================================
# Variables: Threat capability (higher = stronger attacker)
# =============================================================================
# Threat capability modifies per-stage success probabilities during simulation.
THREAT_CAPABILITY_STOCHASTIC = True
THREAT_CAPABILITY_RANGE = (0.4, 0.90)   # higher = more capable attacker

# =============================================================================
# Variables: PyMC sampling controls
# =============================================================================
# These control the Markov Chain Monte Carlo (MCMC) sampling process.
N_SAMPLES = 4000
N_TUNE = 1000
N_CHAINS = 4
TARGET_ACCEPT = 0.90
RANDOM_SEED = 42

# Posterior predictive Monte Carlo (per posterior draw)
N_SIM_PER_DRAW = 1000  # number of attack attempts simulated per posterior draw

# =============================================================================
# Variables: Attacker progression controls
# =============================================================================
# These control the per-attempt simulation of stagewise progression.
MAX_RETRIES_PER_STAGE = 3
RETRY_PENALTY = 0.90
FALLBACK_PROB = 0.25
DETECT_BASE = 0.01
DETECT_INC_MIN = 0.01
DETECT_INC_MAX = 0.05
MAX_FALLBACKS_PER_CHAIN = 3

# Visualization Option: plot monetary values in millions
PLOT_IN_MILLIONS = True

# =============================================================================
# Variables: Adaptability (stochastic per retry) — logistic update mode
# =============================================================================
# Adaptability controls how quickly an attacker learns from failed attempts.
ADAPTABILITY_STOCHASTIC = True
ADAPTABILITY_RANGE = (0.3, 0.7)        # higher = faster learning on retries
ADAPTABILITY_MODE = "logistic"         # "logistic" (recommended) or "linear" (legacy)
ADAPTABILITY_EFFECT_SCALE = 1.0        # multiplier for linear mode; 1.0 = default

# =============================================================================
# Variables: Optional observed data for Poisson conditioning (keep None by default)
# =============================================================================
# If provided, these condition the posterior on observed incident counts.
observed_total_incidents = None
observed_years = None

# =============================================================================
# MITRE STAGES (canonical fallback)
# =============================================================================
MITRE_STAGES = [
    "Initial Access","Execution","Persistence","Privilege Escalation","Defense Evasion",
    "Credential Access","Discovery","Lateral Movement","Collection","Command And Control",
    "Exfiltration","Impact",
]

# =============================================================================
# Variables: SME fallback (tactic → control block range in [0..0.95])
# =============================================================================
_SME_STAGE_CONTROL_MAP_FALLBACK = {
    "Initial Access": (0.20, 0.50),
    "Execution": (0.20, 0.50),
    "Persistence": (0.20, 0.55),
    "Privilege Escalation": (0.25, 0.55),
    "Defense Evasion": (0.25, 0.55),
    "Credential Access": (0.20, 0.50),
    "Discovery": (0.20, 0.55),
    "Lateral Movement": (0.20, 0.50),
    "Collection": (0.20, 0.50),
    "Command And Control": (0.20, 0.55),
    "Exfiltration": (0.20, 0.50),
    "Impact": (0.20, 0.50),
}

# =============================================================================
# Variables: FAIR TAXONOMY — per-incident losses (lognormal bodies + Pareto tails)
# =============================================================================
loss_categories = ["Productivity", "ResponseContainment", "RegulatoryLegal", "ReputationCompetitive"]

# Lognormal parameters from 5th and 95th percentiles (per category)
loss_q5_q95 = {
    "Productivity": (1_000, 200_000),
    "ResponseContainment": (10_000, 1_000_000),
    "RegulatoryLegal": (0, 3_000_000),
    "ReputationCompetitive": (0, 5_000_000),
}

Z_90 = 1.645  # reused

def _lognormal_from_q5_q95(q5: float, q95: float):
    q5, q95 = max(q5, 1.0), max(q95, q5 * 1.0001)
    ln5, ln95 = np.log(q5), np.log(q95)
    sigma = (ln95 - ln5) / (2.0 * Z_90)
    mu = 0.5 * (ln5 + ln95)
    return mu, sigma

cat_mu = np.zeros(len(loss_categories))
cat_sigma = np.zeros(len(loss_categories))
for i, cat in enumerate(loss_categories):
    mu, sg = _lognormal_from_q5_q95(*loss_q5_q95[cat])
    cat_mu[i], cat_sigma[i] = mu, sg

# =============================================================================
# Variables: Pareto tails for legal & reputation categories
# =============================================================================
pareto_defaults = {
    "RegulatoryLegal":       {"xm": 50_000.0, "alpha": 3.5},
    "ReputationCompetitive": {"xm": 100_000.0, "alpha": 2.75},
}

# =============================================================================
# Variables: Impact reduction (from dashboard) — toggle + multipliers
# =============================================================================
# The dashboard returns strengths in PERCENT (0–100) for:
#   "Data Backup" and "Encrypt Sensitive Information"
# We convert to [0..1] and scale category losses accordingly.
BACKUP_IMPACT_MULT  = 0.60   # scales Productivity & ResponseContainment
ENCRYPT_IMPACT_MULT = 0.50   # scales RegulatoryLegal & ReputationCompetitive

# Toggle: when True (default) sample per posterior draw from [min,max];
# when False, use mean strength (deterministic).
STOCHASTIC_IMPACT_REDUCTION = True

# =============================================================================
# Output directory (daily)
# =============================================================================
def make_output_dir(prefix="output"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(base_dir, f"{prefix}_{date_str}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Output directory: {out_dir}")
    return out_dir

OUTPUT_DIR = make_output_dir()
LAST_CATEGORY_LOSSES = None  # populated after simulation

# =============================================================================
# Helpers: dashboard integration, priors, formatting, etc.
# =============================================================================
def _load_from_dashboard_or_fallback(dataset_path: str, csv_path: str):
    """
    Returns a 4-tuple:
      (tactics_included: List[str],
       stage_control_map: Dict[str, (float lo, float hi)],  # control block fractions [0..0.95]
       impact_reduction_controls: Dict[str, Dict[str,float]],  # Data Backup / Encrypt Sensitive Information
       mode_str: str)  # "Filtered" | "Full" | "Fallback"
    """
    if not HAVE_MITRE_ANALYZER:
        print("⚠️ MITRE dashboard unavailable — reverting to internal SME fallback control ranges.")
        return MITRE_STAGES.copy(), _SME_STAGE_CONTROL_MAP_FALLBACK.copy(), {}, "Fallback"

    try:
        detail_df, summary_df, control_strength_map, relevance_metadata, impact_controls = get_mitre_tactic_strengths(
            dataset_path=dataset_path,
            csv_path=csv_path,
            seed=42,
            build_figure=False,
            use_relevance=True,                # dashboard decides based on file presence
            relevance_file="technique_relevance.csv",
            quiet=True,
        )

        if not summary_df.empty and control_strength_map:
            tactics_included = list(summary_df["Tactic"])
            stage_map = {}
            for t in tactics_included:
                row = control_strength_map.get(t, {})
                lo = float(row.get("min_strength", 30.0)) / 100.0
                hi = float(row.get("max_strength", 70.0)) / 100.0
                lo = max(0.0, min(0.95, lo))
                hi = max(0.0, min(0.95, hi))
                if lo > hi:
                    lo, hi = hi, lo
                stage_map[t] = (lo, hi)

            mode = relevance_metadata.get("mode", "Full")
            print(f"✅ Loaded control strengths from MITRE ATT&CK dataset ({mode} mode).")
            print(f"🧩 Included tactics ({len(tactics_included)}): {', '.join(tactics_included)}")

            # Light log of impact-control means (percent values)
            if impact_controls:
                for k, v in impact_controls.items():
                    ms = float(v.get("mean_strength", 0.0))
                    print(f"   • Impact reduction available: {k} — mean {ms:.1f}%")

            return tactics_included, stage_map, impact_controls, mode

        print("⚠️ Dashboard returned no tactic summary — using SME fallback map.")
        return MITRE_STAGES.copy(), _SME_STAGE_CONTROL_MAP_FALLBACK.copy(), {}, "Fallback"

    except Exception as e:
        print(f"⚠️ MITRE dataset load failed in dashboard: {e}. Using SME fallback.")
        return MITRE_STAGES.copy(), _SME_STAGE_CONTROL_MAP_FALLBACK.copy(), {}, "Fallback"

def _print_stage_control_map(stage_map, tactics_included):
    """Diagnostic print + CSV export of tactic control strength ranges (subset-aware)."""
    print("\n--- Tactic Control Strength Parameters Used ---")
    print(f"{'Tactic':<25} {'MinStrength':>12} {'MaxStrength':>12}")
    for t in tactics_included:
        lo, hi = stage_map.get(t, (0.0, 0.0))
        print(f"{t:<25} {lo*100:>11.1f}% {hi*100:>11.1f}%")
    print("------------------------------------------------")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = os.path.join(OUTPUT_DIR, f"tactic_control_strengths_{ts}.csv")
    pd.DataFrame([
        {"Tactic": t, "MinStrength": stage_map[t][0] * 100, "MaxStrength": stage_map[t][1] * 100}
        for t in tactics_included
    ]).to_csv(csv_path, index=False)
    print(f"✅ Saved control strength parameters → {csv_path}")

def _success_interval_from_control(block_lo: float, block_hi: float):
    """
    Convert control block interval (fraction) → attacker success interval.
    success = 1 - block
    """
    block_lo = max(0.0, min(0.95, block_lo))
    block_hi = max(0.0, min(0.95, block_hi))
    lo_succ = 1.0 - block_hi
    hi_succ = 1.0 - block_lo
    if lo_succ > hi_succ:
        lo_succ, hi_succ = hi_succ, lo_succ
    return lo_succ, hi_succ

def _beta_from_interval(lo: float, hi: float, strength: float = 200.0):
    mu = 0.5 * (lo + hi)
    k = max(2.0, float(strength))
    a = max(1e-3, mu * k)
    b = max(1e-3, (1 - mu) * k)
    return a, b

def _fmt_money(x: float, millions: bool = None) -> str:
    if millions is None:
        millions = PLOT_IN_MILLIONS
    if millions:
        return f"${x/1_000_000:,.2f}M"
    return f"${x:,.0f}"

# =============================================================================
# Posterior & simulation core (subset-aware)
# =============================================================================
def _build_beta_priors_from_stage_map(stage_map, tactics_included):
    """Return Beta(a,b) parameters per included tactic; prints preview capability."""
    rng = np.random.default_rng(RANDOM_SEED)

    # ---- FAIR-aligned preview ------------------------------------------------
    # Legacy adaptation preview removed (ADAPTATION_* no longer used).
    # We now preview Threat Capability, which is applied later during simulation.
    try:
        cap_stoch = THREAT_CAPABILITY_STOCHASTIC
        cap_range = THREAT_CAPABILITY_RANGE
    except NameError:
        # Safe defaults if globals were renamed elsewhere
        cap_stoch = True
        cap_range = (0.5, 1.0)
    print(f"Using threat capability range {cap_range} (stochastic={cap_stoch})")

    alphas, betas = [], []
    for t in tactics_included:
        blo, bhi = stage_map[t]
        # Keep resistance (control strengths) exactly as provided by MITRE;
        # threat capability is applied later when simulating attempts.
        slo, shi = _success_interval_from_control(blo, bhi)
        a, b = _beta_from_interval(slo, shi, strength=50.0)
        alphas.append(a); betas.append(b)
    return np.array(alphas), np.array(betas)

def _simulate_attacker_path(success_probs, rng):
    """
    Simulate stage-by-stage attacker progression with retries, detection checks,
    adaptability adjustments, and fallbacks.

    This version is identical to the original except that the detection
    probability increase per retry is no longer static. Instead it is sampled
    from DETECT_INC_MIN to DETECT_INC_MAX once per retry attempt.

    Parameters
    ----------
    success_probs : list of float
        Probability of attacker success for each stage. These values must already
        include any control-strength adjustments or dependency effects.
    rng : numpy.random.Generator
        Random generator supplied by the calling simulation.

    Returns
    -------
    bool
        True if the attacker completes the final stage.
        False if the attacker stalls, is detected, or exhausts retries.
    """

    i = 0
    n_stages = len(success_probs)
    fallback_count = 0

    # No stages means automatic failure.
    if n_stages == 0:
        return False

    while 0 <= i < n_stages:
        p_current = float(success_probs[i])
        detect_prob = DETECT_BASE

        # Attempt retries for this stage
        for _ in range(MAX_RETRIES_PER_STAGE):

            # Try stage
            if rng.random() < p_current:
                # Success, move to next stage
                i += 1
                break

            # Failure: increase detection probability by a sampled increment
            delta = float(rng.uniform(DETECT_INC_MIN, DETECT_INC_MAX))
            detect_prob = min(1.0, detect_prob + delta)

            # Check if detected
            if rng.random() < detect_prob:
                return False

            # Adaptability adjustment (unchanged)
            if ADAPTABILITY_STOCHASTIC:
                adapt_factor = float(rng.uniform(*ADAPTABILITY_RANGE))
            else:
                adapt_factor = float(sum(ADAPTABILITY_RANGE) / 2.0)

            if ADAPTABILITY_MODE == "logistic":
                change = adapt_factor * (1.0 - p_current) * p_current
            else:
                change = (adapt_factor * ADAPTABILITY_EFFECT_SCALE) * (1.0 - p_current)

            p_current = max(0.0, min(1.0, p_current + change))

        else:
            # Retries exhausted: fallback attempt
            if (
                rng.random() < FALLBACK_PROB and
                fallback_count < MAX_FALLBACKS_PER_CHAIN
            ):
                fallback_count += 1
                i = max(0, i - 1)  # step back one stage
                continue
            else:
                return False

    # Success if we exit past the final stage
    return i >= n_stages

def _sample_posterior_lambda_and_success(alphas: np.ndarray, betas: np.ndarray, n_stages: int):
    """
    Build and sample PyMC model for λ and per-stage success (subset-aware).
    Returns (lambda_draws, success_chain_draws, succ_mat_or_None).
    """
    if not HAVE_PYMC:
        # Prior-only fallback (keeps script runnable without PyMC)
        rng = np.random.default_rng(RANDOM_SEED)
        mu_l = np.log(np.sqrt(CI_MIN_FREQ * CI_MAX_FREQ))
        sig_l = (np.log(CI_MAX_FREQ) - np.log(CI_MIN_FREQ)) / (2.0 * Z_90)
        lam = rng.lognormal(mean=mu_l, sigma=sig_l, size=N_SAMPLES)
        succ_mat = rng.beta(alphas, betas, size=(N_SAMPLES, n_stages))
        succ_chain = np.prod(succ_mat, axis=1)
        return lam, succ_chain, None

    with pm.Model() as model:
        # Lognormal prior for attempt frequency (λ)
        mu_lambda = np.log(np.sqrt(CI_MIN_FREQ * CI_MAX_FREQ))
        sigma_lambda = (np.log(CI_MAX_FREQ) - np.log(CI_MIN_FREQ)) / (2.0 * Z_90)
        lambda_rate = pm.Lognormal("lambda_rate", mu=mu_lambda, sigma=sigma_lambda)

        # Per-stage success probabilities (Beta priors) over the included tactics
        success_probs = pm.Beta("success_probs", alpha=alphas, beta=betas, shape=n_stages)

        if (observed_total_incidents is not None) and (observed_years is not None):
            pm.Poisson("obs_incidents", mu=lambda_rate * observed_years, observed=observed_total_incidents)
            print(f"Conditioning on observed data: {observed_total_incidents} incidents over {observed_years} years.")
        else:
            print("No observed incident data provided — running fully prior-driven.")

        trace = pm.sample(
            draws=N_SAMPLES, tune=N_TUNE, chains=N_CHAINS,
            target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED, progressbar=True
        )

    lambda_draws = np.asarray(trace.posterior["lambda_rate"]).reshape(-1)
    succ_mat = np.asarray(trace.posterior["success_probs"]).reshape(-1, n_stages)
    succ_chain_draws = np.prod(succ_mat, axis=1)
    return lambda_draws, succ_chain_draws, succ_mat

def _simulate_annual_losses(lambda_draws, succ_chain_draws, succ_mat,
                            alphas, betas,
                            tactics_included,
                            impact_reduction_controls=None,
                            severity_median=500_000.0,
                            severity_gsd=2.0,
                            rng_seed=1234):
    """
    Posterior predictive per-draw Monte Carlo:
      - Draw attempts ~ Poisson(λ)
      - For each attempt, simulate stage progression over *tactics_included*
      - On success, draw per-category losses; apply impact reductions if provided
    Returns:
      losses (N,), successes (N,)
    Also populates LAST_CATEGORY_LOSSES with per-category annual totals.
    """
    global LAST_CATEGORY_LOSSES

    rng = np.random.default_rng(rng_seed)
    mu = math.log(max(1e-9, severity_median))
    sigma = math.log(max(1.000001, severity_gsd))

    n = len(lambda_draws)
    n_stages = len(tactics_included)
    losses = np.zeros(n, dtype=float)
    successes = np.zeros(n, dtype=int)

    prod_losses = np.zeros(n, dtype=float)
    resp_losses = np.zeros(n, dtype=float)
    reg_losses  = np.zeros(n, dtype=float)
    rep_losses  = np.zeros(n, dtype=float)

    # Pre-extract ranges (converted to [0..1]) for the two impact controls
    backup_lo = backup_hi = encrypt_lo = encrypt_hi = 0.0
    backup_mean = encrypt_mean = 0.0
    if impact_reduction_controls:
        backup_lo   = float(impact_reduction_controls.get("Data Backup", {}).get("min_strength", 0.0)) / 100.0
        backup_hi   = float(impact_reduction_controls.get("Data Backup", {}).get("max_strength", 0.0)) / 100.0
        backup_mean = float(impact_reduction_controls.get("Data Backup", {}).get("mean_strength", 0.0)) / 100.0
        encrypt_lo   = float(impact_reduction_controls.get("Encrypt Sensitive Information", {}).get("min_strength", 0.0)) / 100.0
        encrypt_hi   = float(impact_reduction_controls.get("Encrypt Sensitive Information", {}).get("max_strength", 0.0)) / 100.0
        encrypt_mean = float(impact_reduction_controls.get("Encrypt Sensitive Information", {}).get("mean_strength", 0.0)) / 100.0

    for idx, (lam, p_succ) in enumerate(zip(lambda_draws, succ_chain_draws)):
        attempts = rng.poisson(lam=lam)
        succ_count = 0
        prod_acc = resp_acc = reg_acc = rep_acc = 0.0
        total_loss = 0.0
        # Draw baseline Threat Capability for this posterior draw (higher = stronger attacker)
        if THREAT_CAPABILITY_STOCHASTIC:
            tc = float(rng.uniform(*THREAT_CAPABILITY_RANGE))
        else:
            tc = float(np.mean(THREAT_CAPABILITY_RANGE))

        # --- Sample or fix impact reductions ONCE per posterior draw ---
        if STOCHASTIC_IMPACT_REDUCTION:
            backup_s  = rng.uniform(backup_lo, backup_hi) if backup_hi > backup_lo else backup_lo
            encrypt_s = rng.uniform(encrypt_lo, encrypt_hi) if encrypt_hi > encrypt_lo else encrypt_lo
        else:
            backup_s, encrypt_s = backup_mean, encrypt_mean

        for _ in range(attempts):
        # Use posterior stage success if available; else draw from priors
            if succ_mat is not None:
                stage_success_probs = succ_mat[idx].astype(float)
            else:
                stage_success_probs = rng.beta(alphas, betas).astype(float)

        # Apply baseline Threat Capability to stage success probabilities (higher tc => higher chance to succeed)
        stage_success_probs = np.clip(stage_success_probs + tc * (1.0 - stage_success_probs), 0.0, 1.0)

        # Simulate progression
        if _simulate_attacker_path(stage_success_probs, rng):

                # Draw per-category losses (bounded lognormal + bounded Pareto tails)
                prod = resp = reg = rep = 0.0
                for j, cat in enumerate(loss_categories):
                    mu_j, sigma_j = cat_mu[j], cat_sigma[j]
                    # lognormal body (cap ~99.9th for numerical stability)
                    base_draw = float(rng.lognormal(mean=mu_j, sigma=sigma_j))
                    lognorm_cap = math.exp(mu_j + 3.09 * sigma_j)
                    base_draw = min(base_draw, lognorm_cap)

                    if cat == "RegulatoryLegal":
                        reg = base_draw
                        if rng.random() < 0.025:
                            xm, alpha = pareto_defaults[cat]["xm"], pareto_defaults[cat]["alpha"]
                            u = rng.uniform(0.001, 0.999)
                            tail_draw = xm * (1.0 - u) ** (-1.0 / alpha)
                            tail_cap  = xm * (0.95) ** (-1.0 / alpha)
                            reg = max(reg, min(tail_draw, tail_cap))

                    elif cat == "ReputationCompetitive":
                        rep = base_draw
                        if rng.random() < 0.015:
                            xm, alpha = pareto_defaults[cat]["xm"], pareto_defaults[cat]["alpha"]
                            u = rng.uniform(0.001, 0.999)
                            tail_draw = xm * (1.0 - u) ** (-1.0 / alpha)
                            tail_cap  = xm * (0.95) ** (-1.0 / alpha)
                            rep = max(rep, min(tail_draw, tail_cap))

                    elif cat == "Productivity":
                        prod = base_draw

                    elif cat == "ResponseContainment":
                        resp = base_draw

                # Apply impact reductions
                if backup_s > 0.0:
                    scale = max(0.0, 1.0 - backup_s * BACKUP_IMPACT_MULT)
                    prod *= scale
                    resp *= scale
                if encrypt_s > 0.0:
                    scale = max(0.0, 1.0 - encrypt_s * ENCRYPT_IMPACT_MULT)
                    reg  *= scale
                    rep  *= scale

                # accumulate
                prod_acc += prod
                resp_acc += resp
                reg_acc  += reg
                rep_acc  += rep
                total_loss += (prod + resp + reg + rep)
                succ_count += 1

        losses[idx] = total_loss
        successes[idx] = succ_count
        prod_losses[idx] = prod_acc
        resp_losses[idx] = resp_acc
        reg_losses[idx]  = reg_acc
        rep_losses[idx]  = rep_acc

    LAST_CATEGORY_LOSSES = {
        "Productivity": prod_losses,
        "ResponseContainment": resp_losses,
        "RegulatoryLegal": reg_losses,
        "ReputationCompetitive": rep_losses
    }
    return losses, successes

# =============================================================================
# Console output, viz, exports
# =============================================================================
def _print_aal_summary(losses: np.ndarray, successes: np.ndarray):
    aal_mean   = float(np.mean(losses))
    aal_median = float(np.median(losses))
    lo, hi     = np.quantile(losses, [0.025, 0.975])
    mean_succ  = float(np.mean(successes))
    succ_lo, succ_hi = np.quantile(successes, [0.025, 0.975])
    pct_zero   = float(np.mean(successes == 0) * 100.0)

    print("\nAAL posterior predictive summary (with severity & tails):")
    print(f"Mean AAL: {_fmt_money(aal_mean)}")
    print(f"Median AAL: {_fmt_money(aal_median)}")
    print(f"AAL 95% credible interval (annualized total loss): {_fmt_money(lo)} – {_fmt_money(hi)}")
    print(f"Mean successful incidents / year: {mean_succ:.2f}")
    print(f"95% credible interval (incidents / year): {succ_lo:.2f} – {succ_hi:.2f}")

    # Mean loss per successful incident (SLE)
    valid = successes > 0
    if np.any(valid):
        per_event_losses = np.divide(losses[valid], successes[valid],
                                     out=np.zeros_like(losses[valid]),
                                     where=successes[valid] > 0)
        mean_loss_per_event = float(np.mean(per_event_losses))
        lo_event, hi_event  = np.quantile(per_event_losses, [0.025, 0.975])
        print(f"Mean loss per successful incident: {_fmt_money(mean_loss_per_event)}")
        print(f"95% credible interval (loss / incident): {_fmt_money(lo_event)} – {_fmt_money(hi_event)}")
    else:
        print("Mean loss per successful incident: (no successful incidents in simulation)")

    print(f"% years with zero successful incidents: {pct_zero:.1f}%")

    # Category breakdown
    print("\nCategory-level annual loss 95% credible intervals:")
    if LAST_CATEGORY_LOSSES is not None:
        for c in loss_categories:
            arr = LAST_CATEGORY_LOSSES.get(c, np.zeros_like(losses))
            lw, up = np.quantile(arr, [0.025, 0.975])
            med    = float(np.median(arr))
            pct_of_med = (med / aal_median) * 100.0 if aal_median > 0 else 0.0
            print(f"  {c:<24} {_fmt_money(lw)} – {_fmt_money(up)} "
                  f"(median {_fmt_money(med)}, ~{pct_of_med:.1f}% of median AAL)")
    else:
        print("  (Per-category breakdown unavailable.)")

def _annotate_percentiles(ax, samples, money=False):
    pcts = [50, 90, 95, 99]
    vals = np.percentile(samples, pcts)
    ymin, ymax = ax.get_ylim()
    ytext = ymax * 0.95
    for i, (p, v) in enumerate(zip(pcts, vals)):
        ax.axvline(v, linestyle="--", linewidth=1.0)
        label = _fmt_money(v) if money else (f"{v:.3f}" if v < 10 else f"{v:,.2f}")
        y_offset = (i % 2) * 0.05 * (ymax - ymin)
        ax.text(v, ytext - y_offset, f"P{p}={label}",
                rotation=0, va="bottom", ha="center", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))

def _render_2x2_and_log_ale(losses: np.ndarray,
                            lambda_draws: np.ndarray,
                            success_chain_draws: np.ndarray,
                            show: bool = True):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    succ_per_year = lambda_draws * success_chain_draws

    def _auto_clip(data, low=0.001, high=0.991):
        if len(data) == 0:
            return data
        low_v, high_v = np.percentile(data, [low * 100, high * 100])
        return data[(data >= low_v) & (data <= high_v)]

    lambda_plot = _auto_clip(lambda_draws)
    succ_chain_plot = _auto_clip(success_chain_draws)
    succ_per_year_plot = _auto_clip(succ_per_year)
    losses_plot = _auto_clip(losses)

    def _millions(x, pos): return f"${x/1e6:,.1f}M"

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax = axs[0,0]
    ax.hist(lambda_plot, bins=60, edgecolor="black")
    ax.set_title("Posterior λ (incidents/year)")
    ax.set_xlabel("λ"); ax.set_ylabel("Count")
    _annotate_percentiles(ax, lambda_plot, money=False)

    ax = axs[0,1]
    ax.hist(succ_chain_plot, bins=60, edgecolor="black")
    ax.set_title("Posterior Success Probability (end-to-end)")
    ax.set_xlabel("Success prob"); ax.set_ylabel("Count")
    _annotate_percentiles(ax, succ_chain_plot, money=False)

    ax = axs[1,0]
    ax.hist(succ_per_year_plot, bins=60, edgecolor="black")
    ax.set_title("Successful Incidents / Year (posterior)")
    ax.set_xlabel("Incidents/year"); ax.set_ylabel("Count")
    _annotate_percentiles(ax, succ_per_year_plot, money=False)

    ax = axs[1,1]
    ax.hist(losses_plot, bins=60, edgecolor="black")
    ax.set_title("Annual Loss (posterior predictive)")
    ax.set_xlabel("Annual loss"); ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(FuncFormatter(_millions))
    _annotate_percentiles(ax, losses_plot, money=True)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(top=0.90)
    dash_path = os.path.join(OUTPUT_DIR, f"dashboard_2x2_{ts}.png")
    fig.savefig(dash_path, dpi=150)
    print(f"✅ Saved 2×2 dashboard → {dash_path}")

    # Log-scale ALE histogram
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    bins = np.logspace(np.log10(1e3), np.log10(max(1e5, max(1.0, losses_plot.max()))), 60)
    ax2.hist(losses_plot, bins=bins, edgecolor="black")
    ax2.set_xscale('log')
    ax2.set_title("Annualized Loss (Log Scale)")
    ax2.set_xlabel("Annual loss (log)"); ax2.set_ylabel("Count")
    ax2.xaxis.set_major_formatter(FuncFormatter(_millions))
    _annotate_percentiles(ax2, losses_plot, money=True)
    fig2.tight_layout()
    ale_path = os.path.join(OUTPUT_DIR, f"ale_log_chart_{ts}.png")
    fig2.savefig(ale_path, dpi=150)
    print(f"✅ Saved ALE chart → {ale_path}")

    # Loss Exceedance Curve (LEC)
    sorted_losses = np.sort(losses_plot)
    exceed_probs = 1.0 - np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)
    exceed_probs_percent = exceed_probs * 100

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.plot(sorted_losses, exceed_probs_percent, lw=2, color="orange")
    ax3.set_xscale('log')
    ax3.set_xlabel("Annual Loss")
    ax3.set_ylabel("Exceedance Probability (%)")
    ax3.set_title("Loss Exceedance Curve (Annual Loss)")
    ax3.grid(True, which="both", ls="--", lw=0.5)
    ax3.xaxis.set_major_formatter(FuncFormatter(_millions))
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))

    pcts = [50, 90, 95, 99]
    vals = np.percentile(sorted_losses, pcts)
    for p, v in zip(pcts, vals):
        prob = 100 * (1 - p / 100.0)
        ax3.axvline(v, ls="--", lw=0.8, color="gray")
        y_text = min(100, prob + 5)
        ax3.text(v, y_text, f"P{p}\n${v:,.0f}",
                 rotation=90, va="bottom", ha="left", fontsize=8,
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))

    lec_path = os.path.join(OUTPUT_DIR, f"loss_exceedance_curve_{ts}.png")
    fig3.tight_layout()
    fig3.savefig(lec_path, dpi=150)
    print(f"✅ Saved Loss Exceedance Curve → {lec_path}")

    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"⚠️ Could not display figures: {e}")

def _save_results_csvs(losses: np.ndarray, successes: np.ndarray,
                       lambda_draws: np.ndarray, success_chain_draws: np.ndarray):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    results_csv = os.path.join(OUTPUT_DIR, f"cyber_risk_simulation_results_{ts}.csv")
    pd.DataFrame({
        "lambda": lambda_draws,
        "p_success_chain": success_chain_draws,
        "annual_loss": losses,
        "successful_incidents": successes
    }).to_csv(results_csv, index=False)

    aal_mean   = float(np.mean(losses))
    aal_median = float(np.median(losses))
    aal_lo, aal_hi = np.quantile(losses, [0.025, 0.975])

    mean_succ  = float(np.mean(successes))
    succ_lo, succ_hi = np.quantile(successes, [0.025, 0.975])
    pct_zero   = float(np.mean(successes == 0) * 100.0)

    valid = successes > 0
    if np.any(valid):
        per_event_losses = np.divide(losses[valid], successes[valid],
                                     out=np.zeros_like(losses[valid]),
                                     where=successes[valid] > 0)
        mean_loss_per_event = float(np.mean(per_event_losses))
        lo_event, hi_event  = np.quantile(per_event_losses, [0.025, 0.975])
    else:
        mean_loss_per_event, lo_event, hi_event = 0.0, 0.0, 0.0

    summary_csv = os.path.join(OUTPUT_DIR, f"cyber_risk_simulation_summary_{ts}.csv")
    pd.DataFrame([{
        "Mean_AAL": aal_mean,
        "Median_AAL": aal_median,
        "AAL_95_Lower": aal_lo,
        "AAL_95_Upper": aal_hi,
        "Mean_Incidents": mean_succ,
        "Zero_Incident_Years_%": pct_zero,
        "n": int(losses.size),
        "Incidents_95_Lower": succ_lo,
        "Incidents_95_Upper": succ_hi,
        "Mean_Loss_Per_Incident": mean_loss_per_event,
        "Loss_Per_Incident_95_Lower": lo_event,
        "Loss_Per_Incident_95_Upper": hi_event,
        "Mean_AAL_Check_MeanInc_x_MeanLossPerIncident": mean_succ * mean_loss_per_event
    }]).to_csv(summary_csv, index=False)

    print(f"✅ Detailed results exported → {results_csv}")
    print(f"✅ Summary statistics exported → {summary_csv}")

# =============================================================================
# CLI + main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Cyber incident model with MITRE-informed controls + PyMC + FAIR.")
    p.add_argument("--dataset", default="enterprise-attack.json", help="MITRE ATT&CK STIX bundle path.")
    p.add_argument("--csv", default="mitigation_control_strengths.csv", help="Mitigation strengths CSV path.")
    p.add_argument("--no-adapt-stochastic", action="store_true",
                   help="Disable stochastic adaptation (use fixed factor)")
    p.add_argument("--no-plot", action="store_true", help="Save figures but do not open GUI windows.")
    p.add_argument("--print-control-strengths", action="store_true",
                   help="Print the per-tactic control strength parameters used (for diagnostics).")
    p.add_argument("--no-stochastic-impact", action="store_true",
                   help="Disable stochastic impact reduction (use mean instead).")
    return p.parse_args()

def main():
    global ADAPTATION_FACTOR, ADAPTATION_STOCHASTIC, STOCHASTIC_IMPACT_REDUCTION

    args = parse_args()
    if args.no_adapt_stochastic:
        ADAPTATION_STOCHASTIC = False
    if args.no_stochastic_impact:
        STOCHASTIC_IMPACT_REDUCTION = False

    # Load tactic subset & ranges from dashboard or fallback to SME map
    tactics_included, stage_map, impact_controls, mode = _load_from_dashboard_or_fallback(args.dataset, args.csv)

    if args.print_control_strengths:
        _print_stage_control_map(stage_map, tactics_included)

    # Build Beta priors for per-stage success over the included tactics
    alphas, betas = _build_beta_priors_from_stage_map(stage_map, tactics_included)

    # Sample posterior (λ and end-to-end success probability)
    lambda_draws, success_chain_draws, succ_mat = _sample_posterior_lambda_and_success(
        alphas, betas, n_stages=len(tactics_included)
    )

    # Posterior predictive simulation (annual losses & incident counts)
    losses, successes = _simulate_annual_losses(
        lambda_draws=lambda_draws,
        succ_chain_draws=success_chain_draws,
        succ_mat=succ_mat,
        alphas=alphas,
        betas=betas,
        tactics_included=tactics_included,
        impact_reduction_controls=impact_controls,
        severity_median=500_000.0,
        severity_gsd=2.0,
        rng_seed=RANDOM_SEED + 1,
    )

    _print_aal_summary(losses, successes)
    _save_results_csvs(losses, successes, lambda_draws, success_chain_draws)
    _render_2x2_and_log_ale(losses, lambda_draws, success_chain_draws, show=(not args.no_plot))

# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unhandled error: {e}")
        sys.exit(2)