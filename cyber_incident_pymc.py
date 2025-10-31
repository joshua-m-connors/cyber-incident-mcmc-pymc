"""
=====================================================================================
Cyber Incident Risk Model with MITRE ATT&CK + FAIR (Instructional & Explanatory)
=====================================================================================
PURPOSE
-------------------------------------------------------------------------------------
This script builds a *quantitative* cyber-risk model that ties together:
  • MITRE ATT&CK stage-level defensive control strength → per‑stage success priors.
  • A Bayesian model (PyMC) for attacker attempt frequency and stage success.
  • FAIR‑style per‑incident loss categories, including heavy‑tailed regulatory and
    reputational components (Pareto tails).

The outcome is a *posterior predictive* view of annualized loss, supporting
decision-making (e.g., “Which levers move AAL?” “How strong are our controls by
tactic?”). Visual outputs include a 2×2 dashboard, a log-scale ALE‑style histogram,
and a loss exceedance curve (LEC), all saved under a date‑stamped output folder.

HOW THE PIECES FIT
-------------------------------------------------------------------------------------
1) Control strengths per MITRE tactic are loaded from `mitre_control_strength_dashboard`
   (or a SME fallback map). Strength is treated as a *block* (0–0.95) and inverted to
   success intervals per stage (success = 1 – block). These intervals parameterize Beta
   priors for each stage’s success probability.
2) An “attacker adaptation factor” degrades the control block (e.g., 0.75 ⇒ 25% weaker
   than nominal) to reflect learning/novelty. You can hold it fixed or sample once per
   run from a range (stochastic adaptation).
3) PyMC samples posterior draws for:
   • λ (attempts/year) using a lognormal prior implied by a 90% CI.
   • Per‑stage success probabilities (Beta priors). The end‑to‑end success probability
     for a chain is the product of stage probabilities.
4) Posterior predictive simulation: for each posterior draw, simulate annual attempts,
   pass each through a MITRE chain with retries, detection, and fallback logic, and draw
   per‑category losses. Aggregate to annual totals.
5) Exports: CSVs for detailed samples and a compact summary, plus PNG charts.

WHAT USERS MOST OFTEN TUNE (READ THIS!)
-------------------------------------------------------------------------------------
• Frequency prior (CI_MIN_FREQ / CI_MAX_FREQ)
  - Meaning: Your belief about the 90% range of *attempts per year*.
  - When to change: If you have telemetry suggesting a different threat tempo.

• PyMC sampler knobs (N_SAMPLES, N_TUNE, N_CHAINS, TARGET_ACCEPT)
  - Meaning: Speed vs. accuracy tradeoff for MCMC.
  - When to change: Faster iteration (smaller draws) vs. tighter intervals (larger).

• Monte Carlo per-draw size (N_SIM_PER_DRAW) and CPU workers (N_WORKERS)
  - Meaning: How many simulated attempts per posterior draw, and parallelism.
  - When to change: Increase for smoother predictive distributions.

• MITRE control strengths (provided via the dashboard CSV) and SME fallback ranges
  - Meaning: Tactic‑level *block* intervals in [0, 0.95].
  - When to change: After control reviews, red‑team results, or measurement updates.

• Attacker adaptation (ADAPTATION_FACTOR / ADAPTATION_STOCHASTIC / ADAPTATION_STOCHASTIC_RANGE)
  - Meaning: Degrades the nominal control block (e.g., 0.75 ⇒ 75% of block remains).
  - When to change: Stress tests (“what if attackers adapt more quickly?”).

• Loss assumptions
  - `loss_q5_q95`: 90% CIs (per incident) for each FAIR category’s lognormal body.
  - `pareto_defaults`: Tail frequency and severity for Regulatory/Legal & Reputation.
  - When to change: Calibrate with claims data, incident post‑mortems, or expert input.

INTERPRETING OUTPUTS
-------------------------------------------------------------------------------------
• “Successful Incidents / Year”: Derived from λ × (end‑to‑end chain success).
• “Annual Loss (posterior predictive)”: Includes category allocations & capped tails.
• LEC: Read “% chance annual loss exceeds $X”. Annotated P50/P90/P95/P99 lines help.

REPRODUCIBILITY
-------------------------------------------------------------------------------------
Random seeds are set for both the Bayesian sampling and the predictive simulation.
Charts and CSVs are written under `output_YYYY-MM-DD/` in the script folder.

IMPORTANT: CODE INTEGRITY
-------------------------------------------------------------------------------------
This instructional version *adds comments and docstrings only*. No logic, names,
or defaults are changed below.
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

# ---------- Optional dependencies ----------
try:
    import pymc as pm
    HAVE_PYMC = True
except Exception:
    HAVE_PYMC = False

# MITRE analyzer (tactic-level control strengths)
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
# 2) GLOBAL CONFIG — priors, runtime, plotting
# =============================================================================
# Frequency prior (attempts/year), elicited as a 90% CI -> Lognormal params
CI_MIN_FREQ = 4
CI_MAX_FREQ = 24
Z_90 = 1.645

# PyMC sampling (tune for speed vs. accuracy)
N_SAMPLES = 4000
N_TUNE = 1000
N_CHAINS = 4
TARGET_ACCEPT = 0.90
RANDOM_SEED = 42

# Parallel Monte Carlo (per posterior draw attacker simulations)
N_WORKERS = None              # None -> use all logical cores
N_SIM_PER_DRAW = 1000         # Monte Carlo attempts per posterior draw

# Attacker progression controls
MAX_RETRIES_PER_STAGE = 3          # number of retries per MITRE stage
RETRY_PENALTY = 0.90               # success decay per retry
FALLBACK_PROB = 0.25               # chance of tactical fallback to prior stage
DETECT_BASE = 0.01                 # base detection probability per stage
DETECT_INC_PER_RETRY = 0.03       # incremental detection chance per retry
MAX_FALLBACKS_PER_CHAIN = 3        # max fallbacks allowed in one attack chain

# Visualization (console & charts)
PLOT_IN_MILLIONS = True

# Attacker adaptation (how much of the nominal control block actually works)
ADAPTATION_FACTOR = 0.99                 # deterministic fallback
ADAPTATION_STOCHASTIC = True             # if True, sample once per run
ADAPTATION_STOCHASTIC_RANGE = (0.85, 0.95)

# Optional Observed Data (kept None by default)
observed_total_incidents = None    # e.g., 7 if 7 incidents observed
observed_years = None              # e.g., 3 if over 3 years

# =============================================================================
# 3) MITRE STAGES (fixed order for Enterprise)
# =============================================================================
MITRE_STAGES = [
    "Initial Access","Execution","Persistence","Privilege Escalation","Defense Evasion",
    "Credential Access","Discovery","Lateral Movement","Collection","Command And Control",
    "Exfiltration","Impact",
]

# SME fallback (control block in [0,0.95])
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
# 4) FAIR TAXONOMY — loss categories & distributions (heavy-tailed)
# =============================================================================
loss_categories = ["Productivity", "ResponseContainment", "RegulatoryLegal", "ReputationCompetitive"]

# 90% subject-matter CIs for per-incident loss body (lognormal base)
loss_q5_q95 = {
    "Productivity": (1_000, 200_000),
    "ResponseContainment": (10_000, 1_000_000),
    "RegulatoryLegal": (0, 3_000_000),
    "ReputationCompetitive": (0, 5_000_000),
}

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

# Pareto tail params for secondary losses (Type I Pareto)
pareto_defaults = {
    "RegulatoryLegal":       {"xm": 50_000.0, "alpha": 3.5},
    "ReputationCompetitive": {"xm": 100_000.0, "alpha": 2.75},
}

# =============================================================================
# 5) Output directory (daily, reused)
# =============================================================================
def make_output_dir(prefix="output"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    date_str = datetime.now().strftime("%Y-%m-%d")
    out_dir = os.path.join(base_dir, f"{prefix}_{date_str}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 Output directory: {out_dir}")
    return out_dir

OUTPUT_DIR = make_output_dir()
# Per-category arrays from last simulation (used by summary printing)
LAST_CATEGORY_LOSSES = None

# =============================================================================
# 6) Helpers: MITRE controls → success priors (Beta), formatting, etc.
# =============================================================================
def _load_stage_control_map_from_analyzer(dataset_path: str, csv_path: str = "mitigation_control_strengths.csv"):
    """
    Loads tactic-level control ranges from MITRE analyzer if available;
    else SME fallback. Returns {tactic: (control_min, control_max)} in [0..1].
    """
    if HAVE_MITRE_ANALYZER:
        try:
            df, _ = get_mitre_tactic_strengths(dataset_path, csv_path, build_figure=False)
            control_map = {}
            for _, r in df.iterrows():
                t = str(r["Tactic"])
                lo = max(0.0, min(95.0, float(r["MinStrength"]))) / 100.0
                hi = max(0.0, min(95.0, float(r["MaxStrength"]))) / 100.0
                if lo > hi:
                    lo, hi = hi, lo
                control_map[t] = (lo, hi)
            # ensure full canonical coverage
            for t in MITRE_STAGES:
                control_map.setdefault(t, _SME_STAGE_CONTROL_MAP_FALLBACK[t])
            print("✅ Loaded control strengths from MITRE ATT&CK dataset.")
            return control_map
        except Exception as e:
            print(f"⚠️ MITRE dataset load failed: {e}. Using SME fallback.")
            return _SME_STAGE_CONTROL_MAP_FALLBACK.copy()
    else:
        return _SME_STAGE_CONTROL_MAP_FALLBACK.copy()
    
def _print_stage_control_map(stage_map):
    """Prints the loaded control strength ranges by MITRE tactic."""
    print("\n--- Tactic Control Strength Parameters Used ---")
    print(f"{'Tactic':<25} {'MinStrength':>12} {'MaxStrength':>12}")
    for t in MITRE_STAGES:
        lo, hi = stage_map.get(t, (0.0, 0.0))
        print(f"{t:<25} {lo*100:>11.1f}% {hi*100:>11.1f}%")
    print("------------------------------------------------")
    
    # Optional: also save to CSV for diagnostics
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = os.path.join(OUTPUT_DIR, f"tactic_control_strengths_{ts}.csv")
    pd.DataFrame([
        {"Tactic": t, "MinStrength": lo * 100, "MaxStrength": hi * 100}
        for t, (lo, hi) in stage_map.items()
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
    """
    Format currency for console; defaults to PLOT_IN_MILLIONS if millions=None.
    """
    if millions is None:
        millions = PLOT_IN_MILLIONS
    if millions:
        return f"${x/1_000_000:,.2f}M"
    return f"${x:,.0f}"

# =============================================================================
# 7) Posterior & simulation core
# =============================================================================
def _build_beta_priors_from_stage_map(stage_map):
    """Return alphas, betas in MITRE_STAGES order, applying attacker adaptation."""
    rng = np.random.default_rng(RANDOM_SEED)

    if ADAPTATION_STOCHASTIC:
        adapt_lo, adapt_hi = ADAPTATION_STOCHASTIC_RANGE
        adaptation_factor = float(rng.uniform(adapt_lo, adapt_hi))
        color = YELLOW
    else:
        adaptation_factor = float(ADAPTATION_FACTOR)
        color = GREEN

    print(f"{color}🔧 Using attacker adaptation factor = {adaptation_factor:.3f} "
          f"(stochastic={ADAPTATION_STOCHASTIC}){RESET}")

    alphas, betas = [], []
    for t in MITRE_STAGES:
        blo, bhi = stage_map[t]  # control block fractions [0..0.95]
        # apply adaptation (controls are less effective by adaptation_factor)
        blo_eff = max(0.0, min(0.95, adaptation_factor * blo))
        bhi_eff = max(0.0, min(0.95, adaptation_factor * bhi))

        slo, shi = _success_interval_from_control(blo_eff, bhi_eff)
        a, b = _beta_from_interval(slo, shi, strength=50.0)
        alphas.append(a); betas.append(b)
    return np.array(alphas), np.array(betas)

def _simulate_attacker_path(success_probs, rng):
    """
    Simulates an attacker progressing through MITRE stages with:
      - retries (MAX_RETRIES_PER_STAGE)
      - retry penalty (RETRY_PENALTY)
      - increasing detection per retry (DETECT_BASE, DETECT_INC_PER_RETRY)
      - fallback to prior tactic on full-stage failure (FALLBACK_PROB)
      - a max number of fallbacks to avoid infinite loops

    Args:
        success_probs (list[float]): per-stage success probabilities (0..1)
        rng (np.random.Generator): random generator

    Returns:
        bool: True if attacker reaches final stage ('Impact'), False otherwise
    """
    i = 0
    n_stages = len(success_probs)
    fallback_count = 0

    # Defensive: if there are zero stages, treat as failure
    if n_stages == 0:
        return False

    # Walk stages explicitly (so we can step back on fallback)
    while 0 <= i < n_stages:
        # use the nominal per-stage probability for this attempt
        p_nominal = float(success_probs[i])
        detect_prob = DETECT_BASE

        # Attempt retries for this stage
        for r in range(MAX_RETRIES_PER_STAGE):
            if rng.random() < p_nominal:
                # success at this stage -> move forward
                i += 1
                break

            # failure this attempt -> increase detect probability
            detect_prob = min(1.0, detect_prob + DETECT_INC_PER_RETRY)
            if rng.random() < detect_prob:
                # detected and stopped
                return False

            # apply retry penalty before next attempt
            p_nominal *= RETRY_PENALTY

        else:
            # All retries at this stage exhausted
            if rng.random() < FALLBACK_PROB and fallback_count < MAX_FALLBACKS_PER_CHAIN:
                # Fallback: step back one stage (but not before stage 0)
                fallback_count += 1
                i = max(0, i - 1)
                # continue while-loop to attempt from the prior stage again
                continue
            else:
                # No fallback (or exceeded fallback budget) -> chain fails
                return False

    # If we exit the loop because i == n_stages, attacker reached final stage
    return i >= n_stages

def _sample_posterior_lambda_and_success(alphas: np.ndarray, betas: np.ndarray):
    """
    Build and sample the PyMC model for λ (attacks/year) and per-stage success.
    Returns:
      lambda_draws (N,), success_chain_draws (N,)
    """
    if not HAVE_PYMC:
        # Prior-only fallback to keep script runnable without PyMC
        rng = np.random.default_rng(RANDOM_SEED)
        mu_l = np.log(np.sqrt(CI_MIN_FREQ * CI_MAX_FREQ))
        sig_l = (np.log(CI_MAX_FREQ) - np.log(CI_MIN_FREQ)) / (2.0 * Z_90)
        lam = rng.lognormal(mean=mu_l, sigma=sig_l, size=N_SAMPLES)
        succ_chain = np.prod(rng.beta(alphas, betas, size=(N_SAMPLES, len(MITRE_STAGES))), axis=1)
        return lam, succ_chain

    with pm.Model() as model:
        # Lognormal for attempt frequency
        mu_lambda = np.log(np.sqrt(CI_MIN_FREQ * CI_MAX_FREQ))
        sigma_lambda = (np.log(CI_MAX_FREQ) - np.log(CI_MIN_FREQ)) / (2.0 * Z_90)
        lambda_rate = pm.Lognormal("lambda_rate", mu=mu_lambda, sigma=sigma_lambda)

        # Per-stage success probabilities (Beta priors)
        success_probs = pm.Beta("success_probs", alpha=alphas, beta=betas, shape=len(MITRE_STAGES))

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
    succ_mat = np.asarray(trace.posterior["success_probs"]).reshape(-1, len(MITRE_STAGES))
    succ_chain_draws = np.prod(succ_mat, axis=1)
    return lambda_draws, succ_chain_draws, succ_mat

def _simulate_annual_losses(lambda_draws, succ_chain_draws, succ_mat,
                            alphas, betas,
                            severity_median=500_000.0,
                            severity_gsd=2.0,
                            rng_seed=1234):

    """
    Posterior predictive Monte Carlo per draw:
      - For each posterior draw, simulate attempts as Poisson(λ)
      - Each attempt succeeds with p_succ (end-to-end)
      - If success, draw base loss (lognormal) then allocate categories & apply Pareto tails
    Returns:
      losses (N,), successes (N,)
    Side effect:
      - Populates LAST_CATEGORY_LOSSES dict with arrays for each category
    """
    global LAST_CATEGORY_LOSSES

    rng = np.random.default_rng(rng_seed)
    mu = math.log(max(1e-9, severity_median))
    sigma = math.log(max(1.000001, severity_gsd))

    n = len(lambda_draws)
    losses = np.zeros(n, dtype=float)
    successes = np.zeros(n, dtype=int)

    prod_losses = np.zeros(n, dtype=float)
    resp_losses = np.zeros(n, dtype=float)
    reg_losses  = np.zeros(n, dtype=float)
    rep_losses  = np.zeros(n, dtype=float)

    for idx, (lam, p_succ) in enumerate(zip(lambda_draws, succ_chain_draws)):
        attempts = rng.poisson(lam=lam)
        succ_count = 0
        prod_acc = resp_acc = reg_acc = rep_acc = 0.0
        total_loss = 0.0

        for _ in range(attempts):
            # Draw per-stage success probabilities from the same Beta priors
            # Use posterior success_probs for this draw (if available)
            if succ_mat is not None:
                stage_success_probs = succ_mat[idx].astype(float)
            else:
                stage_success_probs = rng.beta(alphas, betas).astype(float)

            # --- SAMPLE PER-ATTACKER ADAPTATION FACTOR AND ADJUST SUCCESS PROBS ---
            # If ADAPTATION_STOCHASTIC is True, draw a unique adaptation factor
            # for this attacker from ADAPTATION_STOCHASTIC_RANGE; otherwise use
            # the fixed ADAPTATION_FACTOR.
            if ADAPTATION_STOCHASTIC:
                adapt_lo, adapt_hi = ADAPTATION_STOCHASTIC_RANGE
                attacker_adaptation = float(rng.uniform(adapt_lo, adapt_hi))
            else:
                attacker_adaptation = float(ADAPTATION_FACTOR)

            # Convert nominal success -> effective success under this attacker's adaptation.
            # Original model applies adaptation by scaling the control block:
            #   block_eff = adaptation * block  where block = 1 - success_nominal
            # So success_eff = 1 - block_eff = 1 - adaptation * (1 - success_nominal)
            stage_success_probs = 1.0 - attacker_adaptation * (1.0 - stage_success_probs)
            # numerical safety: clamp to [0,1]
            stage_success_probs = np.clip(stage_success_probs, 0.0, 1.0)

            # Optional debug: print sampled adaptation occasionally (uncomment if desired)
            # if idx % max(1, int(len(lambda_draws)/10)) == 0:
            #     print(f"Sim idx {idx}: attacker_adaptation={attacker_adaptation:.4f}")
            # Simulate full progression through all MITRE stages using per-attacker probs
            if _simulate_attacker_path(stage_success_probs, rng):

                # --- draw per-category losses (bounded lognormal + bounded Pareto tails) ---
                prod = resp = reg = rep = 0.0

                for j, cat in enumerate(loss_categories):
                    mu_j, sigma_j = cat_mu[j], cat_sigma[j]

                    # Lognormal body capped at 95th percentile
                    base_draw = float(rng.lognormal(mean=mu_j, sigma=sigma_j))
                    lognorm_cap = math.exp(mu_j + 3.09 * sigma_j)
                    base_draw = min(base_draw, lognorm_cap)

                    if cat == "RegulatoryLegal":
                        reg = base_draw
                        if rng.random() < 0.025:
                            xm, alpha = pareto_defaults[cat]["xm"], pareto_defaults[cat]["alpha"]
                            u = rng.uniform(0.001, 0.999)
                            tail_draw = xm * (1.0 - u) ** (-1.0 / alpha)
                            tail_cap  = xm * (0.95) ** (-1.0 / alpha)   # ≈95th pct cap
                            reg = max(reg, min(tail_draw, tail_cap))

                    elif cat == "ReputationCompetitive":
                        rep = base_draw
                        if rng.random() < 0.015:
                            xm, alpha = pareto_defaults[cat]["xm"], pareto_defaults[cat]["alpha"]
                            u = rng.uniform(0.001, 0.999)
                            tail_draw = xm * (1.0 - u) ** (-1.0 / alpha)
                            tail_cap  = xm * (0.95) ** (-1.0 / alpha)   # ≈95th pct cap
                            rep = max(rep, min(tail_draw, tail_cap))

                    elif cat == "Productivity":
                        prod = base_draw

                    elif cat == "ResponseContainment":
                        resp = base_draw

                # accumulate for this successful incident
                prod_acc += prod
                resp_acc += resp
                reg_acc  += reg
                rep_acc  += rep
                total_loss += (prod + resp + reg + rep)
                succ_count += 1

        # write results for this posterior draw (use idx, not i)
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
# 8) Console output, viz, exports (GitHub-style behavior)
# =============================================================================
def _print_aal_summary(losses: np.ndarray, successes: np.ndarray):
    aal_mean   = float(np.mean(losses))
    aal_median = float(np.median(losses))
    lo, hi     = np.quantile(losses, [0.025, 0.975])
    mean_succ  = float(np.mean(successes))
    succ_lo, succ_hi = np.quantile(successes, [0.025, 0.975])
    pct_zero   = float(np.mean(successes == 0) * 100.0)

    print("\nAAL posterior predictive summary (with fallback, severity, Pareto tails):")
    print(f"Mean AAL: {_fmt_money(aal_mean)}")
    print(f"Median AAL: {_fmt_money(aal_median)}")
    print(f"AAL 95% credible interval (annualized total loss): {_fmt_money(lo)} – {_fmt_money(hi)}")

    # --- Successful incidents / year ---
    print(f"Mean successful incidents / year: {mean_succ:.2f}")
    print(f"95% credible interval (incidents / year): {succ_lo:.2f} – {succ_hi:.2f}")

    # --- Mean loss per successful incident (Single Loss Expectancy) ---
    valid = successes > 0
    if np.any(valid):
        # Compute per-draw loss per incident, then summarize
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

    # ---------- Category breakdown ----------
    print("\nCategory-level annual loss 95% credible intervals:")
    if LAST_CATEGORY_LOSSES is not None:
        med_aal = max(1e-12, aal_median)
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
    """
    Draw P50, P90, P95, P99 vertical lines and label each line with its value (on the line).
    money=True -> use _fmt_money formatting; else numeric compact.
    """
    pcts = [50, 90, 95, 99]
    vals = np.percentile(samples, pcts)

    ymin, ymax = ax.get_ylim()
    ytext = ymax * 0.95  # place text near top of axis

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

        # --- Auto-clip helper (reintroduced from GitHub version) ---
    def _auto_clip(data, low=0.001, high=0.991):
        """Clips data to percentile range for better visualization of long tails."""
        if len(data) == 0:
            return data
        low_v, high_v = np.percentile(data, [low * 100, high * 100])
        return data[(data >= low_v) & (data <= high_v)]

    # Apply clipping to all plotted arrays to avoid distortion from long tails
    lambda_plot = _auto_clip(lambda_draws)
    succ_chain_plot = _auto_clip(success_chain_draws)
    succ_per_year_plot = _auto_clip(succ_per_year)
    losses_plot = _auto_clip(losses)

    def _millions(x, pos): return f"${x/1e6:,.1f}M"

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    # (1) Posterior λ
    ax = axs[0,0]
    ax.hist(lambda_plot, bins=60, edgecolor="black")
    ax.set_title("Posterior λ (incidents/year)")
    ax.set_xlabel("λ"); ax.set_ylabel("Count")
    _annotate_percentiles(ax, lambda_plot, money=False)

    # (2) Posterior end-to-end success probability
    ax = axs[0,1]
    ax.hist(succ_chain_plot, bins=60, edgecolor="black")
    ax.set_title("Posterior Success Probability (end-to-end)")
    ax.set_xlabel("Success prob"); ax.set_ylabel("Count")
    _annotate_percentiles(ax, succ_chain_plot, money=False)

    # (3) Successful incidents/year
    ax = axs[1,0]
    ax.hist(succ_per_year_plot, bins=60, edgecolor="black")
    ax.set_title("Successful Incidents / Year (posterior)")
    ax.set_xlabel("Incidents/year"); ax.set_ylabel("Count")
    _annotate_percentiles(ax, succ_per_year_plot, money=False)

    # (4) Annual loss (posterior predictive)
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

    # Separate log-scale ALE-style histogram
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    bins = np.logspace(np.log10(1e3), np.log10(max(1e5, max(1.0, losses_plot.max()))), 60)
    ax2.hist(losses_plot, bins=bins, edgecolor="black")
    ax2.set_xscale('log')
    ax2.set_title("Annualized Loss (Log Scale)")
    ax2.set_xlabel("Annual loss (log)"); ax2.set_ylabel("Count")
    ax2.xaxis.set_major_formatter(FuncFormatter(_millions))
    # --- Add percentile annotations directly on the bars (GitHub-style) ---
    _annotate_percentiles(ax2, losses_plot, money=True)

    fig2.tight_layout()

    ale_path = os.path.join(OUTPUT_DIR, f"ale_log_chart_{ts}.png")
    fig2.savefig(ale_path, dpi=150)
    print(f"✅ Saved ALE chart → {ale_path}")

    # ---  NEW: Loss Exceedance Curve (LEC) ---
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

    # --- Annotate key percentiles with vertical lines and $ amounts ---
    pcts = [50, 90, 95, 99]
    vals = np.percentile(sorted_losses, pcts)
    for p, v in zip(pcts, vals):
        prob = 100 * (1 - p / 100.0)  # P50=50%, P90=10%, etc.
        ax3.axvline(v, ls="--", lw=0.8, color="gray")

        # Place label slightly above the line and offset to the right
        y_text = min(100, prob + 5)  # avoid going above chart top
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

    # ---------- Detailed results (unchanged) ----------
    results_csv = os.path.join(OUTPUT_DIR, f"cyber_risk_simulation_results_{ts}.csv")
    pd.DataFrame({
        "lambda": lambda_draws,
        "p_success_chain": success_chain_draws,
        "annual_loss": losses,
        "successful_incidents": successes
    }).to_csv(results_csv, index=False)

    # ---------- Summary statistics (expanded) ----------
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

    # ---------- Export summary ----------
    summary_csv = os.path.join(OUTPUT_DIR, f"cyber_risk_simulation_summary_{ts}.csv")
    pd.DataFrame([{
        # Existing fields
        "Mean_AAL": aal_mean,
        "Median_AAL": aal_median,
        "AAL_95_Lower": aal_lo,
        "AAL_95_Upper": aal_hi,
        "Mean_Incidents": mean_succ,
        "Zero_Incident_Years_%": pct_zero,
        "n": int(losses.size),
        # New fields
        "Incidents_95_Lower": succ_lo,
        "Incidents_95_Upper": succ_hi,
        "Mean_Loss_Per_Incident": mean_loss_per_event,
        "Loss_Per_Incident_95_Lower": lo_event,
        "Loss_Per_Incident_95_Upper": hi_event,
        # Optional derived consistency check
        "Mean_AAL_Check_MeanInc_x_MeanLossPerIncident": mean_succ * mean_loss_per_event
    }]).to_csv(summary_csv, index=False)

    print(f"✅ Detailed results exported → {results_csv}")
    print(f"✅ Summary statistics exported → {summary_csv}")

# =============================================================================
# 9) CLI + main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Cyber incident model with MITRE-informed controls + PyMC + FAIR.")
    p.add_argument("--dataset", default="enterprise-attack.json", help="MITRE ATT&CK STIX bundle path.")
    p.add_argument("--csv", default="mitigation_control_strengths.csv", help="Mitigation strengths CSV path.")
    p.add_argument("--adaptation", type=float, default=None,
                   help="Override adaptation factor (default=0.75)")
    p.add_argument("--no-adapt-stochastic", action="store_true",
                   help="Disable stochastic adaptation (use fixed factor)")
    p.add_argument("--no-plot", action="store_true", help="Save figures but do not open GUI windows.")
    p.add_argument("--print-control-strengths", action="store_true",
               help="Print the per-tactic control strength parameters used (for diagnostics).")

    return p.parse_args()

def main():
    global ADAPTATION_FACTOR, ADAPTATION_STOCHASTIC

    args = parse_args()
    if args.adaptation is not None:
        ADAPTATION_FACTOR = args.adaptation
    if args.no_adapt_stochastic:
        ADAPTATION_STOCHASTIC = False

    # Load MITRE tactic-level control strengths (or SME fallback)
    stage_map = _load_stage_control_map_from_analyzer(args.dataset, args.csv)
    if args.print_control_strengths:
        _print_stage_control_map(stage_map)

    # Build per-stage success priors (Beta) from control strengths (applies adaptation inside)
    alphas, betas = _build_beta_priors_from_stage_map(stage_map)

    # PyMC posterior sampling (lambda and end-to-end success)
    if HAVE_PYMC:
        lambda_draws, success_chain_draws, succ_mat = _sample_posterior_lambda_and_success(alphas, betas)
    else:
        # Fallback: run prior-only simulation if PyMC is unavailable
        lambda_draws, success_chain_draws = _sample_posterior_lambda_and_success(alphas, betas)
        succ_mat = None

    # Posterior predictive Monte Carlo for annual losses + incident counts
    losses, successes = _simulate_annual_losses(
        lambda_draws=lambda_draws,
        succ_chain_draws=success_chain_draws,
        succ_mat=succ_mat,
        alphas=alphas,
        betas=betas,
        severity_median=500_000.0,
        severity_gsd=2.0,
        rng_seed=RANDOM_SEED + 1,
    )

    # Console output (uses per-category arrays if available)
    _print_aal_summary(losses, successes)

    # CSV exports and visualizations (2×2 + log-scale)
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
