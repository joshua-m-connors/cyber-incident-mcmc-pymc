#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cyber_risk_simulation_main_explained_final.py

PURPOSE
-------
A comprehensive, end-to-end Bayesian cyber risk simulation that:
  1) infers attacker frequency and per-stage success from priors (PyMC),
  2) simulates attacker progression through MITRE ATT&CK as a Markov process
     (retries with diminishing returns, fallback, increasing detection),
  3) quantifies financial impact using a FAIR-aligned taxonomy with heavy tails,
  4) executes per-draw Monte Carlo in parallel (safe, top-level worker),
  5) summarizes Annualized Loss (AAL) and category-level credible intervals,
  6) visualizes key posteriors in a 2×2 grid and a separate log-scale tail plot.

This file is structured for safe multiprocessing on all OSes (Windows/macOS/Linux)
and can be run as a script. It can also be imported and `main()` invoked explicitly.
"""

# =============================================================================
# 1) IMPORTS — Purpose: all libraries used for inference, simulation, and plots
# =============================================================================

import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pymc as pm
from scipy import stats, optimize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime
import os


# =============================================================================
# 2) GLOBAL CONFIG — Purpose: model priors, runtime controls, plotting options
# =============================================================================
# Frequency prior (attempts/year), elicited as a 90% CI -> Lognormal params
CI_MIN_FREQ, CI_MAX_FREQ, Z_90 = 2, 24, 1.645

# PyMC sampling (tune for speed vs. accuracy)
N_SAMPLES, N_TUNE, N_CHAINS = 1200, 800, 4
TARGET_ACCEPT, RANDOM_SEED = 0.90, 42

# Parallel Monte Carlo (per posterior draw attacker simulations)
N_WORKERS = None              # None -> use all logical cores
N_SIM_PER_DRAW = 400          # Monte Carlo attempts per posterior draw

# Markov attacker behavior
MAX_RETRIES_PER_STAGE = 3     # retries allowed at a stage
RETRY_PENALTY = 0.75          # each retry multiplies base success probability (diminishing returns)
FALLBACK_PROB = 0.25          # probability to step back one stage after exhausting retries
DETECT_BASE = 0.01            # base detection probability on a failed try
DETECT_INC_PER_RETRY = 0.06   # incremental detection added per retry

# Visualization
PLOT_IN_MILLIONS = True       # show currency axes in millions USD


# =============================================================================
# 3) MITRE ATT&CK PRIORS — Purpose: convert control effectiveness to Beta priors
# =============================================================================
# SME-provided 90% CI for control effectiveness per MITRE stage.
# We convert to attacker success: success ∈ [1 - hi, 1 - lo], then fit Beta via quantile-matching.
STAGE_CONTROL_MAP = {
    "Initial Access":        (0.15, 0.50),
    "Execution":             (0.05, 0.30),
    "Persistence":           (0.10, 0.45),
    "Privilege Escalation":  (0.10, 0.35),
    "Defense Evasion":       (0.20, 0.60),
    "Credential Access":     (0.15, 0.50),
    "Discovery":             (0.02, 0.20),
    "Lateral Movement":      (0.25, 0.65),
    "Collection":            (0.05, 0.30),
    "Command and Control":   (0.10, 0.45),
    "Exfiltration":          (0.30, 0.70),
    "Impact":                (0.35, 0.75),
}
MITRE_STAGES = list(STAGE_CONTROL_MAP.keys())


def _quantile_match_beta(p5: float, p95: float, q_low: float = 0.05, q_high: float = 0.95):
    """
    Purpose: derive Beta(α,β) whose 5th and 95th percentiles match (p5, p95).
    This lets you encode SME 90% intervals directly as Beta priors.
    """
    mean = 0.5 * (p5 + p95)
    width = max(p95 - p5, 1e-6)
    concentration_guess = 20.0 * (0.1 / width)   # loose heuristic to start
    a0 = max(1e-3, mean * concentration_guess)
    b0 = max(1e-3, (1.0 - mean) * concentration_guess)

    def resid(params):
        a, b = params
        if a <= 0 or b <= 0:
            return [1e6, 1e6]
        return [stats.beta.ppf(q_low, a, b) - p5,
                stats.beta.ppf(q_high, a, b) - p95]

    sol = optimize.root(resid, [a0, b0], method="hybr")
    if sol.success and np.all(np.array(sol.x) > 0):
        return float(sol.x[0]), float(sol.x[1])
    return a0, b0


# Convert control effectiveness to attacker success 90% CI per stage
_success_90s = [(1.0 - hi, 1.0 - lo) for (lo, hi) in STAGE_CONTROL_MAP.values()]
alphas, betas = zip(*(_quantile_match_beta(lo, hi) for (lo, hi) in _success_90s))
alphas, betas = np.array(alphas), np.array(betas)


# =============================================================================
# 4) FAIR TAXONOMY — Purpose: define loss categories & distributions (heavy-tailed)
# =============================================================================
# Categories: two primary (lognormal bodies), two secondary (lognormal + Pareto tail triggers)
loss_categories = ["Productivity", "ResponseContainment", "RegulatoryLegal", "ReputationCompetitive"]

# Elicited 90% intervals per category (USD)
loss_q5_q95 = {
    "Productivity": (10_000, 150_000),
    "ResponseContainment": (20_000, 500_000),
    "RegulatoryLegal": (0, 1_000_000),
    "ReputationCompetitive": (0, 2_000_000),
}

def _lognormal_from_q5_q95(q5: float, q95: float):
    """
    Purpose: invert 5th/95th percentiles to (mu, sigma) in log-space for Lognormal.
    """
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

# Pareto tail (Type I) params for secondary losses (heavy tails)
pareto_defaults = {
    "RegulatoryLegal":       {"xm": 100_000.0, "alpha": 1.8},
    "ReputationCompetitive": {"xm": 250_000.0, "alpha": 1.5},
}


# =============================================================================
# 5) MARKOV ATTACKER MODEL — Purpose: simulate retries, fallback, detection
# =============================================================================
def simulate_one_attempt(success_probs_stage: np.ndarray,
                         rng: random.Random,
                         max_retries_per_stage: int = MAX_RETRIES_PER_STAGE,
                         retry_penalty: float = RETRY_PENALTY,
                         fallback_prob: float = FALLBACK_PROB,
                         detect_base: float = DETECT_BASE,
                         detect_increase_per_retry: float = DETECT_INC_PER_RETRY) -> bool:
    """
    Simulate a single attacker attempt through ordered MITRE stages.

    Mechanics:
      - At each stage, the attacker gets (retries+1) attempts.
      - Success prob per retry decays: p_try = (retry_penalty**k) * base_p.
      - After each failed try, run a detection check with probability increasing in k.
      - If all retries at a stage fail, attacker may fallback to previous stage.

    Returns:
        True if all stages are completed (full compromise), else False.
    """
    stage_idx = 0
    nstages = len(success_probs_stage)

    while True:
        if stage_idx >= nstages:
            return True  # completed all stages

        p_base = float(success_probs_stage[stage_idx])

        # Attempt initial + retries
        for k in range(max_retries_per_stage + 1):
            p_try = (retry_penalty ** k) * p_base
            if rng.random() < p_try:
                break  # proceed to next stage
            # detection escalates with retries at the current stage
            detect_prob = min(max(detect_base + detect_increase_per_retry * k, 0.0), 1.0)
            if rng.random() < detect_prob:
                return False  # detected & stopped
        else:
            # exhausted retries without success
            if stage_idx == 0 or rng.random() > fallback_prob:
                return False
            stage_idx -= 1  # fallback one stage
            continue

        # success at this stage: move forward
        stage_idx += 1


# =============================================================================
# 6) PARALLEL WORKER — Purpose: per-draw Monte Carlo to estimate success prob
# =============================================================================
def worker_function(args) -> float:
    """
    Top-level (picklable) worker for ProcessPoolExecutor.

    Args:
        args: tuple(per_stage_probs, n_sim, seed)

    Returns:
        Estimated per-attempt success probability for this posterior draw.
    """
    per_stage, n_sim, seed = args
    rng = random.Random(int(seed))
    successes = 0
    for _ in range(int(n_sim)):
        if simulate_one_attempt(np.asarray(per_stage), rng):
            successes += 1
    return successes / float(n_sim)


# =============================================================================
# 7) UTIL: Percentile annotation — Purpose: add P50/P90/P95/P99 lines on plots
# =============================================================================
def _annotate_percentiles(ax, data, percentiles=(50, 90, 95, 99), scale=1.0, money=True):
    ymax = ax.get_ylim()[1]
    for p in percentiles:
        val = np.percentile(data, p) / scale
        ax.axvline(val, color='k', linestyle='--', lw=0.8, alpha=0.8)
        label = f"P{p}=" + (f"${val:,.0f}" if money else f"{val:,.3f}")
        ax.text(val, ymax * 0.92, label, rotation=90, va='top', ha='center',
                fontsize=8, backgroundcolor='white')


# =============================================================================
# 8) MAIN — Purpose: orchestrate inference, parallel MC, loss sim, summaries, plots
# =============================================================================
def main():
    # ---------- 8.1 Build & sample Bayesian model (λ and per-stage success) ----------
    with pm.Model() as model:
        # Frequency prior: lognormal via elicited 90% CI
        mu_lambda = np.log(np.sqrt(CI_MIN_FREQ * CI_MAX_FREQ))
        sigma_lambda = (np.log(CI_MAX_FREQ) - np.log(CI_MIN_FREQ)) / (2.0 * Z_90)
        lambda_rate = pm.Lognormal("lambda_rate", mu=mu_lambda, sigma=sigma_lambda)

        # Per-stage attacker success probabilities (Beta priors from SME intervals)
        success_probs = pm.Beta("success_probs", alpha=alphas, beta=betas, shape=len(MITRE_STAGES))

        # Sample posterior
        trace = pm.sample(draws=N_SAMPLES, tune=N_TUNE, chains=N_CHAINS,
                          target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED, progressbar=True)

    posterior = trace.posterior
    lambda_draws = posterior["lambda_rate"].values.reshape(-1)
    success_probs_draws = posterior["success_probs"].values.reshape(-1, len(MITRE_STAGES))
    R = len(lambda_draws)
    print(f"Posterior samples: {R}")

    # ---------- 8.2 Parallel per-draw Markov simulation of per-attempt success ----------
    args_list = [(success_probs_draws[i, :], N_SIM_PER_DRAW, 1000 + i) for i in range(R)]
    p_success_simulated = np.zeros(R, dtype=float)

    print("Running parallel attack chain simulations...")
    with ProcessPoolExecutor(max_workers=(N_WORKERS or multiprocessing.cpu_count())) as exe:
        for i, res in enumerate(exe.map(worker_function, args_list)):
            p_success_simulated[i] = res
    print("Markov-chain simulation complete.")

    # ---------- 8.3 Posterior predictive: compound loss by FAIR categories ----------
    rng_np = np.random.default_rng(RANDOM_SEED + 1)
    annual_losses = np.zeros(R, dtype=float)
    incident_counts = np.zeros(R, dtype=int)
    cat_loss_matrix = np.zeros((R, len(loss_categories)), dtype=float)

    for i in range(R):
        # Effective success rate per year (frequency × per-attempt success)
        lam_eff = lambda_draws[i] * p_success_simulated[i]
        n_succ = rng_np.poisson(lam_eff)
        incident_counts[i] = n_succ
        if n_succ == 0:
            continue

        # Incident-level severity multiplier (lognormal noise)
        severities = rng_np.lognormal(mean=0.0, sigma=0.6, size=n_succ)

        # Primary categories: always-on bodies (lognormal) × severity
        prod = np.sum(rng_np.lognormal(cat_mu[0], cat_sigma[0], size=n_succ) * severities)
        resp = np.sum(rng_np.lognormal(cat_mu[1], cat_sigma[1], size=n_succ) * severities)

        # Secondary categories: zero-inflated + Pareto heavy tails
        # (Here we trigger at most once per year for simplicity; extend to per-incident if needed.)
        reg = 0.0
        if rng_np.random() < 0.20:  # trigger probability for Regulatory/Legal
            xm, alpha = pareto_defaults["RegulatoryLegal"]["xm"], pareto_defaults["RegulatoryLegal"]["alpha"]
            reg = xm * (1.0 + rng_np.pareto(alpha))  # Pareto Type I shifted by xm

        rep = 0.0
        if rng_np.random() < 0.20:  # trigger probability for Reputation/Competitive
            xm, alpha = pareto_defaults["ReputationCompetitive"]["xm"], pareto_defaults["ReputationCompetitive"]["alpha"]
            rep = xm * (1.0 + rng_np.pareto(alpha))

        cat_loss_matrix[i, :] = [prod, resp, reg, rep]
        annual_losses[i] = prod + resp + reg + rep


    # ---------- 8.4 Summaries: AAL and category credible intervals ----------
    mean_AAL = annual_losses.mean()
    median_AAL = np.median(annual_losses)
    p2_5, p97_5 = np.percentile(annual_losses, [2.5, 97.5])
    mean_incidents = incident_counts.mean()
    pct_zero = 100.0 * np.mean(annual_losses == 0.0)

    print("\nAAL posterior predictive summary (with fallback, severity, Pareto tails):")
    print(f"Mean AAL: ${mean_AAL:,.0f}")
    print(f"Median AAL: ${median_AAL:,.0f}")
    print(f"AAL 95% credible interval (annualized total loss): ${p2_5:,.0f} – ${p97_5:,.0f}")
    print(f"Mean successful incidents / year: {mean_incidents:.2f}")
    print(f"% years with zero successful incidents: {pct_zero:.1f}%\n")

    print("Category-level annual loss 95% credible intervals:")
    for c, cat in enumerate(loss_categories):
        low, med, high = np.percentile(cat_loss_matrix[:, c], [2.5, 50, 97.5])
        share_med = 100.0 * med / (median_AAL + 1e-12)
        print(f"  {cat:<24s} ${low:,.0f} – ${high:,.0f}  (median ${med:,.0f}, ≈{share_med:.1f}% of median AAL)")

    # ---------- 8.5 Visualization: original 2×2 grid + separate log-scale tail ----------
    sns.set_style("whitegrid")
    scale = 1e6 if PLOT_IN_MILLIONS else 1.0
    scale_label = "Million USD" if PLOT_IN_MILLIONS else "USD"

    # 2×2 Dashboard visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Posterior λ (attacks/year)
    axes[0, 0].hist(lambda_draws, bins=40, color="steelblue", alpha=0.85)
    axes[0, 0].set_title("Posterior λ (attacks/year)")
    axes[0, 0].set_xlabel("λ")
    axes[0, 0].set_ylabel("Frequency")

    # (2) Per-attempt success probability (simulated)
    axes[0, 1].hist(p_success_simulated, bins=40, color="steelblue", alpha=0.85)
    axes[0, 1].set_title("Per-attempt success probability (simulated)")
    axes[0, 1].set_xlabel("Probability")
    axes[0, 1].set_ylabel("Frequency")
    _annotate_percentiles(axes[0, 1], p_success_simulated, money=False)

    # (3) Successful incidents per year
    max_bins = max(int(incident_counts.max()) + 1, 10)
    axes[1, 0].hist(incident_counts, bins=range(0, max_bins), color="steelblue", alpha=0.85)
    axes[1, 0].set_title("Successful incidents / year (posterior predictive)")
    axes[1, 0].set_xlabel("Count")
    axes[1, 0].set_ylabel("Frequency")
    _annotate_percentiles(axes[1, 0], incident_counts, money=False)

    # (4) Annual Loss — truncated linear view (0.5–99th percentile)
    nonzero = annual_losses[annual_losses > 0]
    if len(nonzero) > 0:
        low_lin, high_lin = np.percentile(nonzero, [0.5, 99.0])
        mask = (annual_losses > low_lin) & (annual_losses < high_lin)
        axes[1, 1].hist(annual_losses[mask] / scale, bins=80, color="steelblue", alpha=0.85)
        axes[1, 1].set_title("Annual Loss (posterior predictive)\nTruncated 0.5–99th percentile")
        axes[1, 1].set_xlabel(f"Annual Loss ({scale_label})")
        axes[1, 1].set_ylabel("Frequency")
        _annotate_percentiles(axes[1, 1], annual_losses[mask], scale=scale, money=True)
    else:
        axes[1, 1].text(0.5, 0.5, "All draws are zero", ha="center", va="center")

    plt.tight_layout()
    plt.show()

    # Separate log-scale plot for heavy-tail view
    if len(nonzero) > 0:
        low_p, high_p = np.percentile(nonzero, [0.5, 99.5])
        filtered = nonzero[(nonzero >= low_p) & (nonzero <= high_p)]
        filtered = filtered if len(filtered) >= 10 else nonzero
        bins = np.logspace(np.log10(filtered.min() / scale), np.log10(filtered.max() / scale), 100)

        plt.figure(figsize=(8, 5))
        plt.hist(filtered / scale, bins=bins, color="tomato", alpha=0.8)
        plt.xscale("log")
        plt.title("Annual Loss (posterior predictive) — log scale (auto-filtered)")
        plt.xlabel(f"Annual Loss ({scale_label}, log scale)")
        plt.ylabel("Frequency")
        _annotate_percentiles(plt.gca(), filtered, scale=scale, money=True)
        plt.grid(True, which="both", ls="--", alpha=0.35)
        plt.tight_layout()
        plt.show()

    # ============================================================
    # CSV EXPORTS
    # ============================================================
    import os, datetime, pandas as pd

    # Export directory (same as script folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --- Detailed results export (all posterior draws) ---
    outname = os.path.join(script_dir, f"cyber_risk_simulation_results_{timestamp}.csv")

    df = pd.DataFrame({
        "lambda_draw": lambda_draws,
        "p_success": p_success_simulated,
        "incidents": incident_counts,
        "annual_loss": annual_losses,
        **{cat: cat_loss_matrix[:, i] for i, cat in enumerate(loss_categories)}
    })
    df.to_csv(outname, index=False)
    print(f"\n✅ Detailed results exported to {outname}")

    # --- Summary statistics export (with % share of median AAL) ---
    summary_data = []

    # Overall AAL stats
    summary_data.append({
        "Category": "Total Annual Loss",
        "Mean": np.mean(annual_losses),
        "Median": np.median(annual_losses),
        "CI_2.5%": np.percentile(annual_losses, 2.5),
        "CI_97.5%": np.percentile(annual_losses, 97.5),
        "Median Share of AAL (%)": 100.0
    })

    # Category stats with share of median AAL
    median_AAL = np.median(annual_losses)
    for i, cat in enumerate(loss_categories):
        vals = cat_loss_matrix[:, i]
        med_val = np.median(vals)
        mean_val = np.mean(vals)
        ci_low, ci_high = np.percentile(vals, [2.5, 97.5])
        share_med = 100.0 * med_val / (median_AAL + 1e-12)
        summary_data.append({
            "Category": cat,
            "Mean": mean_val,
            "Median": med_val,
            "CI_2.5%": ci_low,
            "CI_97.5%": ci_high,
            "Median Share of AAL (%)": share_med
        })

    summary_df = pd.DataFrame(summary_data)
    summary_name = os.path.join(script_dir, f"cyber_risk_simulation_summary_{timestamp}.csv")
    summary_df.to_csv(summary_name, index=False)
    print(f"✅ Summary statistics exported to {summary_name}\n")

    # ============================================================
    # Return dictionary for programmatic reuse
    # ============================================================
    return {
        "lambda_draws": lambda_draws,
        "p_success_simulated": p_success_simulated,
        "incident_counts": incident_counts,
        "annual_losses": annual_losses,
        "cat_loss_matrix": cat_loss_matrix,
    }



# =============================================================================
# 9) ENTRY POINT — Purpose: safe multiprocessing on all OSes
# =============================================================================
if __name__ == "__main__":
    results = main()
