#!/usr/bin/env python3
"""
cyber_incident_pymc_parallel_full_fixed.py

Full pipeline (wrapped in main()):
 - User-friendly MITRE stage definition
 - PyMC sampling for lambda and per-stage success
 - Parallel Monte Carlo per-posterior-draw Markov attacker simulator (retries/fallback/detection)
 - Posterior-predictive compound-Poisson AAL simulation with FAIR decomposition
 - Improved plotting: truncated linear view + log-scale tail + percentile annotations
 - Saves CSV of posterior predictive draws.

Requirements:
pip install pymc arviz matplotlib pandas scipy numpy
"""
from __future__ import annotations
import time
import math
import random
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats, optimize
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional

# -----------------------
# USER INPUTS (customize)
# -----------------------
CI_MIN_FREQ = 0.01
CI_MAX_FREQ = 10
Z_90 = 1.645

# Observed/synthetic data (choose one or none)
observed_success_counts = None  # e.g. {'attempts': 100, 'successes': 5}
observed_successes_per_year = None  # e.g. 5

# Parallel Monte Carlo settings
N_WORKERS = None  # None => uses cpu_count()
N_SIM_PER_DRAW = 400    # MC attempts per posterior draw to estimate per-attempt success
SUBSAMPLE_POSTERIOR = None  # None => use all posterior draws; set int to subsample for speed

# Markov-simulator behaviour
MAX_RETRIES_PER_STAGE = 3  # 0–5: 0 = no retry, 3 is a common default for moderate persistence.
RETRY_PENALTY = 0.75       # between 0.6–0.9: smaller means retries quickly lose effectiveness.
FALLBACK_PROB = 0.25       # 0.1–0.5: higher means attackers often step back and try alternate techniques.
DETECT_BASE = 0.01         # small (0.005–0.02) and reflects base detection chance per failed attempt.
DETECT_INC_PER_RETRY = 0.06     # 0.02–0.1 are typical if you want detection to ramp up with repeated failed attempts.

# PyMC sampling config
N_SAMPLES = 1200
N_TUNE = 800
N_CHAINS = 4
TARGET_ACCEPT = 0.9
RANDOM_SEED = 42

# Plot settings
PLOT_IN_MILLIONS = True  # display losses in millions on the histograms

# -----------------------
# User-friendly MITRE stage map: edit here
# -----------------------
STAGE_CONTROL_MAP = {
    "Initial Access":        (0.25, 0.65),
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

# Validate and extract
MITRE_STAGES = list(STAGE_CONTROL_MAP.keys())
stage_control_effectiveness: List[Tuple[float, float]] = []
for s, eff in STAGE_CONTROL_MAP.items():
    if not (isinstance(eff, (list, tuple)) and len(eff) == 2):
        raise ValueError(f"Stage '{s}' must have a (min,max) tuple for control effectiveness.")
    low, high = eff
    if not (0 <= low < high <= 1):
        raise ValueError(f"Stage '{s}' has invalid effectiveness bounds ({low}, {high}). Must be within [0,1].")
    stage_control_effectiveness.append((low, high))
NUM_STAGES = len(MITRE_STAGES)

# -----------------------
# Convert control-effectiveness -> attacker success 90% CI (5% & 95%)
# -----------------------
stage_success_ranges = [(1.0 - max_eff, 1.0 - min_eff) for (min_eff, max_eff) in stage_control_effectiveness]

def quantile_match_beta(p5: float, p95: float, q_low: float = 0.05, q_high: float = 0.95):
    mean = 0.5 * (p5 + p95)
    range_width = max(p95 - p5, 1e-6)
    concentration_guess = 20.0 * (0.1 / range_width)
    a0 = max(1e-3, mean * concentration_guess)
    b0 = max(1e-3, (1.0 - mean) * concentration_guess)
    def residuals(params):
        a, b = params
        if a <= 0 or b <= 0:
            return [1e6, 1e6]
        return [
            stats.beta.ppf(q_low, a, b) - p5,
            stats.beta.ppf(q_high, a, b) - p95
        ]
    sol = optimize.root(residuals, [a0, b0], method='hybr')
    if sol.success and np.all(sol.x > 0):
        return float(sol.x[0]), float(sol.x[1])
    return float(a0), float(b0)

alphas, betas = zip(*(quantile_match_beta(lo, hi) for lo, hi in stage_success_ranges))
alphas = np.array(alphas)
betas = np.array(betas)

# -----------------------
# FAIR categories & per-category Q5/Q95 (edit as needed)
# -----------------------
loss_categories = [
    "Productivity",
    "ResponseContainment",
    "RegulatoryLegal",
    "ReputationCompetitive"
]
NCAT = len(loss_categories)
loss_q5_q95 = {
    "Productivity": (10_000, 150_000),
    "ResponseContainment": (20_000, 500_000),
    "RegulatoryLegal": (0, 2_000_000),
    "ReputationCompetitive": (0, 5_000_000)
}

def lognormal_from_q5_q95(q5: float, q95: float):
    q5 = max(q5, 1.0)
    q95 = max(q95, q5 * 1.0001)
    ln_q5, ln_q95 = np.log(q5), np.log(q95)
    sigma = (ln_q95 - ln_q5) / (2.0 * Z_90)
    mu = 0.5 * (ln_q5 + ln_q95)
    return mu, sigma

cat_mu = np.zeros(NCAT)
cat_sigma = np.zeros(NCAT)
for i, cat in enumerate(loss_categories):
    q5, q95 = loss_q5_q95.get(cat, (1.0, 1.0))
    mu, sigma = lognormal_from_q5_q95(q5, q95)
    cat_mu[i] = mu
    cat_sigma[i] = sigma

# Pareto defaults for secondaries
pareto_defaults = {
    "RegulatoryLegal": {"xm": 100_000.0, "alpha": 1.8},
    "ReputationCompetitive": {"xm": 250_000.0, "alpha": 1.5}
}
IDX_REG = loss_categories.index("RegulatoryLegal")
IDX_REP = loss_categories.index("ReputationCompetitive")

# -----------------------
# Pareto sampler helper
# -----------------------
def draw_pareto(xm: float, alpha: float, size: int = 1, rng: Optional[np.random.Generator] = None):
    if rng is None:
        rng = np.random
    return xm * (1.0 + rng.pareto(alpha, size=size))

# -----------------------
# Markov-chain single-attempt simulator (top-level so picklable)
# -----------------------
def simulate_one_attempt(success_probs_stage: np.ndarray,
                         rng: random.Random,
                         max_retries_per_stage: int = 2,
                         retry_penalty: float = 0.8,
                         fallback_prob: float = 0.3,
                         detect_base: float = 0.01,
                         detect_increase_per_retry: float = 0.05) -> bool:
    """
    Simulate a single attacker attempt through ordered stages using python.random.Random rng.
    Return True if attacker completes all stages (success), False if detected or gives up.
    """
    num_stages = len(success_probs_stage)
    stage_idx = 0

    if np.isscalar(max_retries_per_stage):
        max_retries = [int(max_retries_per_stage)] * num_stages
    else:
        max_retries = [int(x) for x in max_retries_per_stage]

    while True:
        if stage_idx >= num_stages:
            return True

        p_base = float(success_probs_stage[stage_idx])
        retries_allowed = max_retries[stage_idx]
        retry_count = 0
        succeeded_this_stage = False

        while retry_count <= retries_allowed:
            p_try = (retry_penalty ** retry_count) * p_base
            u = rng.random()
            if u < p_try:
                succeeded_this_stage = True
                break
            # failed attempt => detection check
            detect_prob = detect_base + detect_increase_per_retry * retry_count
            detect_prob = min(max(detect_prob, 0.0), 1.0)
            if rng.random() < detect_prob:
                return False
            retry_count += 1

        if succeeded_this_stage:
            stage_idx += 1
            continue
        else:
            if stage_idx == 0:
                return False
            if rng.random() < fallback_prob:
                stage_idx = max(0, stage_idx - 1)
                continue
            else:
                return False

# -----------------------
# Monte Carlo estimator for per-draw per-attempt success probability
# (picklable)
# -----------------------
def estimate_per_attempt_success(success_probs_stage: np.ndarray,
                                 n_sim: int = 500,
                                 max_retries_per_stage: int = 2,
                                 retry_penalty: float = 0.8,
                                 fallback_prob: float = 0.3,
                                 detect_base: float = 0.01,
                                 detect_increase_per_retry: float = 0.05,
                                 seed: Optional[int] = None) -> float:
    if seed is None:
        seed = int(time.time() * 1000) % (2**31 - 1)
    rng = random.Random(seed)
    success_count = 0
    for _ in range(int(n_sim)):
        ok = simulate_one_attempt(success_probs_stage,
                                  rng=rng,
                                  max_retries_per_stage=max_retries_per_stage,
                                  retry_penalty=retry_penalty,
                                  fallback_prob=fallback_prob,
                                  detect_base=detect_base,
                                  detect_increase_per_retry=detect_increase_per_retry)
        if ok:
            success_count += 1
    return success_count / float(n_sim)

# Worker wrapper to submit to ProcessPoolExecutor
def worker_arg_tuple(per_stage: np.ndarray, seed: int, n_sim: int) -> Tuple:
    return (per_stage, n_sim, MAX_RETRIES_PER_STAGE, RETRY_PENALTY, FALLBACK_PROB, DETECT_BASE, DETECT_INC_PER_RETRY, seed)

def worker_function(args: Tuple) -> float:
    # args: (per_stage, n_sim, max_retries, retry_penalty, fallback_prob, detect_base, detect_inc, seed)
    per_stage, n_sim, max_retries, retry_penalty, fallback_prob, detect_base, detect_inc, seed = args
    return estimate_per_attempt_success(per_stage,
                                        n_sim=n_sim,
                                        max_retries_per_stage=max_retries,
                                        retry_penalty=retry_penalty,
                                        fallback_prob=fallback_prob,
                                        detect_base=detect_base,
                                        detect_increase_per_retry=detect_inc,
                                        seed=int(seed))

# -----------------------
# Main execution
# -----------------------
def main():
    print(f"Loaded {NUM_STAGES} MITRE stages:")
    for i, (name, (lo, hi)) in enumerate(zip(MITRE_STAGES, stage_control_effectiveness), 1):
        print(f"  {i:2d}. {name:<22s}  control effectiveness: {lo:.2f}–{hi:.2f}")

    print("\nBuilding PyMC model and sampling posterior (may take a few minutes)...")
    with pm.Model() as model:
        mu_lambda = np.log(np.sqrt(CI_MIN_FREQ * CI_MAX_FREQ))
        sigma_lambda = (np.log(CI_MAX_FREQ) - np.log(CI_MIN_FREQ)) / (2.0 * Z_90)
        lambda_rate = pm.Lognormal('lambda_rate', mu=mu_lambda, sigma=sigma_lambda)

        success_probs = pm.Beta('success_probs', alpha=alphas, beta=betas, shape=NUM_STAGES)

        if observed_success_counts is not None:
            attempts = int(observed_success_counts['attempts'])
            successes = int(observed_success_counts['successes'])
            pm.Binomial('obs_stage_aggregate', n=attempts, p=pm.math.prod(success_probs), observed=successes)

        if observed_successes_per_year is not None:
            overall_det = pm.Deterministic('overall_success_prob_det', pm.math.prod(success_probs))
            expected_det = pm.Deterministic('expected_successes_det', lambda_rate * overall_det)
            pm.Poisson('obs_successes_year', mu=expected_det, observed=int(observed_successes_per_year))

        trace = pm.sample(draws=N_SAMPLES, tune=N_TUNE, chains=N_CHAINS,
                          target_accept=TARGET_ACCEPT, random_seed=RANDOM_SEED,
                          return_inferencedata=True)

    print("Sampling finished.")

    posterior = trace.posterior
    nchains = posterior.sizes['chain']
    ndraws = posterior.sizes['draw']
    R_total = nchains * ndraws
    print(f"Posterior draws: chains={nchains}, draws={ndraws}, total={R_total}")

    lambda_draws = posterior['lambda_rate'].values.reshape(-1)
    success_probs_draws = posterior['success_probs'].values.reshape(-1, NUM_STAGES)

    # optional subsample
    if SUBSAMPLE_POSTERIOR is not None and 0 < SUBSAMPLE_POSTERIOR < len(lambda_draws):
        rng_idx = np.random.default_rng(RANDOM_SEED)
        idxs = rng_idx.choice(len(lambda_draws), size=SUBSAMPLE_POSTERIOR, replace=False)
        lambda_draws = lambda_draws[idxs]
        success_probs_draws = success_probs_draws[idxs, :]
        print(f"Subsampled posterior draws to {len(lambda_draws)} for faster MC.")

    R = len(lambda_draws)
    print(f"Running parallel Monte Carlo: {R} posterior draws × {N_SIM_PER_DRAW} sims/draw ...")

    # Build worker args list
    worker_args = []
    for i in range(R):
        seed = 100000 + i
        per_stage = success_probs_draws[i, :].astype(float)
        # Per worker tuple ordering matches worker_function unpacking
        worker_args.append((per_stage, N_SIM_PER_DRAW, MAX_RETRIES_PER_STAGE, RETRY_PENALTY, FALLBACK_PROB, DETECT_BASE, DETECT_INC_PER_RETRY, seed))

    # Run parallel jobs
    p_success_simulated = np.zeros(R, dtype=float)
    import multiprocessing
    n_workers = N_WORKERS or multiprocessing.cpu_count()
    print(f"Using up to {n_workers} worker processes.")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {exe.submit(worker_function, arg): idx for idx, arg in enumerate(worker_args)}
        completed = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                print(f"Worker for draw {idx} failed: {e}")
                result = 0.0
            p_success_simulated[idx] = float(result)
            completed += 1
            if completed % max(1, R // 10) == 0:
                print(f"  Completed {completed}/{R} worker results...")
    t1 = time.time()
    print(f"Parallel MC done in {t1 - t0:.1f} s.")

    # -----------------------
    # Posterior predictive compound-Poisson AAL simulation
    # -----------------------
    rng_np = np.random.default_rng(RANDOM_SEED + 1)
    annual_losses = np.zeros(R)
    incident_counts = np.zeros(R, dtype=int)
    cat_loss_matrix = np.zeros((R, NCAT))

    for i in range(R):
        lam_eff = lambda_draws[i] * p_success_simulated[i]
        n_succ = rng_np.poisson(lam_eff)
        incident_counts[i] = n_succ
        if n_succ == 0:
            annual_losses[i] = 0.0
            continue

        # severity: simple population-level defaults (no severity hyperparameters in this run)
        sev_mu = 0.0
        sev_sigma = 0.6
        severities = rng_np.lognormal(mean=sev_mu, sigma=sev_sigma, size=n_succ)

        total_by_cat = np.zeros(NCAT)

        # Primary categories: always present (lognormal body * severity)
        for c_idx, cat in enumerate(loss_categories):
            if cat in ("Productivity", "ResponseContainment"):
                body_draws = rng_np.lognormal(mean=cat_mu[c_idx], sigma=cat_sigma[c_idx], size=n_succ)
                total_by_cat[c_idx] = np.sum(body_draws * severities)

        # Secondary: Regulatory
        p_trig_reg = 0.2
        p_tail_reg = 0.1
        alpha_reg = pareto_defaults["RegulatoryLegal"]["alpha"]
        xm_reg = pareto_defaults["RegulatoryLegal"]["xm"]
        triggers_r = rng_np.random(n_succ) < p_trig_reg
        if triggers_r.any():
            idxs = np.nonzero(triggers_r)[0]
            tail_flags = rng_np.random(len(idxs)) < p_tail_reg
            if (~tail_flags).any():
                body_count = (~tail_flags).sum()
                body_samples = rng_np.lognormal(mean=cat_mu[IDX_REG], sigma=cat_sigma[IDX_REG], size=body_count)
                total_by_cat[IDX_REG] += np.sum(body_samples * severities[idxs[~tail_flags]])
            if tail_flags.any():
                tail_count = tail_flags.sum()
                tail_samples = draw_pareto(xm_reg, alpha_reg, size=tail_count, rng=rng_np)
                total_by_cat[IDX_REG] += np.sum(tail_samples * severities[idxs[tail_flags]])

        # Secondary: Reputation
        p_trig_rep = 0.2
        p_tail_rep = 0.1
        alpha_rep = pareto_defaults["ReputationCompetitive"]["alpha"]
        xm_rep = pareto_defaults["ReputationCompetitive"]["xm"]
        triggers_rep = rng_np.random(n_succ) < p_trig_rep
        if triggers_rep.any():
            idxs = np.nonzero(triggers_rep)[0]
            tail_flags = rng_np.random(len(idxs)) < p_tail_rep
            if (~tail_flags).any():
                body_count = (~tail_flags).sum()
                body_samples = rng_np.lognormal(mean=cat_mu[IDX_REP], sigma=cat_sigma[IDX_REP], size=body_count)
                total_by_cat[IDX_REP] += np.sum(body_samples * severities[idxs[~tail_flags]])
            if tail_flags.any():
                tail_count = tail_flags.sum()
                tail_samples = draw_pareto(xm_rep, alpha_rep, size=tail_count, rng=rng_np)
                total_by_cat[IDX_REP] += np.sum(tail_samples * severities[idxs[tail_flags]])

        cat_loss_matrix[i, :] = total_by_cat
        annual_losses[i] = total_by_cat.sum()

    # -----------------------
    # Summaries
    # -----------------------
    mean_AAL = annual_losses.mean()
    median_AAL = np.median(annual_losses)
    p2_5, p97_5 = np.percentile(annual_losses, [2.5, 97.5])
    mean_incidents = incident_counts.mean()
    pct_zero = 100.0 * np.mean(annual_losses == 0.0)

    print("\nAAL posterior predictive summary (with fallback, severity, Pareto tails):")
    print(f"Mean AAL: ${mean_AAL:,.0f}")
    print(f"Median AAL: ${median_AAL:,.0f}")
    # print(f"95% CI: [${p2_5:,.0f}, ${p97_5:,.0f}]")
    print(f"AAL 95% credible interval (annualized total loss): ${p2_5:,.0f} – ${p97_5:,.0f}")
    print(f"Mean successful incidents / year: {mean_incidents:.2f}")
    print(f"% years with zero successful incidents: {pct_zero:.1f}%\n")

    print("Category-level annual loss 95% credible intervals:")
    for c, cat in enumerate(loss_categories):
        low, med, high = np.percentile(cat_loss_matrix[:, c], [2.5, 50, 97.5])
        share_med = 100.0 * med / (median_AAL + 1e-12)
        print(f"  {cat:<24s} ${low:,.0f} – ${high:,.0f}  (median ${med:,.0f}, ≈{share_med:.1f}% of median AAL)")

    # -----------------------
    # Save CSV
    # -----------------------
    df = pd.DataFrame({
        "lambda_draw": lambda_draws,
        "p_success_sim": p_success_simulated,
        "incident_count": incident_counts,
        "annual_loss": annual_losses
    })
    for idx, cat in enumerate(loss_categories):
        df[f"loss_{cat}"] = cat_loss_matrix[:, idx]

    csv_name = "posterior_predictive_aal_parallel_full_fixed.csv"
    df.to_csv(csv_name, index=False)
    print(f"\nSaved posterior predictive draws to '{csv_name}'")

    # -----------------------
    # Improved plotting with percentile annotations
    # -----------------------
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except Exception:
        pass

    def annotate_percentiles(ax, data, percentiles=(50, 90, 95, 99), label_y=0.9, scale=1.0, fmt="${:,.0f}"):
        ys = ax.get_ylim()
        y_annot = ys[1] * label_y
        for p in percentiles:
            val = np.percentile(data, p) / scale
            ax.axvline(val, color='k', linestyle='--', linewidth=0.9, alpha=0.7)
            ax.text(val, y_annot, f"P{p}={fmt.format(val)}", rotation=90, va='top', ha='center', fontsize=8, backgroundcolor='white')

    # Prepare figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Posterior λ
    axes[0,0].hist(lambda_draws, bins=40, color='steelblue', alpha=0.85)
    axes[0,0].set_title('Posterior λ (attacks/year)')
    axes[0,0].set_xlabel('λ')
    axes[0,0].set_ylabel('Frequency')

    # 2. Per-attempt success probability (simulated)
    axes[0,1].hist(p_success_simulated, bins=40, color='steelblue', alpha=0.85)
    axes[0,1].set_title('Per-attempt success probability (simulated)')
    axes[0,1].set_xlabel('Probability')
    axes[0,1].set_ylabel('Frequency')
    annotate_percentiles(axes[0,1], p_success_simulated, percentiles=(50,90,95,99), label_y=0.9, scale=1.0, fmt="{:,.3f}")

    # 3. Successful incidents per year
    axes[1,0].hist(incident_counts, bins=range(0, max(incident_counts)+2), color='steelblue', alpha=0.85)
    axes[1,0].set_title('Successful incidents / year (posterior predictive)')
    axes[1,0].set_xlabel('Count')
    axes[1,0].set_ylabel('Frequency')
    annotate_percentiles(axes[1,0], incident_counts, percentiles=(50,90,95,99), label_y=0.9, scale=1.0, fmt="{:,.0f}")

    # 4. Annual Loss: truncated linear view (0.5–99th percentile)
    nonzero = annual_losses[annual_losses > 0]
    if len(nonzero) == 0:
        axes[1,1].text(0.5, 0.5, "All annual losses are zero in draws", ha='center')
    else:
        lowp, highp = np.percentile(nonzero, [0.5, 99.0])
        mask = (annual_losses > lowp) & (annual_losses < highp)
        scale = 1e6 if PLOT_IN_MILLIONS else 1.0
        axes[1,1].hist(annual_losses[mask] / scale, bins=80, color='steelblue', alpha=0.85)
        axes[1,1].set_title('Annual Loss (posterior predictive)\nTruncated 0.5–99th percentile')
        axes[1,1].set_xlabel('Annual Loss (Million USD)' if PLOT_IN_MILLIONS else 'Annual Loss (USD)')
        axes[1,1].set_ylabel('Frequency')
        annotate_percentiles(axes[1,1], annual_losses[mask], percentiles=(50,90,95,99), label_y=0.9, scale=scale, fmt="${:,.0f}")

    plt.tight_layout()
    plt.show()

    # -----------------------
    # Improved log-scale plot (auto-filter & log-space bins)
    # -----------------------
    if len(annual_losses[annual_losses > 0]) > 0:
        nonzero = annual_losses[annual_losses > 0]
        scale = 1e6 if PLOT_IN_MILLIONS else 1.0

        # Detect dominance of small values
        frac_small = np.mean(nonzero < 1e4)
        # Compute robust range (trim 0.5–99.5 percentiles)
        low_p, high_p = np.percentile(nonzero, [0.5, 99.5])

        # Filter only for visualization
        filtered = nonzero[(nonzero >= low_p) & (nonzero <= high_p)]
        if len(filtered) < 10:
            filtered = nonzero  # fallback: use all if filter removed too many

        # Build log-spaced bins between filtered range
        bins = np.logspace(np.log10(filtered.min()/scale),
                        np.log10(filtered.max()/scale), 100)

        plt.figure(figsize=(8, 5))
        plt.hist(filtered / scale, bins=bins, color='tomato', alpha=0.8)
        plt.xscale('log')
        plt.title('Annual Loss (posterior predictive) — log scale (auto-filtered)')
        plt.xlabel('Annual Loss (Million USD, log scale)' if PLOT_IN_MILLIONS else 'Annual Loss (USD, log scale)')
        plt.ylabel('Frequency')
        plt.grid(True, which='both', ls='--', alpha=0.4)

        # Annotate percentiles
        def annotate_percentiles(ax, data, percentiles=(50, 90, 95, 99), scale=1.0):
            for p in percentiles:
                val = np.percentile(data, p) / scale
                ax.axvline(val, color="k", linestyle="--", lw=0.8)
                ax.text(val, ax.get_ylim()[1]*0.9, f"P{p}=${val:,.1f}", rotation=90,
                        ha="center", va="top", fontsize=8, backgroundcolor="white")
        annotate_percentiles(plt.gca(), filtered, percentiles=(50,90,95,99), scale=scale)

        # Optional warning if dominated by near-zero values
        if frac_small > 0.3:
            plt.text(0.02, 0.9*plt.gca().get_ylim()[1],
                    "⚠ Many near-zero values hidden for clarity",
                    fontsize=9, color="gray")

        plt.tight_layout()
        plt.show()
    else:
        print("No nonzero annual losses to plot on log scale.")

if __name__ == "__main__":
    main()
