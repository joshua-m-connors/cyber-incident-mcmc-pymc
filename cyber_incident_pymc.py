# cyber_incident_pymc_fair_full_tail.py
"""
Full FAIR-style AAL model with:
 - Poisson arrival (lambda) for attempts
 - Per-stage Beta priors; overall per-attempt success = product(stage success)
 - Posterior predictive compound-Poisson for annual loss:
    - Per-incident latent severity multiplier -> induces correlation across categories
    - Primary categories: LogNormal magnitude * severity
    - Secondary categories (Regulatory, Reputation): Bernoulli-triggered; when triggered
         draw from LogNormal (body) with prob (1 - tail_prob) OR Pareto (tail) with prob tail_prob
 - Saves posterior predictive draws and category breakdown to CSV.
"""
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats, optimize
import pandas as pd

np.random.seed(42)

# -----------------------
# User priors / setup
# -----------------------
CI_MIN_FREQ = 2
CI_MAX_FREQ = 24
Z_90 = 1.645

# ------------------------------------------------------------
# USER INPUT: MITRE ATT&CK STAGES AND CONTROL EFFECTIVENESS
# ------------------------------------------------------------
# Define each stage as a dictionary entry.
# The keys are the MITRE stage names (in attack flow order),
# and the values are tuples for estimated CONTROL EFFECTIVENESS (min, max).
#
# Control effectiveness means: fraction of attacks stopped by controls at that stage.
# For example, (0.15, 0.50) means controls block 15–50% of attempts at that stage.
#
# You can comment out stages not relevant to your model,
# or adjust the min/max numbers as your organization’s data evolves.
#
# The model automatically:
#   - infers attacker success rates = (1 - control effectiveness)
#   - validates that min < max and values ∈ [0, 1]
# ------------------------------------------------------------

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

# Validate and extract into lists
MITRE_STAGES = list(STAGE_CONTROL_MAP.keys())
stage_control_effectiveness = []

for stage, eff in STAGE_CONTROL_MAP.items():
    if not (isinstance(eff, (list, tuple)) and len(eff) == 2):
        raise ValueError(f"Stage '{stage}' must have a (min,max) tuple for control effectiveness.")
    low, high = eff
    if not (0 <= low < high <= 1):
        raise ValueError(f"Stage '{stage}' has invalid effectiveness bounds ({low}, {high}). Must be within [0,1].")
    stage_control_effectiveness.append((low, high))

NUM_STAGES = len(MITRE_STAGES)
print(f"Loaded {NUM_STAGES} MITRE stages.")
for i, (name, (lo, hi)) in enumerate(zip(MITRE_STAGES, stage_control_effectiveness), 1):
    print(f"  {i:2d}. {name:<22s}  Control effectiveness: {lo:.2f}–{hi:.2f}")


stage_success_ranges = [(1 - max_e, 1 - min_e) for (min_e, max_e) in stage_control_effectiveness]

def quantile_match_beta(p5, p95, q_low=0.05, q_high=0.95):
    mean = 0.5 * (p5 + p95)
    range_width = max(p95 - p5, 1e-6)
    concentration_guess = 20.0 * (0.1 / range_width)
    a0 = max(1e-3, mean * concentration_guess)
    b0 = max(1e-3, (1 - mean) * concentration_guess)
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
# FAIR categories
# -----------------------
loss_categories = [
    "Productivity",         # primary (lognormal body)
    "ResponseContainment",  # primary (lognormal body)
    "RegulatoryLegal",      # secondary (zero-inflated; lognormal body + Pareto tail)
    "ReputationCompetitive"  # secondary (zero-inflated; lognormal body + Pareto tail)
]
NCAT = len(loss_categories)
# category indices
IDX_REG = loss_categories.index("RegulatoryLegal")
IDX_REP = loss_categories.index("ReputationCompetitive")

# Default 90% CI per category (Q5, Q95) - edit these to reflect your beliefs.
loss_q5_q95 = {
    "Productivity": (10_000, 150_000),
    "ResponseContainment": (20_000, 500_000),
    "RegulatoryLegal": (0, 1_000_000),        # Q5=0 means often zero; handled via zero-inflation
    "ReputationCompetitive": (0, 2_000_000)
}

# Helper: lognormal params from Q5/Q95 (clamp q5>0)
def lognormal_from_q5_q95(q5, q95):
    q5 = max(q5, 1.0)
    q95 = max(q95, q5 * 1.0001)
    ln_q5, ln_q95 = np.log(q5), np.log(q95)
    sigma = (ln_q95 - ln_q5) / (2 * Z_90)
    mu = 0.5 * (ln_q5 + ln_q95)
    return mu, sigma

cat_mu = np.zeros(NCAT)
cat_sigma = np.zeros(NCAT)
for i, cat in enumerate(loss_categories):
    q5, q95 = loss_q5_q95.get(cat, (1.0, 1.0))
    mu, sigma = lognormal_from_q5_q95(q5, q95)
    cat_mu[i] = mu
    cat_sigma[i] = sigma

# --- heavy-tail (Pareto) defaults for Regulatory & Reputation
# Pareto: PDF ~ alpha * xm^alpha / x^(alpha+1) for x >= xm
pareto_defaults = {
    "RegulatoryLegal": {"xm": 100_000.0, "alpha": 1.8, "tail_prior_a": 2.0, "tail_prior_b": 2.0},
    "ReputationCompetitive": {"xm": 250_000.0, "alpha": 1.5, "tail_prior_a": 2.0, "tail_prior_b": 2.0}
}
# The above xm is a scale threshold; adjust if you expect tail events to start smaller/larger.

# -----------------------
# Zero-inflation prior for secondaries (learnable)
# Also tail probability prior for secondaries (prob of sampling Pareto vs body)
# -----------------------
# We'll give Beta priors with modest concentration (learn from data if you have it)
# If you prefer to fix these, set observed_* variables below.
# -----------------------

# -----------------------
# Optional observed data (set if you want to condition)
# -----------------------
observed_success_counts = None  # {'attempts': N, 'successes': k}
observed_successes_per_year = 5  # int
# If you have history of category-level incident losses, that would need a separate conditioning block.

# -----------------------
# PyMC model for frequency + per-stage success + zero-inflation/tail priors & severity priors
# Note: We do NOT model per-incident losses inside PyMC (keeps sampling stable). We learn the
# high-level parameters (lambda, success_probs, p_trigger, tail_prob, pareto alpha, severity prior),
# and then run posterior predictive draws outside PyMC to generate compound-Poisson losses.
# -----------------------
with pm.Model() as model:
    # Attack rate prior (lognormal matching 90% CI)
    mu_lambda = np.log(np.sqrt(CI_MIN_FREQ * CI_MAX_FREQ))
    sigma_lambda = (np.log(CI_MAX_FREQ) - np.log(CI_MIN_FREQ)) / (2 * Z_90)
    lambda_rate = pm.Lognormal("lambda_rate", mu=mu_lambda, sigma=sigma_lambda)

    # Per-stage attacker success probabilities
    success_probs = pm.Beta("success_probs", alpha=alphas, beta=betas, shape=NUM_STAGES)
    overall_success_prob = pm.Deterministic("overall_success_prob", pm.math.prod(success_probs))

    expected_successes = pm.Deterministic("expected_successes", lambda_rate * overall_success_prob)

    # Zero-inflation trigger probability for secondaries (Regulatory & Reputation)
    # Use Beta priors: e.g. many years no fine -> expect low p_trigger but allow learning
    p_trigger_reg = pm.Beta("p_trigger_reg", alpha=1.5, beta=6.0)   # prior mean ~0.2
    p_trigger_rep = pm.Beta("p_trigger_rep", alpha=1.0, beta=4.0)   # prior mean ~0.2

    # Tail probability (prob that a triggered secondary event falls in the Pareto tail)
    tail_prob_reg = pm.Beta("tail_prob_reg", alpha=1.0, beta=9.0)   # prior mean ~0.1 (rare tails)
    tail_prob_rep = pm.Beta("tail_prob_rep", alpha=1.0, beta=9.0)

    # Pareto shape (alpha) priors - smaller alpha => heavier tail
    pareto_alpha_reg = pm.Gamma("pareto_alpha_reg", alpha=2.0, beta=1.0)   # mean 2
    pareto_alpha_rep = pm.Gamma("pareto_alpha_rep", alpha=1.8, beta=1.0)   # mean ~1.8

    # Pareto xm (scale) - fixed to defaults here but could be learned with priors if desired.
    pareto_xm_reg = pareto_defaults["RegulatoryLegal"]["xm"]
    pareto_xm_rep = pareto_defaults["ReputationCompetitive"]["xm"]

    # Severity multiplier prior: per-incident latent multiplier (lognormal parameters)
    # We'll learn population-level severity_mu and severity_sigma (in log-space)
    severity_mu = pm.Normal("severity_mu", mu=0.0, sigma=1.0)   # multiplicative factor median = exp(mu)
    severity_sigma = pm.HalfNormal("severity_sigma", sigma=1.0)

    # Optionally condition on observed aggregated success data
    if observed_success_counts is not None:
        attempts = int(observed_success_counts["attempts"])
        successes = int(observed_success_counts["successes"])
        pm.Binomial("obs_stage_aggregate", n=attempts, p=overall_success_prob, observed=successes)

    if observed_successes_per_year is not None:
        pm.Poisson("obs_successes_year", mu=expected_successes, observed=int(observed_successes_per_year))

    # Sample posterior
    trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.92, return_inferencedata=True)

# -----------------------
# Posterior extraction
# -----------------------
posterior = trace.posterior
lambda_draws = posterior["lambda_rate"].values.reshape(-1)
p_overall_draws = posterior["overall_success_prob"].values.reshape(-1)
p_trigger_reg_draws = posterior["p_trigger_reg"].values.reshape(-1)
p_trigger_rep_draws = posterior["p_trigger_rep"].values.reshape(-1)
tail_prob_reg_draws = posterior["tail_prob_reg"].values.reshape(-1)
tail_prob_rep_draws = posterior["tail_prob_rep"].values.reshape(-1)
pareto_alpha_reg_draws = posterior["pareto_alpha_reg"].values.reshape(-1)
pareto_alpha_rep_draws = posterior["pareto_alpha_rep"].values.reshape(-1)
severity_mu_draws = posterior["severity_mu"].values.reshape(-1)
severity_sigma_draws = posterior["severity_sigma"].values.reshape(-1)

R = len(lambda_draws)
print(f"Posterior draws available: {R}")

# -----------------------
# Posterior-predictive compound-Poisson with decomposition, severity, zero-inflation, heavy tail
# For each posterior draw i:
#   lam_eff = lambda_draws[i] * p_overall_draws[i]
#   N_i ~ Poisson(lam_eff)
#   For each incident j=1..N_i:
#       severity_j ~ LogNormal(severity_mu_draws[i], severity_sigma_draws[i])
#       For each category c:
#           - if primary: draw from LogNormal(cat_mu[c], cat_sigma[c]) * severity_j
#           - if secondary (Reg/Rep): Bernoulli(trigger ~ p_trigger_*) -> if triggered:
#                 with prob tail_prob_* -> draw Pareto(xm, alpha) * severity_j
#                 else -> draw LogNormal(cat_mu[c], cat_sigma[c]) * severity_j
# Sum across incidents to get annual totals and per-category totals.
# -----------------------
annual_losses = np.zeros(R)
incident_counts = np.zeros(R, dtype=int)
cat_loss_matrix = np.zeros((R, NCAT))

# Helper: draw from Pareto with parameters xm and alpha: returns values >= xm
# numpy.random.pareto draws values with PDF ~ alpha * (1 + x)^(-alpha-1) for x>=0 (standard Pareto type I)
# To obtain X ~ Pareto(xm, alpha): X = xm * (1 + U) where U ~ pareto(alpha)
def draw_pareto(xm, alpha, size=1):
    # numpy pareto -> returns samples of the "standard" pareto variable Y with pdf alpha*y^(-alpha-1), y>=1
    # Actually numpy.random.pareto returns samples from a distribution with pdf alpha*(1+x)^(-alpha-1), x>=0
    # so standard Pareto (x >= xm): X = xm * (1 + numpy.random.pareto(alpha))
    return xm * (1.0 + np.random.pareto(alpha, size=size))

for i in range(R):
    lam_eff = lambda_draws[i] * p_overall_draws[i]
    # sample number of successful incidents this year
    n_succ = np.random.poisson(lam_eff)
    incident_counts[i] = n_succ
    if n_succ == 0:
        annual_losses[i] = 0.0
        # cat_loss_matrix row stays zeros
        continue

    # sample per-incident severity multipliers
    sev_mu = severity_mu_draws[i]
    sev_sigma = severity_sigma_draws[i]
    # severity_j ~ LogNormal(sev_mu, sev_sigma)
    severities = np.random.lognormal(mean=sev_mu, sigma=sev_sigma, size=n_succ)

    # accumulate loss totals by category for this posterior draw
    total_by_cat = np.zeros(NCAT)

    # Primary categories: Productivity, ResponseContainment -> always present (lognormal body scaled by severity)
    for c_idx, cat in enumerate(loss_categories):
        if cat in ("Productivity", "ResponseContainment"):
            # draw n_succ lognormal magnitudes and scale by severities
            body_draws = np.random.lognormal(mean=cat_mu[c_idx], sigma=cat_sigma[c_idx], size=n_succ)
            total_by_cat[c_idx] = np.sum(body_draws * severities)

    # Secondary categories handling (Regulatory & Reputation)
    # Regulatory
    p_trig_r = p_trigger_reg_draws[i]
    p_tail_r = tail_prob_reg_draws[i]
    alpha_r = max(0.1, pareto_alpha_reg_draws[i])  # avoid degenerate alpha <= 0
    xm_r = pareto_xm_reg
    # Reputation
    p_trig_rep = p_trigger_rep_draws[i]
    p_tail_rep = tail_prob_rep_draws[i]
    alpha_rep = max(0.1, pareto_alpha_rep_draws[i])
    xm_rep = pareto_xm_rep

    # Regulatory: for each incident decide trigger and sample accordingly
    triggers_r = np.random.rand(n_succ) < p_trig_r
    if triggers_r.any():
        # for triggered incidents, decide tail vs body
        tail_flags = np.random.rand(triggers_r.sum()) < p_tail_r
        idxs = np.nonzero(triggers_r)[0]
        # body samples
        if (~tail_flags).any():
            body_count = (~tail_flags).sum()
            body_samples = np.random.lognormal(mean=cat_mu[IDX_REG], sigma=cat_sigma[IDX_REG], size=body_count)
            total_by_cat[IDX_REG] += np.sum(body_samples * severities[idxs[~tail_flags]])
        # tail samples (Pareto)
        if tail_flags.any():
            tail_count = tail_flags.sum()
            tail_samples = draw_pareto(xm_r, alpha_r, size=tail_count)
            total_by_cat[IDX_REG] += np.sum(tail_samples * severities[idxs[tail_flags]])

    # Reputation: similar
    triggers_rep = np.random.rand(n_succ) < p_trig_rep
    if triggers_rep.any():
        tail_flags_rep = np.random.rand(triggers_rep.sum()) < p_tail_rep
        idxs_rep = np.nonzero(triggers_rep)[0]
        if (~tail_flags_rep).any():
            body_count = (~tail_flags_rep).sum()
            body_samples = np.random.lognormal(mean=cat_mu[IDX_REP], sigma=cat_sigma[IDX_REP], size=body_count)
            total_by_cat[IDX_REP] += np.sum(body_samples * severities[idxs_rep[~tail_flags_rep]])
        if tail_flags_rep.any():
            tail_count = tail_flags_rep.sum()
            tail_samples = draw_pareto(xm_rep, alpha_rep, size=tail_count)
            total_by_cat[IDX_REP] += np.sum(tail_samples * severities[idxs_rep[tail_flags_rep]])

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

print("\nAAL posterior predictive summary (with zero-inflation, severity, Pareto tails):")
print(f"Mean AAL: ${mean_AAL:,.0f}")
print(f"Median AAL: ${median_AAL:,.0f}")
print(f"95% CI: [${p2_5:,.0f}, ${p97_5:,.0f}]")
print(f"Mean successful incidents / year: {mean_incidents:.2f}")
print(f"% years with zero successful incidents: {pct_zero:.1f}%\n")

# ------------------------------------------------------------
# Category credible intervals (95% CI instead of means)
# ------------------------------------------------------------
print("Category-level annual loss 95% credible intervals:")
for c, cat in enumerate(loss_categories):
    low, mid, high = np.percentile(cat_loss_matrix[:, c], [2.5, 50, 97.5])
    share_mid = 100.0 * mid / (median_AAL + 1e-12)
    print(
        f"  {cat:<24s} "
        f"${low:,.0f} – ${high:,.0f}  (median ${mid:,.0f}, ≈{share_mid:.1f}% of median AAL)"
    )

# -----------------------
# Plots (basic)
# -----------------------
plt.figure(figsize=(14,10))
plt.subplot(2,2,1)
plt.hist(lambda_draws, bins=40)
plt.title('Posterior λ (attempts/year)')
plt.grid(alpha=0.2)

plt.subplot(2,2,2)
plt.hist(p_overall_draws, bins=40)
plt.title('Posterior overall success probability')
plt.grid(alpha=0.2)

plt.subplot(2,2,3)
plt.hist(incident_counts, bins=range(0, max(incident_counts)+2))
plt.title('Posterior predictive: successful incidents/year')
plt.grid(alpha=0.2)

plt.subplot(2,2,4)
plt.hist(annual_losses, bins=120)
plt.title('Posterior predictive: Annual Loss (USD)')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# Log-tail plot
nonzero = annual_losses[annual_losses > 0]
if len(nonzero) > 0:
    plt.figure(figsize=(6,4))
    plt.hist(np.log10(nonzero), bins=60)
    plt.title('Log10 Annual Loss (years with loss > 0)')
    plt.xlabel('log10(USD)')
    plt.grid(alpha=0.2)
    plt.show()

# Simple error-bar plot for category intervals
lows = np.percentile(cat_loss_matrix, 2.5, axis=0)
highs = np.percentile(cat_loss_matrix, 97.5, axis=0)
meds = np.percentile(cat_loss_matrix, 50, axis=0)

plt.figure(figsize=(7,4))
plt.errorbar(loss_categories, meds, yerr=[meds-lows, highs-meds], fmt='o', capsize=5)
plt.title('Category-level Annual Loss 95% Credible Intervals')
plt.ylabel('USD')
plt.grid(alpha=0.3)
plt.show()

# -----------------------
# Save posterior predictive draws
# -----------------------
df = pd.DataFrame({
    "lambda_draw": lambda_draws,
    "p_overall_draw": p_overall_draws,
    "incident_count": incident_counts,
    "annual_loss": annual_losses
})
for idx, cat in enumerate(loss_categories):
    df[f"loss_{cat}"] = cat_loss_matrix[:, idx]

csv_name = "posterior_predictive_aal_fair_full_tail.csv"
df.to_csv(csv_name, index=False)
print(f"\nSaved posterior predictive draws (with FAIR decomposition) to '{csv_name}'")
