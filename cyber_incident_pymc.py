import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# USER INPUTS: 90% Confidence Interval for attack frequency/probability
# ============================================================================
# Frequency: expected number of successful attacks per year
CI_MIN_FREQ = 2      # Minimum frequency - 5th percentile (attacks/year)
CI_MAX_FREQ = 24     # Maximum frequency - 95th percentile (attacks/year)

print("="*70)
print("MITRE ATT&CK-Based Cybersecurity Incident Probability Model")
print("="*70)
print(f"\nUser-defined 90% CI: [{CI_MIN_FREQ}, {CI_MAX_FREQ}] attacks/year")

# Convert frequency to time between attacks (in days)
# Higher frequency = shorter time between attacks
CI_MIN_TIME = 365 / CI_MAX_FREQ  # Min time corresponds to max frequency
CI_MAX_TIME = 365 / CI_MIN_FREQ  # Max time corresponds to min frequency

print(f"Converted to time between attacks: [{CI_MIN_TIME:.2f}, {CI_MAX_TIME:.2f}] days")

# Calculate lognormal parameters from 90% CI
# For lognormal: P5 = exp(mu - 1.645*sigma), P95 = exp(mu + 1.645*sigma)
z_score = 1.645  # For 90% CI (5th and 95th percentiles)

# Solve for mu and sigma
sigma_initial = (np.log(CI_MAX_TIME) - np.log(CI_MIN_TIME)) / (2 * z_score)
mu_initial = (np.log(CI_MAX_TIME) + np.log(CI_MIN_TIME)) / 2

print(f"\nCalculated lognormal parameters from CI:")
print(f"  μ (log-scale mean): {mu_initial:.4f}")
print(f"  σ (log-scale std): {sigma_initial:.4f}")
print(f"  Median time to attack: {np.exp(mu_initial):.2f} days")
print(f"  Median attack frequency: {365/np.exp(mu_initial):.2f} attacks/year")

# ============================================================================
# MITRE ATT&CK Framework Stages
# ============================================================================
MITRE_STAGES = [
    "Initial Access",
    "Execution", 
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command and Control",
    "Exfiltration",
    "Impact"
]

NUM_STAGES = len(MITRE_STAGES)
print(f"\nMITRE ATT&CK Stages (Total: {NUM_STAGES}):")
for i, stage in enumerate(MITRE_STAGES, 1):
    print(f"  {i}. {stage}")

# ============================================================================
# USER INPUTS: Per-Stage Control Effectiveness (90% CI)
# ============================================================================
# Define 90% CI for CONTROL EFFECTIVENESS per stage (0 = no protection, 1 = perfect protection)
# Each tuple is (min_effectiveness, max_effectiveness) at 5th and 95th percentiles
# HIGHER values = STRONGER controls, LOWER values = WEAKER controls

stage_control_effectiveness = [
    (0.15, 0.50),  # 1. Initial Access - moderate defenses (firewall, email filters)
    (0.05, 0.30),  # 2. Execution - weak defenses, attackers often succeed
    (0.10, 0.45),  # 3. Persistence - moderate defenses
    (0.10, 0.35),  # 4. Privilege Escalation - moderate defenses
    (0.20, 0.60),  # 5. Defense Evasion - strong EDR/XDR, but uncertain effectiveness
    (0.15, 0.50),  # 6. Credential Access - moderate defenses (MFA, PAM)
    (0.02, 0.20),  # 7. Discovery - very weak defenses, hard to prevent
    (0.25, 0.65),  # 8. Lateral Movement - strong segmentation and monitoring
    (0.05, 0.30),  # 9. Collection - weak defenses
    (0.10, 0.45),  # 10. Command and Control - moderate defenses (proxy, IDS)
    (0.30, 0.70),  # 11. Exfiltration - strong DLP and egress filtering
    (0.35, 0.75)   # 12. Impact - strong IR, backups, and resilience
]

print("\n" + "="*70)
print("Per-Stage Control Effectiveness (90% CI):")
print("Higher values = Stronger controls | Lower values = Weaker controls")
print("="*70)
for i, (stage, (min_eff, max_eff)) in enumerate(zip(MITRE_STAGES, stage_control_effectiveness), 1):
    mean_eff = (min_eff + max_eff) / 2
    # Calculate corresponding attacker success rate
    min_success = 1 - max_eff
    max_success = 1 - min_eff
    mean_success = 1 - mean_eff
    print(f"{i:2d}. {stage:25s}: Control Eff [{min_eff:.2f}, {max_eff:.2f}] → "
          f"Attacker Success [{min_success:.2f}, {max_success:.2f}]")

# Convert control effectiveness to attacker success rates
stage_success_ranges = []
for min_eff, max_eff in stage_control_effectiveness:
    # Attacker success = 1 - Control effectiveness
    min_success = 1 - max_eff  # Min success when controls are most effective
    max_success = 1 - min_eff  # Max success when controls are least effective
    stage_success_ranges.append((min_success, max_success))

# Convert CI ranges to Beta distribution parameters
def ci_to_beta_params(min_val, max_val):
    """
    Convert 90% CI (5th and 95th percentiles) to Beta distribution parameters.
    Uses method of moments approximation.
    """
    # Use mean and adjust concentration based on range width
    mean = (min_val + max_val) / 2
    
    # Estimate concentration from range width
    # Wider range = lower concentration (more uncertainty)
    range_width = max_val - min_val
    # Base concentration scaled inversely with range width
    concentration = 15 * (0.4 / range_width)
    
    alpha = mean * concentration
    beta = (1 - mean) * concentration
    
    return alpha, beta

alphas = []
betas = []

for min_success, max_success in stage_success_ranges:
    alpha, beta = ci_to_beta_params(min_success, max_success)
    alphas.append(alpha)
    betas.append(beta)

alphas = np.array(alphas)
betas = np.array(betas)

print(f"\nBeta distribution parameters calculated for attacker success rates")

# ============================================================================
# Simulate observed attack progression data
# ============================================================================
np.random.seed(42)

# Simulate 30 observed attack scenarios
observed_attacks = []
for _ in range(30):
    attack = []
    # Each stage has time in days (lognormal distributed)
    for stage in range(NUM_STAGES):
        # Sample from lognormal with slight variation per stage
        stage_mu = mu_initial - np.log(NUM_STAGES) + np.random.normal(0, 0.1)
        stage_sigma = sigma_initial * 0.8
        time_in_stage = np.random.lognormal(stage_mu, stage_sigma)
        attack.append(max(0.1, time_in_stage))  # Ensure positive
    observed_attacks.append(attack)

observed_attacks = np.array(observed_attacks)
total_attack_times = observed_attacks.sum(axis=1)

print(f"\nObserved attack completion times (days):")
print(f"  Mean: {np.mean(total_attack_times):.2f}")
print(f"  Median: {np.median(total_attack_times):.2f}")
print(f"  Std Dev: {np.std(total_attack_times):.2f}")
print(f"  Min: {np.min(total_attack_times):.2f}, Max: {np.max(total_attack_times):.2f}")

# ============================================================================
# Build the PyMC Model with MITRE ATT&CK stages
# ============================================================================
with pm.Model() as attack_progression_model:
    # Global parameters for attack timing distribution
    # Use user-defined CI to inform priors
    mu_global = pm.Normal('mu_global', mu=mu_initial, sigma=0.5)
    sigma_global = pm.HalfNormal('sigma_global', sigma=sigma_initial)
    
    # Stage-specific timing parameters
    # Each stage can have different characteristics
    stage_modifiers = pm.Normal('stage_modifiers', mu=0, sigma=0.3, 
                                shape=NUM_STAGES)
    
    # Probability of successfully progressing through each stage
    # Use stage-specific Beta priors based on user-defined control effectiveness
    success_probs = pm.Beta('success_probs', alpha=alphas, beta=betas, 
                           shape=NUM_STAGES)
    
    # For each observed attack, model the total time
    stage_times = []
    for i in range(NUM_STAGES):
        # Each stage has its own lognormal distribution
        stage_mu = mu_global + stage_modifiers[i] - np.log(NUM_STAGES)
        stage_time = pm.Lognormal(f'stage_{i}_time', 
                                  mu=stage_mu, 
                                  sigma=sigma_global,
                                  shape=len(observed_attacks))
        stage_times.append(stage_time)
    
    # Total attack time is sum of all stages
    total_time = pm.Deterministic('total_attack_time', 
                                  sum(stage_times))
    
    # Likelihood: observed total attack times
    likelihood = pm.Normal('likelihood', 
                          mu=total_time, 
                          sigma=2.0,  # Observation noise
                          observed=total_attack_times)
    
    # Sample using MCMC
    print("\n" + "="*70)
    print("Running MCMC Sampling...")
    print("="*70)
    trace = pm.sample(2000, tune=1000, chains=4, return_inferencedata=True,
                     progressbar=True)

# ============================================================================
# Analyze Results
# ============================================================================
print("\n" + "="*70)
print("MCMC Sampling Results:")
print("="*70)
print(az.summary(trace, var_names=['mu_global', 'sigma_global', 'success_probs']))

# Extract posterior samples - properly handle xarray format
mu_samples = trace.posterior['mu_global'].values.flatten()
sigma_samples = trace.posterior['sigma_global'].values.flatten()

# Fix: Properly extract and flatten success_probs samples
success_prob_samples_flat = trace.posterior['success_probs'].values.reshape(-1, NUM_STAGES)

# Calculate overall attack success probability
# Attack succeeds only if all stages are completed
overall_success = np.prod(success_prob_samples_flat, axis=1)

print("\n" + "="*70)
print("Attack Success Probability (completing all MITRE stages):")
print("="*70)
print(f"Mean probability: {np.mean(overall_success):.4f} ({np.mean(overall_success)*100:.2f}%)")
print(f"95% Credible Interval: [{np.percentile(overall_success, 2.5):.4f}, "
      f"{np.percentile(overall_success, 97.5):.4f}]")

# Show expected successful attacks per year
expected_successes = np.mean(overall_success) * np.mean([CI_MIN_FREQ, CI_MAX_FREQ])
print(f"\nExpected successful attacks per year: {expected_successes:.2f}")
print(f"  (Based on mean attempt frequency of {np.mean([CI_MIN_FREQ, CI_MAX_FREQ]):.1f} attacks/year)")

# Calculate probability of successful attack within specific timeframes
timeframes = [10, 20, 30, 45, 60]
print("\n" + "="*70)
print("Probability of Successful Attack within Timeframe:")
print("="*70)

for threshold_days in timeframes:
    probabilities = []
    for mu_val, sigma_val, success_val in zip(mu_samples, sigma_samples, overall_success):
        # Time probability * Success probability
        time_prob = stats.lognorm.cdf(threshold_days, s=sigma_val, scale=np.exp(mu_val))
        prob = time_prob * success_val
        probabilities.append(prob)
    
    probabilities = np.array(probabilities)
    print(f"{threshold_days:3d} days: {np.mean(probabilities):.4f} "
          f"(95% CI: [{np.percentile(probabilities, 2.5):.4f}, "
          f"{np.percentile(probabilities, 97.5):.4f}])")

# ============================================================================
# Visualization
# ============================================================================
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# Plot 1: Global parameters trace
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(mu_samples, alpha=0.5, linewidth=0.5)
ax1.set_title('Trace: μ_global', fontweight='bold')
ax1.set_xlabel('Sample')
ax1.set_ylabel('μ')
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(sigma_samples, alpha=0.5, linewidth=0.5)
ax2.set_title('Trace: σ_global', fontweight='bold')
ax2.set_xlabel('Sample')
ax2.set_ylabel('σ')
ax2.grid(alpha=0.3)

# Plot 2: Posterior distributions
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(mu_samples, bins=50, alpha=0.7, color='blue', edgecolor='black')
ax3.axvline(np.mean(mu_samples), color='red', linestyle='--', 
           label=f'Mean: {np.mean(mu_samples):.3f}')
ax3.set_title('Posterior: μ_global', fontweight='bold')
ax3.set_xlabel('μ')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 3: Stage success probabilities with prior ranges
ax4 = fig.add_subplot(gs[1, :])

# Calculate statistics using flattened samples
stage_means = np.mean(success_prob_samples_flat, axis=0)
stage_lower = np.percentile(success_prob_samples_flat, 2.5, axis=0)
stage_upper = np.percentile(success_prob_samples_flat, 97.5, axis=0)

x_pos = np.arange(NUM_STAGES)

# Plot prior ranges as shaded regions (attacker success rate)
for i, (min_success, max_success) in enumerate(stage_success_ranges):
    ax4.axhspan(min_success, max_success, xmin=(i-0.4)/NUM_STAGES, xmax=(i+0.4)/NUM_STAGES,
               alpha=0.2, color='gray', zorder=1)

# Plot posterior estimates
bars = ax4.bar(x_pos, stage_means, alpha=0.8, edgecolor='black', zorder=2)
ax4.errorbar(x_pos, stage_means, 
            yerr=[stage_means - stage_lower, stage_upper - stage_means],
            fmt='none', ecolor='red', capsize=5, linewidth=2, zorder=3)

# Color bars based on success rate (red=high success, green=low success)
for i, bar in enumerate(bars):
    bar.set_color(plt.cm.RdYlGn_r(stage_means[i]))

ax4.set_xlabel('MITRE ATT&CK Stage', fontweight='bold', fontsize=11)
ax4.set_ylabel('Attacker Success Probability', fontweight='bold', fontsize=11)
ax4.set_title('Stage-wise Attacker Success Probabilities\n(Gray boxes = prior 90% CI, Bars = posterior mean, Red lines = posterior 95% CI)', 
              fontweight='bold', fontsize=12)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(range(1, NUM_STAGES + 1))
ax4.set_ylim([0, 1])
ax4.grid(axis='y', alpha=0.3, zorder=0)

# Plot 4: Stage names reference with BOTH control effectiveness and attacker success
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')
stage_text = "MITRE ATT&CK Stages:\n"
stage_text += "Control Effectiveness = Higher is better | Attacker Success = Lower is better\n"
stage_text += "-" * 90 + "\n"
for i, stage in enumerate(MITRE_STAGES, 1):
    min_eff, max_eff = stage_control_effectiveness[i-1]
    mean_eff = (min_eff + max_eff) / 2
    mean_success = stage_means[i-1]
    stage_text += f"{i:2d}. {stage:25s} | Control: [{min_eff:.2f}, {max_eff:.2f}] ({mean_eff:.2f}) | "
    stage_text += f"Attack Success: {mean_success:.3f}\n"
ax5.text(0.05, 0.95, stage_text, transform=ax5.transAxes, 
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Plot 5: Overall attack success probability
ax6 = fig.add_subplot(gs[3, 0])
ax6.hist(overall_success, bins=50, alpha=0.7, color='darkred', edgecolor='black')
ax6.axvline(np.mean(overall_success), color='yellow', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(overall_success):.4f}')
ax6.set_title('Overall Attack Success\nProbability', fontweight='bold')
ax6.set_xlabel('Probability')
ax6.set_ylabel('Frequency')
ax6.legend()
ax6.grid(alpha=0.3)

# Plot 6: Attack time distribution
ax7 = fig.add_subplot(gs[3, 1:])
total_time_samples = trace.posterior['total_attack_time'].values.flatten()
ax7.hist(total_time_samples, bins=50, alpha=0.7, color='green', edgecolor='black')
ax7.axvline(np.mean(total_time_samples), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(total_time_samples):.2f} days')
ax7.axvline(CI_MIN_TIME, color='orange', linestyle=':', linewidth=2, label='User 90% CI')
ax7.axvline(CI_MAX_TIME, color='orange', linestyle=':', linewidth=2)
ax7.set_title('Total Attack Completion Time Distribution', fontweight='bold')
ax7.set_xlabel('Time (days)', fontweight='bold')
ax7.set_ylabel('Frequency')
ax7.legend()
ax7.grid(alpha=0.3)

plt.savefig('mitre_attack_analysis.png', dpi=300, bbox_inches='tight')
print("\n" + "="*70)
print("Visualization saved as 'mitre_attack_analysis.png'")
print("="*70)

plt.show()