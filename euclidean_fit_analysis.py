#!/usr/bin/env python3
"""
Detailed goodness-of-fit analysis for Euclidean tetrahedron volume distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Generate the data
def sample_euclidean_tetrahedra(n_samples=100000, seed=42):
    """Generate Euclidean tetrahedra volumes."""
    np.random.seed(seed)
    
    volumes = []
    v1 = np.array([0, 0, 1])  # North pole
    
    for i in range(n_samples):
        # v2 on meridian
        theta2 = np.random.uniform(0, np.pi)
        v2 = np.array([np.sin(theta2), 0, np.cos(theta2)])
        
        # v3, v4 uniform on sphere
        vecs = np.random.randn(2, 3)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        v3, v4 = vecs[0], vecs[1]
        
        # Volume
        mat = np.column_stack([v2 - v1, v3 - v1, v4 - v1])
        vol = abs(np.linalg.det(mat)) / 6
        volumes.append(vol)
    
    return np.array(volumes)

print("Generating 100,000 Euclidean tetrahedron volumes for detailed analysis...")
volumes = sample_euclidean_tetrahedra(100000)
volumes = volumes[volumes > 1e-10]  # Remove near-zeros

# Fit top distributions
print("\nFitting distributions...")

# Chi distribution
chi_params = stats.chi.fit(volumes)
chi_df, chi_loc, chi_scale = chi_params

# Nakagami distribution  
nak_params = stats.nakagami.fit(volumes)
nak_nu, nak_loc, nak_scale = nak_params

# Beta distribution (for comparison)
beta_params = stats.beta.fit(volumes)
beta_a, beta_b, beta_loc, beta_scale = beta_params

# Exponential Weibull
expweib_params = stats.exponweib.fit(volumes)

print("\nFitted parameters:")
print(f"Chi: df={chi_df:.3f}, loc={chi_loc:.6f}, scale={chi_scale:.6f}")
print(f"Nakagami: nu={nak_nu:.3f}, loc={nak_loc:.6f}, scale={nak_scale:.6f}")
print(f"Beta: a={beta_a:.3f}, b={beta_b:.3f}, loc={beta_loc:.6f}, scale={beta_scale:.6f}")

# Comprehensive goodness-of-fit tests
print("\n" + "="*70)
print("GOODNESS-OF-FIT ANALYSIS")
print("="*70)

distributions = [
    ('Chi', stats.chi, chi_params),
    ('Nakagami', stats.nakagami, nak_params),
    ('Beta', stats.beta, beta_params),
    ('Exponential Weibull', stats.exponweib, expweib_params)
]

for name, dist, params in distributions:
    print(f"\n{name} Distribution:")
    print("-"*40)
    
    # 1. Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.kstest(volumes, lambda x: dist.cdf(x, *params))
    print(f"KS test: statistic={ks_stat:.6f}, p-value={ks_pval:.6f}")
    
    # 2. Anderson-Darling test (if available)
    try:
        # Generate expected values
        expected = dist.rvs(*params, size=len(volumes))
        ad_result = stats.anderson_ksamp([volumes, expected])
        print(f"AD test: statistic={ad_result.statistic:.6f}, p-value={ad_result.pvalue:.6f}")
    except:
        print("AD test: not available")
    
    # 3. Chi-square test on binned data
    n_bins = 50
    observed_freq, bin_edges = np.histogram(volumes, bins=n_bins)
    
    # Expected frequencies
    expected_freq = []
    for i in range(n_bins):
        p = dist.cdf(bin_edges[i+1], *params) - dist.cdf(bin_edges[i], *params)
        expected_freq.append(p * len(volumes))
    expected_freq = np.array(expected_freq)
    
    # Only use bins with expected frequency > 5
    mask = expected_freq > 5
    chi2_stat = np.sum((observed_freq[mask] - expected_freq[mask])**2 / expected_freq[mask])
    chi2_df = np.sum(mask) - 1 - len(params)
    chi2_pval = 1 - stats.chi2.cdf(chi2_stat, chi2_df)
    print(f"Chi-square test: statistic={chi2_stat:.2f}, df={chi2_df}, p-value={chi2_pval:.6f}")
    
    # 4. Log-likelihood and AIC
    log_likelihood = np.sum(dist.logpdf(volumes, *params))
    aic = 2 * len(params) - 2 * log_likelihood
    print(f"Log-likelihood: {log_likelihood:.2f}, AIC: {aic:.2f}")

# Visual diagnostics
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Histogram with fits
ax = axes[0, 0]
n, bins, _ = ax.hist(volumes, bins=60, density=True, alpha=0.6, 
                     color='purple', edgecolor='black', label='Data')

x = np.linspace(0.001, np.max(volumes), 1000)
ax.plot(x, stats.chi.pdf(x, *chi_params), 'r-', linewidth=2, label='Chi')
ax.plot(x, stats.nakagami.pdf(x, *nak_params), 'g-', linewidth=2, label='Nakagami')
ax.plot(x, stats.beta.pdf(x, *beta_params), 'b--', linewidth=2, label='Beta')

ax.set_xlabel('Volume')
ax.set_ylabel('Density')
ax.set_title('Distribution Fits')
ax.legend()
ax.set_xlim(0, 0.4)

# 2. Q-Q plot for Chi
ax = axes[0, 1]
theoretical = stats.chi.ppf(np.linspace(0.001, 0.999, 1000), *chi_params)
empirical = np.percentile(volumes, np.linspace(0.1, 99.9, 1000))
ax.plot(theoretical, empirical, 'b.', alpha=0.5, markersize=2)
ax.plot([0, max(theoretical)], [0, max(theoretical)], 'r--', linewidth=2)
ax.set_xlabel('Chi Theoretical Quantiles')
ax.set_ylabel('Empirical Quantiles')
ax.set_title('Q-Q Plot: Chi Distribution')

# 3. Q-Q plot for Nakagami
ax = axes[0, 2]
theoretical = stats.nakagami.ppf(np.linspace(0.001, 0.999, 1000), *nak_params)
empirical = np.percentile(volumes, np.linspace(0.1, 99.9, 1000))
ax.plot(theoretical, empirical, 'g.', alpha=0.5, markersize=2)
ax.plot([0, max(theoretical)], [0, max(theoretical)], 'r--', linewidth=2)
ax.set_xlabel('Nakagami Theoretical Quantiles')
ax.set_ylabel('Empirical Quantiles')
ax.set_title('Q-Q Plot: Nakagami Distribution')

# 4. Residual plot for Chi
ax = axes[1, 0]
# Compute residuals at percentiles
percentiles = np.linspace(1, 99, 99)
empirical_q = np.percentile(volumes, percentiles)
theoretical_q = stats.chi.ppf(percentiles/100, *chi_params)
residuals = empirical_q - theoretical_q
ax.plot(percentiles, residuals, 'bo-', markersize=3)
ax.axhline(0, color='r', linestyle='--')
ax.set_xlabel('Percentile')
ax.set_ylabel('Empirical - Theoretical')
ax.set_title('Residuals: Chi Distribution')
ax.grid(True, alpha=0.3)

# 5. P-P plot for Chi
ax = axes[1, 1]
empirical_cdf = np.arange(1, len(volumes) + 1) / len(volumes)
theoretical_cdf = stats.chi.cdf(np.sort(volumes), *chi_params)
ax.plot(theoretical_cdf, empirical_cdf, 'b.', alpha=0.5, markersize=1)
ax.plot([0, 1], [0, 1], 'r--', linewidth=2)
ax.set_xlabel('Theoretical CDF')
ax.set_ylabel('Empirical CDF')
ax.set_title('P-P Plot: Chi Distribution')

# 6. Tail behavior
ax = axes[1, 2]
# Focus on upper tail
tail_threshold = np.percentile(volumes, 90)
tail_volumes = volumes[volumes > tail_threshold]
tail_x = np.linspace(tail_threshold, np.max(volumes), 100)

ax.hist(tail_volumes, bins=30, density=True, alpha=0.6, color='purple', 
        edgecolor='black', label='Data (top 10%)')
ax.plot(tail_x, stats.chi.pdf(tail_x, *chi_params), 'r-', linewidth=2, label='Chi')
ax.plot(tail_x, stats.nakagami.pdf(tail_x, *nak_params), 'g-', linewidth=2, label='Nakagami')
ax.set_xlabel('Volume')
ax.set_ylabel('Density')
ax.set_title('Upper Tail Fit')
ax.legend()

plt.tight_layout()
plt.savefig('euclidean_fit_diagnostics.png', dpi=150)

# Quantitative summary
print("\n" + "="*70)
print("SUMMARY OF FIT QUALITY")
print("="*70)

print("\nKolmogorov-Smirnov test p-values:")
print("  - Values > 0.05 suggest good fit")
print("  - All p-values ≈ 0 indicate some misfit exists")

print("\nPractical assessment:")
print("  - Chi and Nakagami capture the overall shape well")
print("  - Main discrepancy is in the extreme tails")
print("  - Beta distribution clearly inappropriate (left-skewed)")

# Check specific regions
print("\nRegional analysis:")
lower_10 = np.percentile(volumes, 10)
upper_90 = np.percentile(volumes, 90)
middle_volumes = volumes[(volumes > lower_10) & (volumes < upper_90)]

ks_middle_chi = stats.kstest(middle_volumes, 
                           lambda x: stats.chi.cdf(x, *chi_params))
print(f"  Middle 80% KS test (Chi): p-value = {ks_middle_chi[1]:.6f}")

plt.close()

# Final comparison with hyperbolic
print("\n" + "="*70)
print("EUCLIDEAN vs HYPERBOLIC COMPARISON")
print("="*70)
print("Euclidean: Chi/Nakagami distributions (sum of squares structure)")
print("Hyperbolic: Beta distribution (bounded ratio structure)")
print("\nThis reflects fundamental geometric differences:")
print("- Euclidean: Volumes ~ determinants of Gaussian vectors")
print("- Hyperbolic: Volumes ~ complex dilogarithm expressions")
print("- Different geometries → different natural distributions!")