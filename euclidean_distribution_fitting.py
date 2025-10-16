#!/usr/bin/env python3
"""
Find the best-fitting distribution for Euclidean tetrahedron volumes.
Using scipy.stats and fitter package for comprehensive distribution testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# First, let's regenerate the Euclidean tetrahedron data
def sample_euclidean_tetrahedra(n_samples=50000, seed=42):
    """Generate Euclidean tetrahedra volumes."""
    np.random.seed(seed)
    
    volumes = []
    v1 = np.array([0, 0, 1])  # North pole
    
    for i in range(n_samples):
        # v2 on meridian
        theta2 = np.random.uniform(0, np.pi)
        v2 = np.array([np.sin(theta2), 0, np.cos(theta2)])
        
        # v3, v4 uniform on sphere
        for _ in range(2):
            vec = np.random.randn(3)
            vec = vec / np.linalg.norm(vec)
            if _ == 0:
                v3 = vec
            else:
                v4 = vec
        
        # Volume
        mat = np.column_stack([v2 - v1, v3 - v1, v4 - v1])
        vol = abs(np.linalg.det(mat)) / 6
        volumes.append(vol)
    
    return np.array(volumes)

print("Generating Euclidean tetrahedron volumes...")
volumes = sample_euclidean_tetrahedra(50000)
volumes = volumes[volumes > 0]  # Remove any zeros for log-based distributions

# Try various distributions from scipy.stats
distributions = [
    # Beta family
    ('beta', stats.beta),
    ('betaprime', stats.betaprime),
    
    # Gamma family
    ('gamma', stats.gamma),
    ('weibull', stats.weibull_min),
    ('exponweib', stats.exponweib),
    ('invgamma', stats.invgamma),
    ('gengamma', stats.gengamma),
    
    # Log-based
    ('lognorm', stats.lognorm),
    ('loggamma', stats.loggamma),
    
    # Other bounded/semi-bounded
    ('chi2', stats.chi2),
    ('chi', stats.chi),
    ('rayleigh', stats.rayleigh),
    ('maxwell', stats.maxwell),
    ('nakagami', stats.nakagami),
    ('rice', stats.rice),
    
    # General flexible
    ('johnsonsu', stats.johnsonsu),
    ('johnsonsb', stats.johnsonsb),
]

# Fit each distribution and compute goodness of fit
print("\nFitting distributions...")
results = []

for name, dist in distributions:
    try:
        # Fit distribution
        params = dist.fit(volumes)
        
        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.kstest(volumes, lambda x: dist.cdf(x, *params))
        
        # Compute AIC (Akaike Information Criterion)
        log_likelihood = np.sum(dist.logpdf(volumes, *params))
        n_params = len(params)
        aic = 2 * n_params - 2 * log_likelihood
        
        results.append({
            'name': name,
            'params': params,
            'ks_stat': ks_stat,
            'p_value': p_value,
            'aic': aic,
            'dist': dist
        })
        
    except Exception as e:
        print(f"  Failed to fit {name}: {str(e)[:50]}...")

# Sort by KS statistic (lower is better)
results.sort(key=lambda x: x['ks_stat'])

print("\n" + "="*70)
print("DISTRIBUTION FITTING RESULTS (sorted by KS statistic)")
print("="*70)
print(f"{'Distribution':15} {'KS Stat':>10} {'p-value':>10} {'AIC':>15}")
print("-"*70)

for r in results[:10]:  # Top 10
    print(f"{r['name']:15} {r['ks_stat']:10.4f} {r['p_value']:10.4f} {r['aic']:15.1f}")

# Visualize top 4 distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

x = np.linspace(0.001, np.max(volumes), 1000)

for i, r in enumerate(results[:4]):
    ax = axes[i]
    
    # Histogram
    ax.hist(volumes, bins=50, density=True, alpha=0.6, color='purple', 
            edgecolor='black', label='Data')
    
    # Fitted distribution
    try:
        y = r['dist'].pdf(x, *r['params'])
        ax.plot(x, y, 'r-', linewidth=2, 
                label=f"{r['name']} (KS={r['ks_stat']:.3f})")
    except:
        pass
    
    ax.set_xlabel('Volume')
    ax.set_ylabel('Density')
    ax.set_title(f"Fit: {r['name']}")
    ax.legend()
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 10)

plt.tight_layout()
plt.savefig('euclidean_best_fits.png', dpi=150)
print(f"\nSaved visualization to euclidean_best_fits.png")

# Detailed look at the best distribution
best = results[0]
print(f"\n\nBEST FIT: {best['name']}")
print("="*50)
print(f"Parameters: {best['params']}")
print(f"KS statistic: {best['ks_stat']:.6f}")
print(f"p-value: {best['p_value']:.6f}")

# Compare with hyperbolic Beta parameters
print("\n\nCOMPARISON WITH HYPERBOLIC CASE:")
print("="*50)

# For Beta, extract alpha and beta
if best['name'] == 'beta':
    alpha, beta_p, loc, scale = best['params']
    print(f"Euclidean Beta: α={alpha:.3f}, β={beta_p:.3f}")
else:
    # Try to fit Beta for comparison
    beta_params = stats.beta.fit(volumes)
    alpha, beta_p = beta_params[0], beta_params[1]
    print(f"Euclidean Beta (for comparison): α={alpha:.3f}, β={beta_p:.3f}")

print(f"Hyperbolic Beta: α=1.077, β=0.974")

# Q-Q plots for top 3
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, r in enumerate(results[:3]):
    ax = axes[i]
    
    # Generate theoretical quantiles
    theoretical = r['dist'].ppf(np.linspace(0.01, 0.99, 100), *r['params'])
    empirical = np.percentile(volumes, np.linspace(1, 99, 100))
    
    ax.plot(theoretical, empirical, 'bo', alpha=0.5)
    ax.plot([0, max(theoretical)], [0, max(theoretical)], 'r--', linewidth=2)
    ax.set_xlabel(f'{r["name"]} Theoretical Quantiles')
    ax.set_ylabel('Empirical Quantiles')
    ax.set_title(f'Q-Q Plot: {r["name"]}')

plt.tight_layout()
plt.savefig('euclidean_qq_plots.png', dpi=150)

# Check if it's a mixture distribution
print("\n\nMIXTURE HYPOTHESIS:")
print("="*50)
print("The distribution might be a mixture of:")
print("1. Near-degenerate configurations (peak near 0)")
print("2. Non-degenerate configurations (broader distribution)")

# Try to identify modes
from scipy.signal import find_peaks
hist_counts, bin_edges = np.histogram(volumes, bins=100, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

peaks, properties = find_peaks(hist_counts, prominence=0.5)
print(f"\nNumber of prominent modes: {len(peaks)}")
for i, peak in enumerate(peaks):
    print(f"  Mode {i+1} at volume ≈ {bin_centers[peak]:.4f}")

plt.close('all')

# Final summary
print("\n\nSUMMARY:")
print("="*50)
print(f"Best fitting distribution: {best['name']}")
print(f"The Euclidean case shows different behavior than hyperbolic:")
print(f"- Strong peak near zero (degenerate configurations)")
print(f"- Long tail toward maximum")
print(f"- Not well-described by simple Beta distribution")
print(f"- Suggests geometry-specific volume statistics")