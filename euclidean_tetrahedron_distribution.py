#!/usr/bin/env python3
"""
Distribution of volumes for random Euclidean tetrahedra inscribed in unit sphere.
Configuration: 
- Vertex 1: North pole (0, 0, 1)
- Vertex 2: On Greenwich meridian (x, 0, z) with x² + z² = 1
- Vertices 3, 4: Uniform on sphere
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def sample_sphere_point():
    """Sample uniformly on unit sphere."""
    # Use normal distribution and normalize (Muller's method)
    vec = np.random.randn(3)
    return vec / np.linalg.norm(vec)

def sample_meridian_point():
    """Sample point on Greenwich meridian (y=0 plane) on unit sphere."""
    # Parameterize by angle from north pole
    theta = np.random.uniform(0, np.pi)
    return np.array([np.sin(theta), 0, np.cos(theta)])

def tetrahedron_volume(v1, v2, v3, v4):
    """Compute volume of tetrahedron with vertices v1, v2, v3, v4."""
    # Volume = |det(v2-v1, v3-v1, v4-v1)| / 6
    mat = np.column_stack([v2 - v1, v3 - v1, v4 - v1])
    return abs(np.linalg.det(mat)) / 6

def compute_euclidean_tetrahedron_volumes(n_samples=100000, seed=42):
    """Compute distribution of Euclidean tetrahedron volumes."""
    np.random.seed(seed)
    
    print(f"Computing {n_samples} random Euclidean tetrahedron volumes...")
    print("Configuration:")
    print("  v1: North pole (0, 0, 1)")
    print("  v2: Greenwich meridian (x, 0, z)")
    print("  v3, v4: Uniform on sphere\n")
    
    volumes = []
    
    # Fixed north pole
    v1 = np.array([0, 0, 1])
    
    for i in range(n_samples):
        if i % 10000 == 0 and i > 0:
            print(f"  Progress: {i}/{n_samples} ({i/n_samples*100:.1f}%)")
        
        # Sample other vertices
        v2 = sample_meridian_point()
        v3 = sample_sphere_point()
        v4 = sample_sphere_point()
        
        # Compute volume
        vol = tetrahedron_volume(v1, v2, v3, v4)
        volumes.append(vol)
    
    volumes = np.array(volumes)
    return volumes

# Run the analysis
print("="*70)
print("EUCLIDEAN TETRAHEDRON VOLUME DISTRIBUTION")
print("="*70)
print()

volumes = compute_euclidean_tetrahedron_volumes(100000)

# What's the maximum possible volume?
# For tetrahedron inscribed in unit sphere, max volume is achieved by regular tetrahedron
# Regular tetrahedron inscribed in unit sphere has volume = 8/(9√3) ≈ 0.513
max_theoretical = 8 / (9 * np.sqrt(3))

print(f"\n\nDistribution Statistics:")
print(f"  Count: {len(volumes)}")
print(f"  Mean: {np.mean(volumes):.6f}")
print(f"  Median: {np.median(volumes):.6f}")
print(f"  Std Dev: {np.std(volumes):.6f}")
print(f"  Min: {np.min(volumes):.6f}")
print(f"  Max: {np.max(volumes):.6f}")
print(f"\nTheoretical maximum (regular tetrahedron): {max_theoretical:.6f}")
print(f"Empirical max as % of theoretical: {np.max(volumes)/max_theoretical*100:.1f}%")
print(f"Mean as % of theoretical max: {np.mean(volumes)/max_theoretical*100:.1f}%")

# Fit distributions
print("\n\nFitting distributions...")

# Scale to [0, 1] for beta distribution
scaled_volumes = volumes / np.max(volumes)

# Method of moments for Beta
mean_scaled = np.mean(scaled_volumes)
var_scaled = np.var(scaled_volumes)
if var_scaled < mean_scaled * (1 - mean_scaled):  # Valid Beta parameters
    alpha = mean_scaled * (mean_scaled * (1 - mean_scaled) / var_scaled - 1)
    beta_param = (1 - mean_scaled) * (mean_scaled * (1 - mean_scaled) / var_scaled - 1)
    print(f"\nBeta distribution fit (scaled to [0, {np.max(volumes):.4f}]):")
    print(f"  α = {alpha:.4f}")
    print(f"  β = {beta_param:.4f}")
    print(f"  α + β = {alpha + beta_param:.2f} (concentration)")
    print(f"  α - β = {alpha - beta_param:.4f}")
else:
    print("Beta distribution fitting failed - variance too large")
    alpha, beta_param = None, None

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Histogram with Beta fit
ax = axes[0, 0]
n, bins, _ = ax.hist(volumes, bins=60, density=True, alpha=0.6, 
                     color='purple', edgecolor='black', label='Data')

if alpha is not None:
    # Overlay beta fit
    x = np.linspace(0, np.max(volumes)*1.02, 1000)
    x_scaled = x / np.max(volumes)
    ax.plot(x, stats.beta.pdf(x_scaled, alpha, beta_param) / np.max(volumes), 
            'r-', linewidth=2, label=f'Beta({alpha:.2f}, {beta_param:.2f})')

ax.axvline(max_theoretical, color='green', linestyle='--', linewidth=2, 
           label=f'Theoretical max ({max_theoretical:.3f})')
ax.axvline(np.mean(volumes), color='orange', linestyle='--', linewidth=2,
           label=f'Mean ({np.mean(volumes):.3f})')
ax.set_xlabel('Volume')
ax.set_ylabel('Density')
ax.set_title('Euclidean Tetrahedron Volume Distribution')
ax.legend()

# 2. Comparison with hyperbolic case
ax = axes[0, 1]
# Normalized histograms
ax.hist(volumes/max_theoretical, bins=50, density=True, alpha=0.5, 
        color='purple', label='Euclidean', edgecolor='purple')
ax.axvline(1, color='red', linestyle='--', linewidth=2, label='Maximum')
ax.set_xlabel('Volume / Theoretical Maximum')
ax.set_ylabel('Density')
ax.set_title('Normalized Volume Distribution')
ax.legend()

# Add text showing hyperbolic comparison
hyperbolic_mean_ratio = 0.537  # From our 4-vertex analysis
euclidean_mean_ratio = np.mean(volumes) / max_theoretical
ax.text(0.05, 0.95, f'Mean/Max ratios:\nEuclidean: {euclidean_mean_ratio:.1%}\nHyperbolic: {hyperbolic_mean_ratio:.1%}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. Log-scale histogram
ax = axes[1, 0]
ax.hist(volumes, bins=60, density=True, alpha=0.7, color='darkviolet', log=True)
ax.axvline(max_theoretical, color='green', linestyle='--', linewidth=2)
ax.set_xlabel('Volume')
ax.set_ylabel('Log Density')
ax.set_title('Log-Scale Distribution')

# 4. Q-Q plot if Beta fit succeeded
ax = axes[1, 1]
if alpha is not None:
    theoretical_quantiles = np.linspace(0.01, 0.99, 99)
    empirical_quantiles = np.percentile(volumes, theoretical_quantiles * 100)
    scaled_empirical = empirical_quantiles / np.max(volumes)
    beta_quantiles = stats.beta.ppf(theoretical_quantiles, alpha, beta_param)
    ax.plot(beta_quantiles * np.max(volumes), empirical_quantiles, 'b.', alpha=0.5)
    max_val = np.max(volumes) * 1.05
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    ax.set_xlabel('Beta Distribution Quantiles')
    ax.set_ylabel('Empirical Quantiles')
    ax.set_title('Q-Q Plot vs Beta Distribution')
else:
    ax.text(0.5, 0.5, 'Beta fitting failed\n(variance too large)', 
            ha='center', va='center', transform=ax.transAxes, fontsize=14)

plt.tight_layout()
plt.savefig('euclidean_tetrahedron_distribution.png', dpi=150)
print(f"\nSaved plots to euclidean_tetrahedron_distribution.png")

# Additional analysis
print("\n" + "="*70)
print("COMPARISON WITH HYPERBOLIC CASE:")
print("="*70)
print(f"                    Euclidean   |   Hyperbolic (4 vertices)")
print("-"*70)
print(f"Mean/Max ratio:     {euclidean_mean_ratio:8.1%}   |   53.7%")
if alpha is not None:
    print(f"Beta α:             {alpha:8.2f}   |   1.08")
    print(f"Beta β:             {beta_param:8.2f}   |   0.97")
    print(f"Concentration:      {alpha + beta_param:8.1f}   |   2.05")
    
    # Check goodness of fit
    from scipy.stats import kstest
    ks_stat, p_value = kstest(scaled_volumes, lambda x: stats.beta.cdf(x, alpha, beta_param))
    print(f"\nKolmogorov-Smirnov test for Beta fit:")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    if p_value > 0.05:
        print("  ✓ Beta distribution is a good fit!")
    else:
        print("  ✗ Beta distribution may not be appropriate")

plt.close()