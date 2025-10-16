import numpy as np
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay

def compute_volume_statistics(n_samples, seed=42):
    """Compute volumes and return statistics"""
    np.random.seed(seed)
    volumes = []
    
    for i in range(n_samples):
        # Uniform point on sphere
        vec = np.random.randn(3)
        vec = vec / np.linalg.norm(vec)
        x, y, z = vec
        
        if z > 0.999:  # Skip north pole
            continue
        
        # Stereographic projection
        w = complex(x/(1-z), y/(1-z))
        
        if abs(w) < 0.01 or abs(w-1) < 0.01:
            continue
        
        vertices = np.array([0+0j, 1+0j, w])
        vol = ideal_poly_volume_via_delaunay(vertices, mode='fast', series_terms=96)
        volumes.append(vol)
    
    volumes = np.array(volumes)
    n = len(volumes)
    mean = np.mean(volumes)
    std = np.std(volumes, ddof=1)  # Sample standard deviation
    se_mean = std / np.sqrt(n)  # Standard error of mean
    se_std = std / np.sqrt(2 * (n - 1))  # Standard error of std dev
    
    return {
        'n': n,
        'mean': mean,
        'std': std,
        'se_mean': se_mean,
        'se_std': se_std,
        'volumes': volumes
    }

print("Statistical Precision Analysis for Random Ideal Tetrahedron Volumes")
print("="*70)

# Test with different sample sizes
sample_sizes = [1000, 5000, 10000, 25000, 50000]

results = {}
for n in sample_sizes:
    print(f"\nComputing {n} samples...")
    stats = compute_volume_statistics(n)
    results[n] = stats
    
    print(f"  Actual samples: {stats['n']}")
    print(f"  Mean: {stats['mean']:.6f} ± {stats['se_mean']:.6f}")
    print(f"  Std:  {stats['std']:.6f} ± {stats['se_std']:.6f}")
    print(f"  95% CI for mean: [{stats['mean'] - 1.96*stats['se_mean']:.6f}, "
          f"{stats['mean'] + 1.96*stats['se_mean']:.6f}]")
    
    # How many decimal places are reliable?
    mean_precision = -int(np.floor(np.log10(2 * stats['se_mean'])))
    std_precision = -int(np.floor(np.log10(2 * stats['se_std'])))
    print(f"  Reliable decimal places: mean={mean_precision}, std={std_precision}")

# Convergence analysis
print("\n" + "="*70)
print("Convergence Analysis:")
print("-"*70)

# Check how mean and std converge
means = [results[n]['mean'] for n in sample_sizes]
stds = [results[n]['std'] for n in sample_sizes]

print("\nMean convergence:")
for i, n in enumerate(sample_sizes):
    print(f"  n={n:6d}: {means[i]:.6f}")
    
print("\nStd convergence:")
for i, n in enumerate(sample_sizes):
    print(f"  n={n:6d}: {stds[i]:.6f}")

# Estimate how many samples needed for different precisions
print("\n" + "="*70)
print("Samples needed for different precision levels:")
print("-"*70)

# Use the largest sample to estimate true parameters
best_estimate = results[sample_sizes[-1]]
true_std = best_estimate['std']

# For mean
for decimals in [2, 3, 4, 5]:
    required_se = 0.5 * 10**(-decimals)
    required_n = int(np.ceil((true_std / required_se)**2))
    print(f"  {decimals} decimal places in mean: {required_n:,} samples")

# For standard deviation
print("\nFor standard deviation:")
for decimals in [2, 3, 4]:
    required_se = 0.5 * 10**(-decimals)
    required_n = int(np.ceil(2 * (true_std / (2 * required_se))**2))
    print(f"  {decimals} decimal places in std: {required_n:,} samples")

# Final recommendation
print("\n" + "="*70)
print("SUMMARY:")
print(f"  Standard deviation: ~{true_std:.4f}")
print(f"  This means the distribution is quite broad (CV = {true_std/best_estimate['mean']:.2f})")
print(f"\nFor 4 significant digits:")
print(f"  In mean: need ~{int(np.ceil((true_std / 0.00005)**2)):,} samples")
print(f"  In std: need ~{int(np.ceil(2 * (true_std / 0.0001)**2)):,} samples")
print(f"\nCurrent precision with {sample_sizes[-1]} samples:")
print(f"  Mean: {best_estimate['mean']:.4f} (±{best_estimate['se_mean']:.4f})")
print(f"  Std:  {best_estimate['std']:.4f} (±{best_estimate['se_std']:.4f})")