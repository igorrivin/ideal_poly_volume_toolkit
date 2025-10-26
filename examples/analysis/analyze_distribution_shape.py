import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay

# Generate more samples for better statistics
n_samples = 10000
np.random.seed(42)
volumes = []

print(f"Generating {n_samples} samples...")
for i in range(n_samples):
    if i % 1000 == 0:
        print(f"  {i}/{n_samples}")
    
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
regular_vol = 1.01494160640965

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Histogram with multiple overlays
ax = axes[0, 0]
n, bins, _ = ax.hist(volumes, bins=60, density=True, alpha=0.6, 
                     color='blue', edgecolor='black', label='Data')

# Try to fit a beta distribution
a, b, loc, scale = stats.beta.fit(volumes)
x = np.linspace(0, 1.02, 1000)
ax.plot(x, stats.beta.pdf(x, a, b, loc, scale), 'r-', linewidth=2, 
        label=f'Beta fit (α={a:.2f}, β={b:.2f})')

ax.axvline(regular_vol, color='green', linestyle='--', linewidth=2, 
           label='Regular tetrahedron')
ax.set_xlabel('Volume')
ax.set_ylabel('Density')
ax.set_title('Volume Distribution with Beta Fit')
ax.legend()
ax.set_xlim(0, 1.1)

# 2. Log-log plot to check for power law in tail
ax = axes[0, 1]
# Create empirical survival function
sorted_vols = np.sort(volumes)
survival = 1 - np.arange(len(volumes)) / len(volumes)

# Plot only the tail
tail_start = int(0.9 * len(volumes))
ax.loglog(regular_vol - sorted_vols[tail_start:], survival[tail_start:], 
          'b.', markersize=3, alpha=0.5)
ax.set_xlabel('Regular volume - Volume')
ax.set_ylabel('P(V > v)')
ax.set_title('Tail Behavior (log-log)')
ax.grid(True, which='both', alpha=0.3)

# 3. Theoretical vs empirical quantiles
ax = axes[1, 0]
theoretical_quantiles = np.linspace(0.01, 0.99, 99)
empirical_quantiles = np.percentile(volumes, theoretical_quantiles * 100)
beta_quantiles = stats.beta.ppf(theoretical_quantiles, a, b, loc, scale)

ax.plot(beta_quantiles, empirical_quantiles, 'b.', alpha=0.5)
ax.plot([0, 1.1], [0, 1.1], 'r--', linewidth=2)
ax.set_xlabel('Beta Distribution Quantiles')
ax.set_ylabel('Empirical Quantiles')
ax.set_title('Q-Q Plot vs Beta Distribution')
ax.grid(True, alpha=0.3)

# 4. Relationship to distance from regular config
ax = axes[1, 1]
# For each volume, compute how far the fourth vertex is from "regular" position
# Regular tetrahedron has fourth vertex at exp(2πi/3)
regular_w = np.exp(2j * np.pi / 3)

# Recompute to get w values
w_values = []
vol_values = []
for i in range(1000):  # Subset for clarity
    vec = np.random.randn(3)
    vec = vec / np.linalg.norm(vec)
    x, y, z = vec
    if z > 0.999:
        continue
    w = complex(x/(1-z), y/(1-z))
    if abs(w) < 0.01 or abs(w-1) < 0.01:
        continue
    
    w_values.append(w)
    vertices = np.array([0+0j, 1+0j, w])
    vol = ideal_poly_volume_via_delaunay(vertices, mode='fast')
    vol_values.append(vol)

w_values = np.array(w_values)
vol_values = np.array(vol_values)

# Compute "regularity" metric - distance from regular position
# Actually there are 3 regular positions due to symmetry
regular_positions = [np.exp(2j * np.pi * k / 3) for k in range(3)]
distances = []
for w in w_values:
    min_dist = min(abs(w - reg_pos) for reg_pos in regular_positions)
    distances.append(min_dist)

ax.scatter(distances, vol_values, alpha=0.3, s=10)
ax.set_xlabel('Distance from nearest regular position')
ax.set_ylabel('Volume')
ax.set_title('Volume vs Distance from Regularity')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tetrahedron_distribution_analysis.png', dpi=150)

# Print summary statistics
print(f"\nDistribution Analysis Summary:")
print(f"  Total samples: {len(volumes)}")
print(f"  Mean: {np.mean(volumes):.4f} ({np.mean(volumes)/regular_vol*100:.1f}% of regular)")
print(f"  Median: {np.median(volumes):.4f} ({np.median(volumes)/regular_vol*100:.1f}% of regular)")
print(f"  Mode (approx): {bins[np.argmax(n)]:.4f}")
print(f"  Skewness: {stats.skew(volumes):.4f}")
print(f"  Kurtosis: {stats.kurtosis(volumes):.4f}")
print(f"\nBeta distribution parameters:")
print(f"  α = {a:.4f}, β = {b:.4f}")
print(f"  This suggests the distribution is {('right' if a < b else 'left')}-skewed")

plt.close()