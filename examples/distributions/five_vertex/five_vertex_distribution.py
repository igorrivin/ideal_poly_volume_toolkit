import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay
import warnings
warnings.filterwarnings('ignore')

def sample_two_points_on_sphere(n_samples, seed=42):
    """Sample two points uniformly on unit sphere"""
    np.random.seed(seed)
    points = []
    
    for i in range(n_samples):
        # Two independent uniform points on sphere
        vecs = np.random.randn(2, 3)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        points.append(vecs)
    
    return np.array(points)

def compute_five_vertex_volumes(n_samples=50000, seed=42):
    """
    Compute distribution of ideal polyhedron volumes with 5 vertices:
    Fixed: 0, 1, infinity
    Random: two points from sphere
    """
    print(f"Computing {n_samples} five-vertex polyhedron volumes...")
    print("Configuration: vertices at 0, 1, ∞, w₁, w₂")
    
    # Sample points on sphere
    sphere_points = sample_two_points_on_sphere(n_samples, seed)
    
    volumes = []
    failed = 0
    degenerate = 0
    
    for i in range(n_samples):
        if i % 5000 == 0 and i > 0:
            print(f"  Progress: {i}/{n_samples} ({i/n_samples*100:.1f}%)")
        
        # Get two points
        p1, p2 = sphere_points[i]
        
        # Stereographic projections
        w_list = []
        for p in [p1, p2]:
            x, y, z = p
            if abs(z - 1) < 0.001:  # Skip north pole
                w_list.append(None)
            else:
                w_list.append(complex(x/(1-z), y/(1-z)))
        
        # Skip if either projection failed
        if None in w_list:
            failed += 1
            continue
            
        w1, w2 = w_list
        
        # Skip if too close to fixed points or each other
        if (abs(w1) < 0.01 or abs(w1-1) < 0.01 or 
            abs(w2) < 0.01 or abs(w2-1) < 0.01 or 
            abs(w1-w2) < 0.01):
            degenerate += 1
            continue
        
        # Compute volume with vertices at 0, 1, w1, w2 (and infinity implicitly)
        vertices = np.array([0+0j, 1+0j, w1, w2])
        
        try:
            volume = ideal_poly_volume_via_delaunay(vertices, mode='fast', series_terms=96)
            if np.isfinite(volume) and volume > 0:
                volumes.append(volume)
            else:
                failed += 1
        except:
            failed += 1
    
    volumes = np.array(volumes)
    print(f"\nCompleted: {len(volumes)} successful, {failed} failed, {degenerate} degenerate")
    
    return volumes, sphere_points

# Run the analysis
volumes, sphere_points = compute_five_vertex_volumes(50000, seed=42)

# Known maximum for 5 vertices
bipyramid_vol = 2.029883211  # 2 × tetrahedron

print("\n" + "="*70)
print("Distribution Statistics for 5-vertex polyhedra:")
print("="*70)
print(f"  Count: {len(volumes)}")
print(f"  Mean: {np.mean(volumes):.6f} ({np.mean(volumes)/bipyramid_vol*100:.1f}% of max)")
print(f"  Median: {np.median(volumes):.6f} ({np.median(volumes)/bipyramid_vol*100:.1f}% of max)")
print(f"  Std Dev: {np.std(volumes):.6f}")
print(f"  Min: {np.min(volumes):.6f}")
print(f"  Max: {np.max(volumes):.6f} ({np.max(volumes)/bipyramid_vol*100:.1f}% of max)")
print(f"\nKnown maximum (triangular bipyramid): {bipyramid_vol:.6f}")
print(f"Fraction below maximum: {np.sum(volumes < bipyramid_vol) / len(volumes):.4f}")

# Fit distributions
print("\nFitting distributions...")

# Scale to [0, max] for beta distribution
scaled_volumes = volumes / np.max(volumes)
alpha, beta_param, loc, scale = stats.beta.fit(scaled_volumes)
print(f"\nBeta distribution fit (scaled to [0, {np.max(volumes):.4f}]):")
print(f"  α = {alpha:.4f}")
print(f"  β = {beta_param:.4f}")
print(f"  α - β = {alpha - beta_param:.4f}")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Histogram with fits
ax = axes[0, 0]
n, bins, _ = ax.hist(volumes, bins=60, density=True, alpha=0.6, 
                     color='blue', edgecolor='black', label='Data')

# Overlay beta fit
x = np.linspace(0, np.max(volumes)*1.02, 1000)
x_scaled = x / np.max(volumes)
ax.plot(x, stats.beta.pdf(x_scaled, alpha, beta_param) / np.max(volumes), 
        'r-', linewidth=2, label=f'Beta({alpha:.2f}, {beta_param:.2f})')

ax.axvline(bipyramid_vol, color='green', linestyle='--', linewidth=2, 
           label=f'Bipyramid max ({bipyramid_vol:.3f})')
ax.axvline(np.mean(volumes), color='orange', linestyle='--', linewidth=2,
           label=f'Mean ({np.mean(volumes):.3f})')
ax.set_xlabel('Volume')
ax.set_ylabel('Density')
ax.set_title('Distribution of 5-Vertex Polyhedron Volumes')
ax.legend()
ax.set_xlim(0, 2.2)

# 2. Comparison with tetrahedron distribution
ax = axes[0, 1]
ax.hist(volumes/bipyramid_vol, bins=50, density=True, alpha=0.5, 
        color='blue', label='5 vertices', edgecolor='blue')
# Note: we'd need to load or recompute 4-vertex data for actual comparison
ax.axvline(1, color='red', linestyle='--', linewidth=2, label='Maximum')
ax.set_xlabel('Volume / Maximum')
ax.set_ylabel('Density')
ax.set_title('Normalized Volume Distribution')
ax.legend()
ax.set_xlim(0, 1.1)

# 3. Log-scale histogram
ax = axes[1, 0]
ax.hist(volumes, bins=60, density=True, alpha=0.7, color='purple', log=True)
ax.axvline(bipyramid_vol, color='green', linestyle='--', linewidth=2)
ax.set_xlabel('Volume')
ax.set_ylabel('Log Density')
ax.set_title('Log-Scale Distribution')
ax.set_xlim(0, 2.2)

# 4. Q-Q plot
ax = axes[1, 1]
theoretical_quantiles = np.linspace(0.01, 0.99, 99)
empirical_quantiles = np.percentile(volumes, theoretical_quantiles * 100)
scaled_empirical = empirical_quantiles / np.max(volumes)
beta_quantiles = stats.beta.ppf(theoretical_quantiles, alpha, beta_param)
ax.plot(beta_quantiles * np.max(volumes), empirical_quantiles, 'b.', alpha=0.5)
ax.plot([0, 2.2], [0, 2.2], 'r--', linewidth=2)
ax.set_xlabel('Beta Distribution Quantiles')
ax.set_ylabel('Empirical Quantiles')
ax.set_title('Q-Q Plot vs Beta Distribution')
ax.set_xlim(0, 2.2)
ax.set_ylim(0, 2.2)

plt.tight_layout()
plt.savefig('five_vertex_distribution.png', dpi=150)
print("\nSaved plots to five_vertex_distribution.png")

# Additional analysis
print("\n" + "="*70)
print("Comparison with 4-vertex case:")
print("="*70)
print("4 vertices (1 free):")
print("  Mean/Max ratio: ~54%")
print("  Beta parameters: α ≈ 1.08, β ≈ 0.97")
print("  Can get within 0.01% of maximum")
print("\n5 vertices (2 free):")
print(f"  Mean/Max ratio: {np.mean(volumes)/bipyramid_vol*100:.1f}%")
print(f"  Beta parameters: α ≈ {alpha:.2f}, β ≈ {beta_param:.2f}")
print(f"  Closest to maximum: {(bipyramid_vol - np.max(volumes))/bipyramid_vol*100:.2f}% away")

# Look at configurations close to maximum
close_to_max = np.where(volumes > 0.95 * bipyramid_vol)[0]
print(f"\nConfigurations within 95% of maximum: {len(close_to_max)} ({len(close_to_max)/len(volumes)*100:.2f}%)")
close_to_max = np.where(volumes > 0.99 * bipyramid_vol)[0]
print(f"Configurations within 99% of maximum: {len(close_to_max)} ({len(close_to_max)/len(volumes)*100:.2f}%)")

plt.close()