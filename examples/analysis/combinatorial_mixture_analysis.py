import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay
import warnings
warnings.filterwarnings('ignore')

print("Combinatorial Mixture Effects in Ideal Polyhedra Volume Distributions")
print("="*70)
print("\nInvestigating how different triangulations create mixture distributions\n")

def analyze_triangulation_changes(n_samples=10000, n_vertices=5, seed=42):
    """Track how triangulation changes as we move vertices"""
    np.random.seed(seed)
    
    print(f"Analyzing {n_vertices} vertices with {n_vertices-3} free points...")
    
    # We'll track triangulation "signatures" and their volumes
    triangulation_types = {}
    volumes_by_type = {}
    
    # Fixed vertices at 0, 1, infinity
    n_free = n_vertices - 3
    
    for i in range(n_samples):
        if i % 1000 == 0 and i > 0:
            print(f"  Progress: {i}/{n_samples}")
        
        # Generate random points on sphere
        sphere_points = np.random.randn(n_free, 3)
        sphere_points = sphere_points / np.linalg.norm(sphere_points, axis=1, keepdims=True)
        
        # Convert to complex via stereographic projection
        vertices = [0+0j, 1+0j]  # Fixed vertices
        
        valid = True
        for p in sphere_points:
            x, y, z = p
            if abs(z - 1) < 0.01:  # Skip north pole
                valid = False
                break
            w = complex(x/(1-z), y/(1-z))
            
            # Skip if too close to fixed points
            if abs(w) < 0.01 or abs(w-1) < 0.01:
                valid = False
                break
            vertices.append(w)
        
        if not valid:
            continue
        
        vertices = np.array(vertices)
        
        try:
            # Get triangulation
            points_2d = np.column_stack([vertices.real, vertices.imag])
            tri = Delaunay(points_2d)
            
            # Create a "signature" for this triangulation
            # Sort the simplices to get a canonical form
            simplices_sorted = np.sort(tri.simplices, axis=1)
            simplices_sorted = np.sort(simplices_sorted, axis=0)
            signature = tuple(map(tuple, simplices_sorted))
            
            # Compute volume
            volume = ideal_poly_volume_via_delaunay(vertices, mode='fast')
            
            if signature not in triangulation_types:
                triangulation_types[signature] = 0
                volumes_by_type[signature] = []
            
            triangulation_types[signature] += 1
            volumes_by_type[signature].append(volume)
            
        except:
            continue
    
    return triangulation_types, volumes_by_type

# Run analysis for 5 vertices
print("\nAnalyzing 5-vertex configurations...")
types_5, volumes_5 = analyze_triangulation_changes(10000, 5)

print(f"\nFound {len(types_5)} different triangulation types")
print("\nMost common triangulation types:")
sorted_types = sorted(types_5.items(), key=lambda x: x[1], reverse=True)
for i, (sig, count) in enumerate(sorted_types[:5]):
    print(f"  Type {i+1}: {count} occurrences ({count/100:.1f}%)")
    mean_vol = np.mean(volumes_5[sig])
    std_vol = np.std(volumes_5[sig])
    print(f"    Mean volume: {mean_vol:.4f}, Std: {std_vol:.4f}")

# Analyze mixture components
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Distribution of triangulation type frequencies
ax = axes[0, 0]
frequencies = list(types_5.values())
ax.hist(frequencies, bins=30, alpha=0.7, color='blue', edgecolor='black')
ax.set_xlabel('Number of Occurrences')
ax.set_ylabel('Count')
ax.set_title(f'Distribution of Triangulation Type Frequencies ({len(types_5)} types)')
ax.set_yscale('log')

# 2. Volume distributions for top triangulation types
ax = axes[0, 1]
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, (sig, count) in enumerate(sorted_types[:5]):
    if count > 50:  # Only plot if we have enough samples
        vols = volumes_5[sig]
        ax.hist(vols, bins=30, alpha=0.5, density=True, 
                label=f'Type {i+1} (n={count})', color=colors[i % len(colors)])
ax.set_xlabel('Volume')
ax.set_ylabel('Density')
ax.set_title('Volume Distributions by Triangulation Type')
ax.legend()

# 3. Mean volume vs frequency
ax = axes[1, 0]
mean_vols = []
counts = []
for sig, count in types_5.items():
    if count > 10:  # Need enough samples for reliable mean
        mean_vols.append(np.mean(volumes_5[sig]))
        counts.append(count)

ax.scatter(counts, mean_vols, alpha=0.6)
ax.set_xlabel('Frequency of Triangulation Type')
ax.set_ylabel('Mean Volume')
ax.set_title('Mean Volume vs Triangulation Frequency')
ax.set_xscale('log')

# 4. Analysis summary
ax = axes[1, 1]
ax.text(0.5, 0.9, "Mixture Distribution Analysis", 
        fontsize=14, weight='bold', ha='center', transform=ax.transAxes)

analysis_text = f"""
5 vertices: {len(types_5)} triangulation types found

Top type: {sorted_types[0][1]/100:.1f}% of configurations
Top 5 types: {sum(x[1] for x in sorted_types[:5])/100:.1f}% of configurations

Key findings:
• Most configurations use common triangulations
• Rare triangulations create distribution tails
• Each type has its own volume distribution
• Overall distribution is a weighted mixture

This explains:
→ Smooth overall distribution (averaging)
→ Potential for subtle multimodality
→ Deviations from pure CLT behavior
"""

ax.text(0.05, 0.05, analysis_text, fontsize=10, 
        ha='left', va='bottom', transform=ax.transAxes,
        family='monospace')
ax.axis('off')

plt.tight_layout()
plt.savefig('combinatorial_mixture_analysis.png', dpi=150)
print("\nSaved analysis to combinatorial_mixture_analysis.png")

# Now let's see if this explains the Beta distribution shape
print("\n\nTesting mixture hypothesis:")
print("-"*50)

# Combine all volumes
all_volumes = []
for vols in volumes_5.values():
    all_volumes.extend(vols)
all_volumes = np.array(all_volumes)

# Fit Beta distribution
from scipy import stats
scaled_vols = all_volumes / np.max(all_volumes)
alpha, beta, loc, scale = stats.beta.fit(scaled_vols)

print(f"Overall Beta fit: α={alpha:.2f}, β={beta:.2f}")
print(f"Mean volume across all types: {np.mean(all_volumes):.4f}")
print(f"Std deviation: {np.std(all_volumes):.4f}")

# Check for multimodality
from scipy.stats import gaussian_kde
kde = gaussian_kde(all_volumes)
x_range = np.linspace(0, np.max(all_volumes), 1000)
density = kde(x_range)

# Find local maxima
from scipy.signal import find_peaks
peaks, _ = find_peaks(density, prominence=0.1)

print(f"\nNumber of significant modes in distribution: {len(peaks)}")
if len(peaks) > 1:
    print("Evidence of multimodality due to mixture effects!")
else:
    print("Distribution appears unimodal despite mixture")

plt.close()