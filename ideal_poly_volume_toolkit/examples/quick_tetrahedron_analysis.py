import numpy as np
import matplotlib.pyplot as plt
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay

# Quick analysis with 5000 samples
n_samples = 5000
print(f"Analyzing {n_samples} random ideal tetrahedra...")

# Uniform sampling on sphere
np.random.seed(42)
volumes = []

for i in range(n_samples):
    if i % 500 == 0:
        print(f"Progress: {i}/{n_samples}")
    
    # Sample uniform point on sphere using Gaussian method
    vec = np.random.randn(3)
    vec = vec / np.linalg.norm(vec)
    x, y, z = vec
    
    # Skip north pole (maps to infinity)
    if z > 0.999:
        continue
        
    # Stereographic projection
    w = complex(x/(1-z), y/(1-z))
    
    # Skip if too close to 0 or 1
    if abs(w) < 0.01 or abs(w-1) < 0.01:
        continue
    
    # Compute volume
    vertices = np.array([0+0j, 1+0j, w])
    vol = ideal_poly_volume_via_delaunay(vertices, mode='fast', series_terms=96)
    volumes.append(vol)

volumes = np.array(volumes)
print(f"\nAnalyzed {len(volumes)} configurations")

# Statistics
regular_vol = 1.01494160640965
print(f"\nStatistics:")
print(f"  Regular tetrahedron: {regular_vol:.6f}")
print(f"  Mean volume: {np.mean(volumes):.6f} ({np.mean(volumes)/regular_vol*100:.1f}% of regular)")
print(f"  Median volume: {np.median(volumes):.6f} ({np.median(volumes)/regular_vol*100:.1f}% of regular)")
print(f"  Max volume: {np.max(volumes):.6f} ({np.max(volumes)/regular_vol*100:.1f}% of regular)")
print(f"  Min volume: {np.min(volumes):.6f}")
print(f"  Std dev: {np.std(volumes):.6f}")

# Create simple histogram
plt.figure(figsize=(10, 6))
plt.hist(volumes, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(regular_vol, color='red', linestyle='--', linewidth=2, 
           label=f'Regular tetrahedron ({regular_vol:.3f})')
plt.axvline(np.mean(volumes), color='green', linestyle='--', linewidth=2,
           label=f'Mean ({np.mean(volumes):.3f})')
plt.xlabel('Volume', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Distribution of Random Ideal Tetrahedron Volumes', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tetrahedron_volume_histogram.png', dpi=150)
print("\nSaved histogram to tetrahedron_volume_histogram.png")

# Check how volume depends on position
# Find the configurations closest to regular
distances_from_regular = np.abs(volumes - regular_vol)
closest_indices = np.argsort(distances_from_regular)[:5]

print(f"\nConfigurations closest to regular tetrahedron:")
for idx in closest_indices:
    print(f"  Volume: {volumes[idx]:.6f} (diff: {regular_vol - volumes[idx]:.6f})")

# Theoretical consideration
print(f"\nTheoretical note:")
print(f"The regular tetrahedron has all dihedral angles equal.")
print(f"Random configurations break this symmetry, leading to smaller volumes.")
print(f"The distribution appears to have a long tail towards small volumes.")

plt.close()