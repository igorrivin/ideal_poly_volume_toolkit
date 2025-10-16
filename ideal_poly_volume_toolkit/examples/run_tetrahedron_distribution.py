#!/usr/bin/env python
"""Run tetrahedron volume distribution analysis"""

from tetrahedron_volume_distribution import compute_tetrahedron_volumes, analyze_distribution, plot_distribution
import numpy as np

# Run with 20,000 samples for good statistics
print("Running tetrahedron volume distribution analysis...")
print("=" * 60)

n_samples = 20000
volumes, sphere_points = compute_tetrahedron_volumes(n_samples, seed=42)

# Analyze
print("\n" + "=" * 60)
stats = analyze_distribution(volumes)

# Additional analysis
print("\nAdditional Analysis:")
print(f"  Coefficient of variation: {stats['std']/stats['mean']:.4f}")
print(f"  Skewness: {np.mean(((volumes - stats['mean'])/stats['std'])**3):.4f}")

# Look at percentiles more carefully
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"\nPercentiles:")
for p in percentiles:
    print(f"  {p:3d}%: {np.percentile(volumes, p):.6f}")

# How close can we get to the regular tetrahedron?
regular_vol = 1.01494160640965
closest_idx = np.argmax(volumes)
closest_vol = volumes[closest_idx]
print(f"\nClosest to regular tetrahedron:")
print(f"  Volume: {closest_vol:.6f} ({closest_vol/regular_vol*100:.2f}% of regular)")
z_closest = np.array([0+0j, 1+0j, 
                     complex(sphere_points[closest_idx][0]/(1-sphere_points[closest_idx][2]),
                            sphere_points[closest_idx][1]/(1-sphere_points[closest_idx][2]))])
print(f"  Vertices: 0, 1, ∞, {z_closest[2]:.4f}")

# Visualize
plot_distribution(volumes, bins=100)

print("\nAnalysis complete!")