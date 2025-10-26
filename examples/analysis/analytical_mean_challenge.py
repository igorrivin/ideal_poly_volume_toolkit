import numpy as np
import matplotlib.pyplot as plt
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay

print("The Analytical Challenge of Computing Mean Tetrahedron Volume")
print("="*70)

print("\nThe setup:")
print("- Vertices at 0, 1, ∞, and w (fourth vertex)")
print("- w = stereographic projection of uniform point on S²")
print("- Volume = Re[D(w) + D(w/(w-1)) + D(1-1/w)]")
print("  where D is the Bloch-Wigner dilogarithm")

print("\nThe integral we need:")
print("  E[Volume] = (1/4π) ∫∫_S² Volume(stereographic(x,y,z)) dS")
print("            = (1/2π) ∫₀^π ∫₀^{2π} Volume(w(θ,φ)) sin(θ) dφ dθ")

print("\nWhy it's intractable:")
print("1. The stereographic projection: w = cot(θ/2)e^{iφ}")
print("2. Each dilogarithm term becomes highly complex")
print("3. The integrand involves compositions of transcendental functions")
print("4. No known closed form for integrals of dilogarithm compositions")

# Let's visualize how the volume varies with position on the sphere
print("\n\nVisualizing the complexity...")

# Sample on a grid
n_theta = 50
n_phi = 100
theta = np.linspace(0.1, np.pi-0.1, n_theta)  # Avoid poles
phi = np.linspace(0, 2*np.pi, n_phi)

# Compute volumes on grid
volumes_grid = np.zeros((n_theta, n_phi))

for i, t in enumerate(theta):
    if i % 10 == 0:
        print(f"Computing row {i}/{n_theta}")
    for j, p in enumerate(phi):
        # Spherical to Cartesian
        x = np.sin(t) * np.cos(p)
        y = np.sin(t) * np.sin(p)
        z = np.cos(t)
        
        # Stereographic projection
        if abs(z - 1) < 0.01:  # Skip near north pole
            volumes_grid[i, j] = 0
            continue
        
        w = complex(x/(1-z), y/(1-z))
        
        # Skip if near 0 or 1
        if abs(w) < 0.01 or abs(w-1) < 0.01:
            volumes_grid[i, j] = 0
            continue
        
        # Compute volume
        vertices = np.array([0+0j, 1+0j, w])
        vol = ideal_poly_volume_via_delaunay(vertices, mode='fast')
        volumes_grid[i, j] = vol

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Volume as function of position on sphere
ax = axes[0, 0]
im = ax.imshow(volumes_grid, extent=[0, 2*np.pi, np.pi, 0], 
               aspect='auto', cmap='viridis', interpolation='bilinear')
ax.set_xlabel('φ (azimuthal angle)')
ax.set_ylabel('θ (polar angle)')
ax.set_title('Volume as Function of Sphere Position')
plt.colorbar(im, ax=ax, label='Volume')

# 2. Radial average
ax = axes[0, 1]
radial_avg = np.mean(volumes_grid, axis=1)
ax.plot(theta, radial_avg, 'b-', linewidth=2)
ax.set_xlabel('θ (polar angle)')
ax.set_ylabel('Average Volume')
ax.set_title('Azimuthally Averaged Volume')
ax.grid(True, alpha=0.3)

# 3. Look at specific slices
ax = axes[1, 0]
for i in [10, 20, 30, 40]:
    ax.plot(phi, volumes_grid[i, :], label=f'θ={theta[i]:.2f}')
ax.set_xlabel('φ (azimuthal angle)')
ax.set_ylabel('Volume')
ax.set_title('Volume vs Azimuthal Angle (fixed θ)')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Distribution of grid values
ax = axes[1, 1]
flat_volumes = volumes_grid.flatten()
flat_volumes = flat_volumes[flat_volumes > 0]  # Remove zeros
ax.hist(flat_volumes, bins=50, density=True, alpha=0.7, color='purple')
ax.set_xlabel('Volume')
ax.set_ylabel('Density')
ax.set_title('Distribution of Grid Volumes')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analytical_mean_challenge.png', dpi=150)

# Numerical integration attempt
print("\n\nNumerical integration on grid:")
# Account for measure sin(θ)
weights = np.sin(theta[:, np.newaxis]) * (theta[1]-theta[0]) * (phi[1]-phi[0])
total_weight = np.sum(weights)
mean_estimate = np.sum(volumes_grid * weights) / total_weight

print(f"Grid-based mean estimate: {mean_estimate:.4f}")
print(f"Monte Carlo mean (from earlier): ~0.543")
print(f"Difference: {abs(mean_estimate - 0.543):.4f}")

# Show why analytical is hopeless
print("\n\nThe analytical integrand would involve terms like:")
print("  ∫∫ Re[Li₂(cot(θ/2)e^{iφ})] sin(θ) dθ dφ")
print("  ∫∫ Re[Li₂(cot(θ/2)e^{iφ}/(cot(θ/2)e^{iφ}-1))] sin(θ) dθ dφ")
print("  etc.")
print("\nEach integral involves:")
print("- Dilogarithm of complex argument")
print("- Rational functions of trigonometric expressions")
print("- No separation of variables possible")
print("- No known special function representations")

print("\nConclusion: Monte Carlo simulation is likely the only practical approach!")

plt.close()