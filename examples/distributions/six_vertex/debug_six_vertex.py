import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Let's debug the 6-vertex distribution
print("Debugging 6-vertex distribution anomaly...")
print("="*70)

# Load a small sample to investigate
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay

# First, let's check if the distribution is actually that concentrated
np.random.seed(42)
volumes = []

for i in range(1000):
    # Three random points on sphere
    vecs = np.random.randn(3, 3)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    
    # Stereographic projections
    w_list = []
    for p in vecs:
        x, y, z = p
        if abs(z - 1) < 0.01:
            continue
        w_list.append(complex(x/(1-z), y/(1-z)))
    
    if len(w_list) != 3:
        continue
    
    # Check for degeneracies
    skip = False
    for w in w_list:
        if abs(w) < 0.01 or abs(w-1) < 0.01:
            skip = True
    if skip:
        continue
    
    vertices = np.array([0+0j, 1+0j] + w_list)
    
    try:
        vol = ideal_poly_volume_via_delaunay(vertices, mode='fast')
        if vol > 0:
            volumes.append(vol)
    except:
        pass

volumes = np.array(volumes)
print(f"Sample size: {len(volumes)}")
print(f"Mean: {np.mean(volumes):.4f}")
print(f"Std: {np.std(volumes):.4f}")
print(f"Min: {np.min(volumes):.4f}")
print(f"Max: {np.max(volumes):.4f}")

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(volumes, bins=50, density=True, alpha=0.7, edgecolor='black')
plt.xlabel('Volume')
plt.ylabel('Density')
plt.title('6-Vertex Distribution (Debug Sample)')

# Try different fitting approaches
print("\n\nTrying different distribution fits:")
print("-"*50)

# 1. Standard Beta fit
scaled = (volumes - np.min(volumes)) / (np.max(volumes) - np.min(volumes))
scaled = np.clip(scaled, 1e-10, 1-1e-10)  # Ensure strictly in (0,1)
params_beta = stats.beta.fit(scaled)
print(f"Beta fit: α={params_beta[0]:.2f}, β={params_beta[1]:.2f}")

# 2. Method of moments
mean_scaled = np.mean(scaled)
var_scaled = np.var(scaled)
alpha_mom = mean_scaled * (mean_scaled * (1 - mean_scaled) / var_scaled - 1)
beta_mom = (1 - mean_scaled) * (mean_scaled * (1 - mean_scaled) / var_scaled - 1)
print(f"Beta (moments): α={alpha_mom:.2f}, β={beta_mom:.2f}")

# 3. Check if it's actually nearly constant
print(f"\nCoefficient of variation: {np.std(volumes)/np.mean(volumes):.4f}")
print(f"Range/Mean: {(np.max(volumes)-np.min(volumes))/np.mean(volumes):.4f}")

# Look at the actual distribution shape
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(volumes, bins=50, density=True, alpha=0.7)
plt.xlabel('Volume')
plt.title('Raw Distribution')

plt.subplot(1, 3, 2)
plt.hist(scaled, bins=50, density=True, alpha=0.7)
x = np.linspace(0, 1, 100)
plt.plot(x, stats.beta.pdf(x, alpha_mom, beta_mom), 'r-', linewidth=2)
plt.xlabel('Scaled Volume')
plt.title('Scaled with Beta Fit')

plt.subplot(1, 3, 3)
# Check normality of log volumes
log_vols = np.log(volumes)
plt.hist(log_vols, bins=50, density=True, alpha=0.7)
plt.xlabel('Log(Volume)')
plt.title('Log-transformed')

plt.tight_layout()
plt.savefig('debug_six_vertex.png')

print("\n\nHypothesis: For 6 vertices (octahedron-like):")
print("The regular octahedron might be very close to optimal,")
print("causing extreme concentration near this configuration.")
print("This would explain the huge α value in the Beta fit.")

plt.close('all')