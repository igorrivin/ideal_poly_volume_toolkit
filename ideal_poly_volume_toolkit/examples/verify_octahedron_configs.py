import numpy as np
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay

# Configuration 1: Points on unit circle at 90° intervals
# Vertices: 0, 1, i, -1, -i, ∞
config1 = np.array([0+0j, 1+0j, 0+1j, -1+0j, 0-1j])
vol1 = ideal_poly_volume_via_delaunay(config1, mode='exact', dps=250)
print("Configuration 1 (unit circle):")
print("  Vertices: 0, 1, i, -1, -i, ∞")
print(f"  Volume: {vol1:.12f}")

# Configuration 2: The one found by free optimization
# Vertices: 0, 1, 1+i, 2, 1-i, ∞
config2 = np.array([0+0j, 1+0j, 1+1j, 2+0j, 1-1j])
vol2 = ideal_poly_volume_via_delaunay(config2, mode='exact', dps=250)
print("\nConfiguration 2 (free optimization):")
print("  Vertices: 0, 1, 1+i, 2, 1-i, ∞")
print(f"  Volume: {vol2:.12f}")

# Check if they're the same
print(f"\nVolumes equal? {abs(vol1 - vol2) < 1e-10}")

# These two configurations are related by a Möbius transformation!
# The transformation z → 2z/(z+1) maps the unit circle to a different circle
# Let's verify this maps config1 to config2
def mobius(z):
    """Apply Möbius transformation z → 2z/(z+1)"""
    return 2*z / (z + 1)

print("\nApplying Möbius transformation z → 2z/(z+1) to config1:")
for i, z in enumerate([0+0j, 1+0j, 0+1j, -1+0j, 0-1j]):
    w = mobius(z) if np.isfinite(z) else np.inf
    print(f"  {z} → {w}")

# The transformation maps:
# 0 → 0
# 1 → 1  
# i → 2i/(i+1) = 2i(1-i)/2 = i(1-i) = i-i² = i+1 = 1+i
# -1 → ∞
# -i → -2i/(-i+1) = -2i(1+i)/2 = -i(1+i) = -i-i² = -i+1 = 1-i

# And ∞ → 2 (limit as z→∞)
print("  ∞ → 2")

print("\nThis matches configuration 2 (with -1 mapped to ∞)!")