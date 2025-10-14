import numpy as np

# Golden ratio
phi = (1 + np.sqrt(5)) / 2
print(f"Golden ratio φ = {phi:.6f}")
print(f"φ² = {phi**2:.6f}")
print(f"φ³ = {phi**3:.6f}")

# The configuration from seed 42 has these values:
print("\nConfiguration from seed 42:")
print(f"  1.6180 ≈ φ = {phi:.4f}")
print(f"  2.6180 ≈ φ² = {phi**2:.4f}") 
print(f"  3.8911 ≈ ?")
print(f"  4.2361 ≈ φ³ = {phi**3:.4f}")

# Check the approximate values
print("\nChecking special values:")
print(f"  3.8911 / φ = {3.8911 / phi:.4f}")
print(f"  3.8911 / φ² = {3.8911 / (phi**2):.4f}")
print(f"  sqrt(15) = {np.sqrt(15):.4f}")
print(f"  2φ + 1/φ = {2*phi + 1/phi:.4f}")

# Volume comparisons
vol_tetrahedron = 1.0149
vol_cube = 5.0747
vol_config1 = 6.4885
vol_config2 = 6.0017

print(f"\nVolume analysis:")
print(f"  Config 1: {vol_config1:.4f} = {vol_config1/vol_tetrahedron:.4f} × tetrahedron")
print(f"  Config 2: {vol_config2:.4f} = {vol_config2/vol_tetrahedron:.4f} × tetrahedron")
print(f"  Cube: {vol_cube:.4f} = {vol_cube/vol_tetrahedron:.4f} × tetrahedron")

# Check if these are known polyhedra volumes
print(f"\nChecking known ideal polyhedra:")
print(f"  Regular tetrahedron: 1.0149")
print(f"  Regular octahedron: 3.6639") 
print(f"  Regular cube: 5.0747")
print(f"  Config 2 / octahedron = {vol_config2/3.6639:.4f}")

# The factor 6 suggests hexagonal symmetry?
print(f"\nIs config 1 volume related to 2π?")
print(f"  6.4885 / 2π = {6.4885 / (2*np.pi):.4f}")
print(f"  6.4885 / π = {6.4885 / np.pi:.4f}")