import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("Theoretical Reasons for Beta Distribution in Ideal Polyhedra Volumes")
print("="*70)
print("\nExploring why volumes naturally follow a Beta distribution...\n")

print("Key insights about Beta distributions:")
print("-"*50)
print("1. Beta is the 'natural' distribution for bounded random variables on [0,1]")
print("2. It's the conjugate prior for binomial/Bernoulli processes")
print("3. It emerges from order statistics of uniform random variables")
print("4. Maximum entropy distribution given constraints on E[log(X)] and E[log(1-X)]")

print("\n\nPossible theoretical explanations for our case:")
print("="*70)

print("\n1. BOUNDED DOMAIN + GEOMETRIC CONSTRAINTS")
print("-"*50)
print("• Volumes are bounded: 0 < V < V_max")
print("• Vertices constrained to sphere (compact domain)")
print("• Beta naturally models variables with hard bounds")
print("• Rescaling to [0,1] makes Beta the canonical choice")

print("\n2. PRODUCT STRUCTURE OF VOLUME FORMULA")
print("-"*50)
print("The Lobachevsky function Λ(θ) = -∫[0,θ] log|2sin(t)| dt")
print("For tetrahedron: V = Σ Λ(dihedral angles)")
print("• Products/sums of bounded terms → Beta-like distributions")
print("• Similar to how products of uniforms give Beta")

print("\n3. PROJECTIVE GEOMETRY CONNECTION")
print("-"*50)
print("• Möbius transformations preserve cross-ratios")
print("• Cross-ratios of random points → Beta distributions")
print("• Our volume formula involves cross-ratio-like terms")
print("• Stereographic projection preserves these properties")

print("\n4. ORDER STATISTICS INTERPRETATION")
print("-"*50)
print("Beta(α,β) is the distribution of the k-th order statistic")
print("of n uniform random variables, where α=k, β=n-k+1")
print("• Our increasing α with more vertices suggests")
print("  we're seeing higher order statistics")
print("• Volume might act like a 'sorted' geometric quantity")

# Visualize theoretical connections
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Show how Beta emerges from order statistics
ax = axes[0, 0]
n_samples = 10000
n_uniform = 5
uniforms = np.random.uniform(0, 1, (n_samples, n_uniform))
kth_order = np.sort(uniforms, axis=1)[:, 3]  # 4th order statistic
ax.hist(kth_order, bins=50, density=True, alpha=0.6, label='4th order stat of 5 uniforms')
x = np.linspace(0, 1, 100)
ax.plot(x, stats.beta.pdf(x, 4, 2), 'r-', linewidth=2, label='Beta(4,2)')
ax.set_title('Order Statistics → Beta Distribution')
ax.legend()
ax.set_xlabel('Value')
ax.set_ylabel('Density')

# 2. Maximum entropy interpretation
ax = axes[0, 1]
ax.text(0.5, 0.9, "Maximum Entropy Principle", 
        fontsize=14, weight='bold', ha='center', transform=ax.transAxes)
explanation = """
Given constraints:
• X ∈ [0, 1]
• E[log(X)] = fixed
• E[log(1-X)] = fixed

The maximum entropy distribution is Beta(α,β)
where α and β encode the constraints.

For ideal polyhedra:
• Geometric constraints on vertices
• Topological constraints (triangulation)
• Volume bounded by maximum

→ Beta emerges as the most "unbiased"
   distribution satisfying all constraints
"""
ax.text(0.05, 0.05, explanation, fontsize=10, 
        ha='left', va='bottom', transform=ax.transAxes, 
        family='monospace')
ax.axis('off')

# 3. Connection to hyperbolic geometry
ax = axes[1, 0]
theta = np.linspace(0.1, np.pi-0.1, 100)
lobachevsky_integrand = -np.log(2*np.abs(np.sin(theta)))
ax.plot(theta, lobachevsky_integrand, 'b-', linewidth=2)
ax.fill_between(theta, 0, lobachevsky_integrand, alpha=0.3)
ax.set_xlabel('θ (dihedral angle)')
ax.set_ylabel('-log|2sin(θ)|')
ax.set_title('Lobachevsky Function Integrand')
ax.text(np.pi/2, 2, 'Area = Λ(θ)', fontsize=12, ha='center')

# 4. Why α increases with vertices
ax = axes[1, 1]
vertices = [4, 5, 6, 7, 8]
predicted_alpha = [1.08, 4.77, 13, 25, 40]  # Rough exponential growth
ax.semilogy(vertices, predicted_alpha, 'go-', markersize=10, linewidth=2)
ax.set_xlabel('Number of Vertices')
ax.set_ylabel('Beta α parameter (log scale)')
ax.set_title('Growth of Concentration Parameter')
ax.grid(True, alpha=0.3)
ax.text(6, 5, 'Exponential\ngrowth', fontsize=12, ha='center',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))

plt.tight_layout()
plt.savefig('beta_distribution_theory.png', dpi=150)
print("\nSaved theoretical analysis to beta_distribution_theory.png")

# Mathematical conjecture
print("\n\nMATHEMATICAL CONJECTURE:")
print("="*70)
print("The volume of a random ideal polyhedron with n vertices")
print("(3 fixed at 0, 1, ∞ and n-3 uniform on the sphere)")
print("follows approximately Beta(α_n, β_n) where:")
print("")
print("• α_n grows exponentially with n")
print("• β_n grows sublinearly")
print("• Mean → some limit < 1 as n → ∞")
print("")
print("This might connect to:")
print("1. Random matrix theory (hyperbolic isometries as SL(2,C))")
print("2. Free probability (non-commutative distributions)")
print("3. Quantum gravity (random triangulations of hyperbolic space)")

# Test universality
print("\n\nTesting universality hypothesis:")
print("-"*50)
print("Question: Does the Beta emergence depend on our specific setup?")
print("\nVariations to test:")
print("• Different measures on the sphere (not uniform)")
print("• Different fixed points (not 0, 1, ∞)")
print("• Different hyperbolic polyhedra (not ideal)")
print("\nPrediction: Beta distribution is universal due to:")
print("• Bounded domain")
print("• Maximum entropy under geometric constraints")
print("• Central limit effects in hyperbolic geometry")

plt.close()