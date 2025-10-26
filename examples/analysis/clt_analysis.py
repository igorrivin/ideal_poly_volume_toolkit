import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay
import warnings
warnings.filterwarnings('ignore')

print("Central Limit Theorem-like Behavior in Ideal Polyhedra Volumes")
print("="*70)
print("\nHypothesis: As we add more free vertices, the volume distribution")
print("concentrates around its mean, similar to a CLT phenomenon.\n")

# Summary of what we've found
results = {
    4: {"mean_ratio": 0.537, "std_ratio": 0.22, "alpha": 1.077, "beta": 0.974, 
        "max_vol": 1.01494, "name": "tetrahedron"},
    5: {"mean_ratio": 0.658, "std_ratio": 0.14, "alpha": 4.77, "beta": 1.80,
        "max_vol": 2.02988, "name": "triangular bipyramid"}
}

print("Observed pattern:")
print("Vertices | Mean/Max | Std/Max | Beta(α, β)")
print("-"*50)
for n_verts, data in results.items():
    print(f"{n_verts:8d} | {data['mean_ratio']:8.3f} | {data['std_ratio']:7.3f} | "
          f"Beta({data['alpha']:.2f}, {data['beta']:.2f})")

# Theoretical CLT-like scaling
print("\n\nCLT-like concentration analysis:")
print("-"*50)

# For CLT, we'd expect std ~ 1/sqrt(degrees of freedom)
# But here it's more complex due to:
# 1. Non-linear volume formula
# 2. Constrained domain (sphere)
# 3. Discrete combinatorial changes

print("\nKey observations:")
print("1. Mean/Max ratio increases: 0.537 → 0.658")
print("2. Relative std decreases: 0.22 → 0.14")
print("3. Beta α parameter increases rapidly: 1.08 → 4.77")
print("4. Distribution becomes more concentrated")

# Visualize the concentration
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Compare the Beta distributions
ax = axes[0, 0]
x = np.linspace(0, 1, 1000)
for n_verts, data in results.items():
    y = stats.beta.pdf(x, data['alpha'], data['beta'])
    ax.plot(x, y, linewidth=2, label=f"{n_verts} vertices")
ax.set_xlabel('Normalized Volume')
ax.set_ylabel('Density')
ax.set_title('Beta Distribution Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Concentration measure (α + β)
ax = axes[0, 1]
n_verts_list = list(results.keys())
concentration = [data['alpha'] + data['beta'] for data in results.values()]
ax.plot(n_verts_list, concentration, 'bo-', markersize=10, linewidth=2)
ax.set_xlabel('Number of Vertices')
ax.set_ylabel('α + β (concentration parameter)')
ax.set_title('Distribution Concentration Growth')
ax.grid(True, alpha=0.3)
ax.text(4.5, 5, f"Growth: {concentration[1]/concentration[0]:.1f}x", 
        fontsize=12, ha='center')

# 3. Mean and std evolution
ax = axes[1, 0]
mean_ratios = [data['mean_ratio'] for data in results.values()]
std_ratios = [data['std_ratio'] for data in results.values()]
ax.plot(n_verts_list, mean_ratios, 'go-', markersize=10, linewidth=2, label='Mean/Max')
ax.plot(n_verts_list, std_ratios, 'ro-', markersize=10, linewidth=2, label='Std/Max')
ax.set_xlabel('Number of Vertices')
ax.set_ylabel('Ratio')
ax.set_title('Mean and Standard Deviation Evolution')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Theoretical analysis
ax = axes[1, 1]
ax.text(0.5, 0.9, "Why CLT-like but not exactly CLT:", 
        fontsize=14, weight='bold', ha='center', transform=ax.transAxes)

reasons = [
    "1. Volume is highly nonlinear in vertex positions",
    "2. Different combinatorial types (triangulations)",
    "3. Domain is compact (sphere), not unbounded",
    "4. Dependencies through Delaunay triangulation",
    "5. Mixture of discrete and continuous effects"
]

for i, reason in enumerate(reasons):
    ax.text(0.1, 0.7 - i*0.12, reason, fontsize=11, 
            ha='left', transform=ax.transAxes)

ax.text(0.5, 0.1, "Result: Concentration but with unique characteristics",
        fontsize=12, style='italic', ha='center', transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
ax.axis('off')

plt.tight_layout()
plt.savefig('clt_analysis.png', dpi=150)
print("\nSaved analysis to clt_analysis.png")

# Predict next values
print("\n\nPredictions for 6 vertices (3 free points):")
print("-"*50)

# Rough extrapolation
if len(n_verts_list) >= 2:
    # Mean ratio appears to grow
    mean_growth = mean_ratios[1] / mean_ratios[0]
    predicted_mean = mean_ratios[-1] * (mean_growth ** 0.7)  # Sublinear growth
    
    # Std ratio appears to shrink
    std_shrink = std_ratios[1] / std_ratios[0]
    predicted_std = std_ratios[-1] * std_shrink
    
    # Concentration grows rapidly
    conc_growth = concentration[1] / concentration[0]
    predicted_conc = concentration[-1] * (conc_growth ** 0.8)
    
    print(f"Predicted mean/max ratio: ~{predicted_mean:.2f}")
    print(f"Predicted std/max ratio: ~{predicted_std:.2f}")
    print(f"Predicted α + β: ~{predicted_conc:.0f}")
    print(f"This suggests Beta(~{predicted_conc*0.8:.0f}, ~{predicted_conc*0.2:.0f})")

print("\n\nMixture effects from combinatorial types:")
print("-"*50)
print("As vertices increase, we get more triangulation possibilities.")
print("Each triangulation type might have its own 'mode' in the distribution.")
print("This could create:")
print("- Multi-modal distributions")
print("- Heavier tails than pure CLT would predict")
print("- Discrete jumps in the volume function")
print("\nThe observed smooth Beta distributions suggest these effects")
print("are being 'averaged out' by the continuous vertex positions.")

plt.close()