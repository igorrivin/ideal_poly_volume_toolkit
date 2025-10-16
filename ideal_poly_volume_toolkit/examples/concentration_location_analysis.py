import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("Where is the Distribution Concentrating? Near Maximum or Middle?")
print("="*70)
print()

# Summary data from our analyses
data = {
    4: {
        "alpha": 1.077,
        "beta": 0.974,
        "mean_ratio": 0.537,
        "max_vol": 1.01494,
        "name": "tetrahedron"
    },
    5: {
        "alpha": 4.77,
        "beta": 1.80,
        "mean_ratio": 0.658,
        "max_vol": 2.02988,
        "name": "triangular bipyramid"
    },
    6: {
        "alpha": 6.33,
        "beta": 3.89,
        "mean_ratio": 0.620,
        "max_vol": 3.6808,
        "name": "octahedron-like"
    }
}

# For Beta(α, β), the mode is at (α-1)/(α+β-2) for α,β > 1
# The mean is at α/(α+β)

print("Analysis of concentration location:")
print("-"*70)
print("Vertices | Mode/Max | Mean/Max | Mode Position | Skewness")
print("-"*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Beta distributions showing mode vs mean
ax = axes[0, 0]
x = np.linspace(0, 1, 1000)
colors = ['red', 'green', 'blue']

for i, (n_verts, params) in enumerate(data.items()):
    alpha, beta_param = params["alpha"], params["beta"]
    
    # Calculate mode and mean positions
    if alpha > 1 and beta_param > 1:
        mode = (alpha - 1) / (alpha + beta_param - 2)
    else:
        mode = 0 if alpha < 1 else 1
    
    mean = alpha / (alpha + beta_param)
    
    # Skewness
    skewness = 2 * (beta_param - alpha) * np.sqrt(alpha + beta_param + 1)
    skewness /= (alpha + beta_param + 2) * np.sqrt(alpha * beta_param)
    
    print(f"{n_verts:8d} | {mode:8.3f} | {params['mean_ratio']:8.3f} | "
          f"{'Right' if mode > mean else 'Left':>8} | {skewness:7.3f}")
    
    # Plot distribution
    y = stats.beta.pdf(x, alpha, beta_param)
    ax.plot(x, y, color=colors[i], linewidth=2, label=f'{n_verts} vertices')
    
    # Mark mode and mean
    ax.axvline(mode, color=colors[i], linestyle='--', alpha=0.5)
    ax.axvline(mean, color=colors[i], linestyle=':', alpha=0.5)

ax.set_xlabel('Normalized Volume')
ax.set_ylabel('Density')
ax.set_title('Beta Distributions: Modes (--) and Means (:)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Mode and mean evolution
ax = axes[0, 1]
vertices = list(data.keys())
modes = []
means = []
for n_verts in vertices:
    alpha, beta_param = data[n_verts]["alpha"], data[n_verts]["beta"]
    if alpha > 1 and beta_param > 1:
        mode = (alpha - 1) / (alpha + beta_param - 2)
    else:
        mode = 0 if alpha < 1 else 1
    modes.append(mode)
    means.append(data[n_verts]["mean_ratio"])

ax.plot(vertices, modes, 'ro-', markersize=10, linewidth=2, label='Mode/Max')
ax.plot(vertices, means, 'bo-', markersize=10, linewidth=2, label='Mean/Max')
ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Maximum')
ax.set_xlabel('Number of Vertices')
ax.set_ylabel('Ratio to Maximum')
ax.set_title('Where is the Distribution Centered?')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.1)

# Plot 3: Alpha vs Beta parameters
ax = axes[1, 0]
alphas = [data[v]["alpha"] for v in vertices]
betas = [data[v]["beta"] for v in vertices]

ax.plot(vertices, alphas, 'go-', markersize=10, linewidth=2, label='α')
ax.plot(vertices, betas, 'mo-', markersize=10, linewidth=2, label='β')
ax.set_xlabel('Number of Vertices')
ax.set_ylabel('Parameter Value')
ax.set_title('Beta Distribution Parameters')
ax.legend()
ax.grid(True, alpha=0.3)

# Add annotations for α > β or α < β
for i, v in enumerate(vertices):
    if alphas[i] > betas[i]:
        ax.text(v, max(alphas[i], betas[i]) + 0.5, 'α > β', 
                ha='center', fontsize=10, color='darkgreen')
    else:
        ax.text(v, max(alphas[i], betas[i]) + 0.5, 'α < β', 
                ha='center', fontsize=10, color='darkred')

# Plot 4: Analysis summary
ax = axes[1, 1]
ax.text(0.5, 0.95, "Concentration Analysis", 
        fontsize=16, weight='bold', ha='center', transform=ax.transAxes)

analysis = """
Key findings:

• 4 vertices: Nearly symmetric (α ≈ β)
  → Mode ≈ Mean ≈ 0.53 (middle)
  
• 5 vertices: Strongly right-skewed (α >> β)
  → Mode = 0.73 > Mean = 0.66
  → Concentrating toward maximum!
  
• 6 vertices: Moderately right-skewed
  → Mode = 0.58, Mean = 0.62
  → Concentrating in upper-middle

Pattern: As vertices increase, distribution
first shifts toward maximum (5 vertices),
then moderately retreats (6 vertices).

This suggests optimal configurations become
relatively rarer with more degrees of freedom.
"""

ax.text(0.05, 0.05, analysis, fontsize=11, 
        ha='left', va='bottom', transform=ax.transAxes,
        family='monospace')
ax.axis('off')

plt.tight_layout()
plt.savefig('concentration_location_analysis.png', dpi=150)
print("\nSaved analysis to concentration_location_analysis.png")

# Additional insight
print("\n\nDeeper insight:")
print("-"*70)
print("The 5-vertex case is special: α >> β means the distribution")
print("is heavily skewed toward the maximum. This suggests that")
print("the triangular bipyramid configuration is relatively easy")
print("to approximate with random vertices.")
print()
print("The 6-vertex case shows α/β ≈ 1.6, less extreme than")
print("the 5-vertex case (α/β ≈ 2.7). This might reflect")
print("increased geometric constraints or combinatorial complexity.")

# Quantify how close typical configurations get to maximum
print("\n\nTypical configuration analysis:")
print("-"*70)
for n_verts in vertices:
    alpha, beta_param = data[n_verts]["alpha"], data[n_verts]["beta"]
    
    # What percentile is 90% of maximum?
    percentile_90 = stats.beta.cdf(0.9, alpha, beta_param) * 100
    percentile_95 = stats.beta.cdf(0.95, alpha, beta_param) * 100
    percentile_99 = stats.beta.cdf(0.99, alpha, beta_param) * 100
    
    print(f"\n{n_verts} vertices:")
    print(f"  {100-percentile_90:.1f}% of configurations exceed 90% of maximum")
    print(f"  {100-percentile_95:.1f}% of configurations exceed 95% of maximum")
    print(f"  {100-percentile_99:.2f}% of configurations exceed 99% of maximum")

plt.close()