import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Data from our analyses
data = {
    4: {"alpha": 1.077, "beta": 0.974, "mean_ratio": 0.537, "name": "4 vertices"},
    5: {"alpha": 4.77, "beta": 1.80, "mean_ratio": 0.658, "name": "5 vertices"}, 
    6: {"alpha": 6.33, "beta": 3.89, "mean_ratio": 0.620, "name": "6 vertices"}
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Beta distributions comparison
ax = axes[0, 0]
x = np.linspace(0, 1, 1000)
colors = ['blue', 'green', 'red']

for i, (n_verts, params) in enumerate(data.items()):
    y = stats.beta.pdf(x, params["alpha"], params["beta"])
    ax.plot(x, y, color=colors[i], linewidth=2.5, label=params["name"])
    
    # Mark mean
    mean = params["alpha"] / (params["alpha"] + params["beta"])
    ax.axvline(mean, color=colors[i], linestyle=':', alpha=0.7)

ax.set_xlabel('Normalized Volume (V/V_max)', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Volume Distributions for Random Ideal Polyhedra', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# 2. Concentration evolution
ax = axes[0, 1]
vertices = list(data.keys())
concentrations = [d["alpha"] + d["beta"] for d in data.values()]
alphas = [d["alpha"] for d in data.values()]
betas = [d["beta"] for d in data.values()]

ax.plot(vertices, concentrations, 'ko-', markersize=10, linewidth=2, label='α + β')
ax.plot(vertices, alphas, 'go-', markersize=8, linewidth=2, label='α')
ax.plot(vertices, betas, 'ro-', markersize=8, linewidth=2, label='β')

ax.set_xlabel('Number of Vertices', fontsize=12)
ax.set_ylabel('Parameter Value', fontsize=12)
ax.set_title('Beta Distribution Parameters', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# 3. Mean/Max ratio
ax = axes[1, 0]
mean_ratios = [d["mean_ratio"] for d in data.values()]
ax.plot(vertices, mean_ratios, 'bo-', markersize=12, linewidth=3)
ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Maximum')

for i, (v, r) in enumerate(zip(vertices, mean_ratios)):
    ax.text(v, r + 0.02, f'{r:.1%}', ha='center', fontsize=11)

ax.set_xlabel('Number of Vertices', fontsize=12)
ax.set_ylabel('Mean Volume / Maximum Volume', fontsize=12)
ax.set_title('How Close Random Configurations Get to Maximum', fontsize=14)
ax.set_ylim(0.4, 0.8)
ax.grid(True, alpha=0.3)

# 4. Summary statistics
ax = axes[1, 1]
ax.axis('off')

summary_text = """
VOLUME DISTRIBUTIONS OF RANDOM IDEAL POLYHEDRA

Key Patterns:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• All follow Beta distributions
• Concentration increases: 2.1 → 6.6 → 10.2
• Mean/max ratio peaks at 5 vertices

Special Behaviors:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4 vertices: Nearly symmetric (α ≈ β)
           Mean = 54% of maximum

5 vertices: Strongly right-skewed (α >> β)  
           Mean = 66% of maximum
           Easiest to approximate maximum

6 vertices: Moderately skewed
           Mean = 62% of maximum
           Concentration continues growing

Theoretical Insight:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Beta emerges as natural distribution for
bounded geometric quantities, similar to
how Gaussian emerges for unbounded sums
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('volume_distributions_summary.png', dpi=150, bbox_inches='tight')
print("Saved comprehensive summary to volume_distributions_summary.png")

# Also create a simple table
print("\n" + "="*60)
print("BETA DISTRIBUTION PARAMETERS SUMMARY")
print("="*60)
print("Vertices | α      | β      | α+β    | Mean/Max")
print("-"*60)
for v, d in data.items():
    print(f"{v:^8} | {d['alpha']:^6.2f} | {d['beta']:^6.2f} | "
          f"{d['alpha']+d['beta']:^6.1f} | {d['mean_ratio']:^8.1%}")
print("="*60)