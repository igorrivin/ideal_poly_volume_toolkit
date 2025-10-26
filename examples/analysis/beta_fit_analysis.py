import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ideal_poly_volume_toolkit.geometry import ideal_poly_volume_via_delaunay
import warnings
warnings.filterwarnings('ignore')

def generate_volumes(n_samples, seed):
    """Generate volume samples"""
    np.random.seed(seed)
    volumes = []
    
    for i in range(n_samples):
        vec = np.random.randn(3)
        vec = vec / np.linalg.norm(vec)
        x, y, z = vec
        
        if z > 0.999:
            continue
            
        w = complex(x/(1-z), y/(1-z))
        
        if abs(w) < 0.01 or abs(w-1) < 0.01:
            continue
        
        vertices = np.array([0+0j, 1+0j, w])
        vol = ideal_poly_volume_via_delaunay(vertices, mode='fast', series_terms=96)
        volumes.append(vol)
    
    return np.array(volumes)

print("Beta Distribution Fit Analysis")
print("="*70)

# Generate a large sample
print("Generating 100,000 samples for precise parameter estimation...")
volumes = generate_volumes(100000, seed=42)
print(f"Actual samples: {len(volumes)}")

# Fit beta distribution
print("\nFitting beta distribution...")
params = stats.beta.fit(volumes, floc=0, fscale=1.02)  # Fix location and scale
alpha, beta_param = params[0], params[1]

print(f"\nBeta parameters:")
print(f"  α = {alpha:.6f}")
print(f"  β = {beta_param:.6f}")
print(f"  α - β = {alpha - beta_param:.6f}")

# Bootstrap confidence intervals
print("\nBootstrap confidence intervals (1000 resamples)...")
n_bootstrap = 1000
alpha_samples = []
beta_samples = []
asymmetry_samples = []

for i in range(n_bootstrap):
    if i % 100 == 0:
        print(f"  Progress: {i}/{n_bootstrap}")
    
    # Resample with replacement
    resample = np.random.choice(volumes, size=len(volumes), replace=True)
    
    # Fit beta
    try:
        params_boot = stats.beta.fit(resample, floc=0, fscale=1.02)
        alpha_boot, beta_boot = params_boot[0], params_boot[1]
        alpha_samples.append(alpha_boot)
        beta_samples.append(beta_boot)
        asymmetry_samples.append(alpha_boot - beta_boot)
    except:
        pass

alpha_samples = np.array(alpha_samples)
beta_samples = np.array(beta_samples)
asymmetry_samples = np.array(asymmetry_samples)

# Confidence intervals
print(f"\n95% Confidence Intervals:")
print(f"  α: [{np.percentile(alpha_samples, 2.5):.6f}, {np.percentile(alpha_samples, 97.5):.6f}]")
print(f"  β: [{np.percentile(beta_samples, 2.5):.6f}, {np.percentile(beta_samples, 97.5):.6f}]")
print(f"  α - β: [{np.percentile(asymmetry_samples, 2.5):.6f}, {np.percentile(asymmetry_samples, 97.5):.6f}]")

# Test if α = β = 1 (uniform distribution)
print("\n\nHypothesis Tests:")
print("-"*50)

# Test 1: Are parameters equal to 1?
print("Test 1: H0: α = 1")
t_alpha = (alpha - 1) / np.std(alpha_samples)
p_alpha = 2 * (1 - stats.norm.cdf(abs(t_alpha)))
print(f"  t-statistic: {t_alpha:.4f}")
print(f"  p-value: {p_alpha:.4f}")
print(f"  Conclusion: {'Reject H0' if p_alpha < 0.05 else 'Fail to reject H0'}")

print("\nTest 2: H0: β = 1")
t_beta = (beta_param - 1) / np.std(beta_samples)
p_beta = 2 * (1 - stats.norm.cdf(abs(t_beta)))
print(f"  t-statistic: {t_beta:.4f}")
print(f"  p-value: {p_beta:.4f}")
print(f"  Conclusion: {'Reject H0' if p_beta < 0.05 else 'Fail to reject H0'}")

# Test 2: Is there asymmetry?
print("\nTest 3: H0: α = β (symmetric distribution)")
contains_zero = np.percentile(asymmetry_samples, 2.5) <= 0 <= np.percentile(asymmetry_samples, 97.5)
print(f"  95% CI for α - β: [{np.percentile(asymmetry_samples, 2.5):.6f}, {np.percentile(asymmetry_samples, 97.5):.6f}]")
print(f"  Contains 0? {'Yes' if contains_zero else 'No'}")
print(f"  Conclusion: {'Fail to reject H0 (no asymmetry)' if contains_zero else 'Reject H0 (significant asymmetry)'}")

# Visual check
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Histogram with beta fit
ax = axes[0, 0]
n, bins, _ = ax.hist(volumes, bins=100, density=True, alpha=0.6, color='blue', label='Data')
x = np.linspace(0.001, 1.019, 1000)
ax.plot(x, stats.beta.pdf(x, alpha, beta_param, 0, 1.02), 'r-', linewidth=2, 
        label=f'Beta({alpha:.3f}, {beta_param:.3f})')
ax.plot(x, stats.uniform.pdf(x, 0, 1.02), 'g--', linewidth=2, label='Uniform(0, 1.02)')
ax.set_xlabel('Volume')
ax.set_ylabel('Density')
ax.set_title('Data vs Beta Fit vs Uniform')
ax.legend()
ax.set_xlim(0, 1.05)

# 2. Q-Q plot
ax = axes[0, 1]
theoretical_quantiles = np.linspace(0.001, 0.999, 999)
empirical_quantiles = np.percentile(volumes, theoretical_quantiles * 100)
beta_quantiles = stats.beta.ppf(theoretical_quantiles, alpha, beta_param, 0, 1.02)
ax.plot(beta_quantiles, empirical_quantiles, 'b.', alpha=0.5, markersize=2)
ax.plot([0, 1.02], [0, 1.02], 'r--', linewidth=2)
ax.set_xlabel('Beta Distribution Quantiles')
ax.set_ylabel('Empirical Quantiles')
ax.set_title('Q-Q Plot: Data vs Beta')

# 3. Residual plot
ax = axes[1, 0]
residuals = empirical_quantiles - beta_quantiles
ax.plot(beta_quantiles, residuals, 'b.', alpha=0.5, markersize=2)
ax.axhline(0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Beta Quantiles')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot')
ax.grid(True, alpha=0.3)

# 4. Bootstrap distribution of asymmetry
ax = axes[1, 1]
ax.hist(asymmetry_samples, bins=50, density=True, alpha=0.7, color='purple')
ax.axvline(0, color='r', linestyle='--', linewidth=2, label='α = β')
ax.axvline(alpha - beta_param, color='g', linestyle='-', linewidth=2, 
           label=f'Observed: {alpha - beta_param:.4f}')
ax.set_xlabel('α - β')
ax.set_ylabel('Density')
ax.set_title('Bootstrap Distribution of Asymmetry')
ax.legend()

plt.tight_layout()
plt.savefig('beta_fit_analysis.png', dpi=150)
print("\nSaved plots to beta_fit_analysis.png")

# Alternative distribution tests
print("\n\nAlternative Distribution Fits:")
print("-"*50)

# Try other distributions
distributions = [
    ('uniform', stats.uniform),
    ('beta', stats.beta),
    ('kumaraswamy', None),  # Would need special implementation
]

# Kolmogorov-Smirnov tests
print("\nKolmogorov-Smirnov Tests:")
ks_uniform = stats.kstest(volumes, lambda x: stats.uniform.cdf(x, 0, 1.02))
print(f"  Uniform(0, 1.02): D={ks_uniform.statistic:.6f}, p={ks_uniform.pvalue:.6f}")

beta_cdf = lambda x: stats.beta.cdf(x, alpha, beta_param, 0, 1.02)
ks_beta = stats.kstest(volumes, beta_cdf)
print(f"  Beta({alpha:.3f}, {beta_param:.3f}): D={ks_beta.statistic:.6f}, p={ks_beta.pvalue:.6f}")

# Final summary
print("\n" + "="*70)
print("SUMMARY:")
print(f"  With {len(volumes)} samples, the fitted parameters are:")
print(f"    α = {alpha:.4f} ± {np.std(alpha_samples):.4f}")
print(f"    β = {beta_param:.4f} ± {np.std(beta_samples):.4f}")
print(f"\n  The asymmetry α - β = {alpha - beta_param:.4f} is "
      f"{'statistically significant' if not contains_zero else 'NOT statistically significant'}")
print(f"\n  The distribution is {'very close to' if abs(alpha-1) < 0.1 and abs(beta_param-1) < 0.1 else 'significantly different from'} uniform")

plt.close()