import numpy as np
import matplotlib.pyplot as plt

print("Why Computing E[Volume] Analytically is Intractable")
print("="*70)

print("\nThe mathematical setup:")
print("Ideal tetrahedron with vertices at 0, 1, ∞, and w")
print("where w is uniformly distributed on the Riemann sphere\n")

print("The volume formula (Bloch-Wigner):")
print("V(w) = Re[D(w) + D(w/(w-1)) + D(1-1/w)]")
print("where D(z) = Im[Li₂(z) + log|z| log(1-z)]")

print("\nThe expectation integral:")
print("E[V] = (1/4π) ∫∫_S² V(stereographic(θ,φ)) sin(θ) dθ dφ")
print("     = (1/4π) ∫₀^π ∫₀^{2π} V(cot(θ/2)e^{iφ}) sin(θ) dθ dφ")

print("\nExpanding one dilogarithm term:")
print("D(cot(θ/2)e^{iφ}) involves:")
print("  - Li₂(cot(θ/2)e^{iφ})")
print("  - log|cot(θ/2)| · log|1 - cot(θ/2)e^{iφ}|")
print("  - Arg[1 - cot(θ/2)e^{iφ}] · log|cot(θ/2)|")

print("\nThe other terms are even more complex:")
print("D(cot(θ/2)e^{iφ}/(cot(θ/2)e^{iφ}-1)) - rational function argument")
print("D(1-1/cot(θ/2)e^{-iφ}) - another composition")

print("\nWhy analytical integration fails:")
print("1. No separation of variables - θ and φ are entangled")
print("2. Dilogarithm has no simple antiderivative")
print("3. Branch cuts create discontinuities")
print("4. Compositions of special functions lack closed forms")

# Show how volume varies with parameters
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Simplified visualization - just show the integrand complexity
theta = np.linspace(0.1, np.pi-0.1, 100)
phi = np.linspace(0, 2*np.pi, 100)

# 1. |w| as function of theta (stereographic projection)
ax = axes[0, 0]
w_abs = 1/np.tan(theta/2)  # |cot(θ/2)|
ax.semilogy(theta*180/np.pi, w_abs)
ax.set_xlabel('θ (degrees)')
ax.set_ylabel('|w| = |cot(θ/2)|')
ax.set_title('Magnitude of w vs Polar Angle')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.1, 100)

# 2. Argument structure
ax = axes[0, 1]
theta_sample = np.pi/3
phi_grid = np.linspace(0, 2*np.pi, 200)
w_sample = np.exp(1j*phi_grid) / np.tan(theta_sample/2)
# Show one of the dilog arguments
arg1 = w_sample / (w_sample - 1)
ax.plot(phi_grid*180/np.pi, np.abs(arg1), label='|w/(w-1)|')
ax.plot(phi_grid*180/np.pi, np.angle(arg1)*180/np.pi/50, label='Arg[w/(w-1)]/50')
ax.set_xlabel('φ (degrees)')
ax.set_ylabel('Value')
ax.set_title(f'Dilog argument at θ={theta_sample*180/np.pi:.0f}°')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Measure element
ax = axes[1, 0]
ax.plot(theta*180/np.pi, np.sin(theta), 'g-', linewidth=2)
ax.fill_between(theta*180/np.pi, 0, np.sin(theta), alpha=0.3, color='green')
ax.set_xlabel('θ (degrees)')
ax.set_ylabel('sin(θ)')
ax.set_title('Measure Element on Sphere')
ax.grid(True, alpha=0.3)

# 4. Schematic of integral
ax = axes[1, 1]
ax.text(0.5, 0.8, r'$E[V] = \frac{1}{4\pi} \int_0^\pi \int_0^{2\pi} V(w(\theta,\phi)) \sin\theta \, d\phi \, d\theta$',
        ha='center', va='center', fontsize=14, transform=ax.transAxes)
ax.text(0.5, 0.6, r'where $w = \cot(\theta/2) e^{i\phi}$',
        ha='center', va='center', fontsize=12, transform=ax.transAxes)
ax.text(0.5, 0.4, r'and $V(w) = \Re[\mathcal{D}(w) + \mathcal{D}\left(\frac{w}{w-1}\right) + \mathcal{D}\left(1-\frac{1}{w}\right)]$',
        ha='center', va='center', fontsize=12, transform=ax.transAxes)
ax.text(0.5, 0.2, 'No known analytical solution!',
        ha='center', va='center', fontsize=16, weight='bold', color='red',
        transform=ax.transAxes)
ax.axis('off')

plt.tight_layout()
plt.savefig('analytical_challenge.png', dpi=150)
print("\nSaved visualization to analytical_challenge.png")

# Compare approaches
print("\n\nComparison of approaches:")
print("-"*50)
print("Analytical: ❌ Intractable due to dilogarithm integrals")
print("Numerical quadrature: ⚠️  Difficult due to branch cuts")
print("Monte Carlo: ✓ Works well, converges as 1/√n")
print("  - 10K samples: ~3 digits precision")
print("  - 100K samples: ~3.5 digits precision")
print("  - 10M samples: ~4.5 digits precision")

print("\nThe power of Monte Carlo:")
print("- Handles branch cuts automatically")
print("- No need to understand the analytic structure")
print("- Parallelizable")
print("- Provides error estimates via CLT")

print("\nPhilosophical note:")
print("Even though we have an 'exact' formula (Bloch-Wigner),")
print("computing its expectation requires numerical methods.")
print("This is a beautiful example where 'exact' ≠ 'analytically tractable'!")

plt.close()