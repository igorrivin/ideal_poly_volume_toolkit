import numpy as np

# The Lobachevsky function is defined as:
# Λ(θ) = 0.5 * Σ(k=1 to ∞) sin(2kθ)/k²

# Its derivative is:
# dΛ/dθ = 0.5 * Σ(k=1 to ∞) 2k*cos(2kθ)/k² = Σ(k=1 to ∞) cos(2kθ)/k

# Let's verify this matches the claimed formula log(2|sin(θ)|)

def lob_series(theta, n=100):
    """Compute Lobachevsky function using series"""
    result = 0.0
    for k in range(1, n+1):
        result += np.sin(2*k*theta) / (k*k)
    return 0.5 * result

def lob_deriv_series(theta, n=100):
    """Compute derivative using series"""
    result = 0.0
    for k in range(1, n+1):
        result += np.cos(2*k*theta) / k
    return result

# Test at several angles
angles = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

print("Comparing derivative formulas:")
print("θ      Series deriv   log(2|sin(θ)|)  -log(2|sin(θ)|)  Match?")
print("-" * 65)

for theta in angles:
    series = lob_deriv_series(theta, 200)
    log_formula = np.log(2 * np.abs(np.sin(theta)))
    neg_log_formula = -np.log(2 * np.abs(np.sin(theta)))
    
    match_pos = "YES" if abs(series - log_formula) < 0.01 else "NO"
    match_neg = "YES" if abs(series - neg_log_formula) < 0.01 else "NO"
    
    print(f"{theta:.1f}    {series:+.6f}      {log_formula:+.6f}      {neg_log_formula:+.6f}      {match_neg}")

print("\nConclusion: The correct formula is dΛ/dθ = -log(2|sin(θ)|)")

# Double check with finite differences
print("\n\nDouble-checking with finite differences:")
eps = 1e-7
for theta in [0.5, 1.0, 1.5]:
    lob_plus = lob_series(theta + eps, 200)
    lob_minus = lob_series(theta - eps, 200)
    fd = (lob_plus - lob_minus) / (2 * eps)
    series = lob_deriv_series(theta, 200)
    print(f"θ={theta}: FD={fd:+.6f}, Series={series:+.6f}, Difference={abs(fd-series):.2e}")