import torch
import numpy as np
from ideal_poly_volume_toolkit.geometry import lob_fast, _lob_value_series_torch

# Test the Lobachevsky function derivative directly
print("Testing Lobachevsky function derivative:")

# Test at a simple angle
theta = torch.tensor(np.pi/4, dtype=torch.float64, requires_grad=True)  # 45 degrees

# Compute Lobachevsky value
lob_val = lob_fast(theta, 96)
print(f"\nAt theta = π/4:")
print(f"  Λ(θ) = {lob_val.item():.6f}")

# Get autograd derivative
lob_val.backward()
print(f"  dΛ/dθ (autograd) = {theta.grad.item():.6f}")

# Finite difference
eps = 1e-7
lob_plus = _lob_value_series_torch(theta + eps, 96)
lob_minus = _lob_value_series_torch(theta - eps, 96) 
fd_deriv = (lob_plus - lob_minus) / (2 * eps)
print(f"  dΛ/dθ (finite diff) = {fd_deriv.item():.6f}")

# Theoretical derivative
theoretical = np.log(2 * np.abs(np.sin(np.pi/4)))
print(f"  dΛ/dθ (theoretical) = log(2|sin(θ)|) = {theoretical:.6f}")

# Test at several angles
print("\n\nTesting at multiple angles:")
test_angles = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]

for angle in test_angles:
    theta = torch.tensor(angle, dtype=torch.float64, requires_grad=True)
    lob_val = lob_fast(theta, 96)
    lob_val.backward()
    
    # Finite difference
    theta_no_grad = torch.tensor(angle, dtype=torch.float64)
    lob_plus = _lob_value_series_torch(theta_no_grad + eps, 96)
    lob_minus = _lob_value_series_torch(theta_no_grad - eps, 96)
    fd_deriv = (lob_plus - lob_minus) / (2 * eps)
    
    # Theoretical
    theoretical = np.log(2 * np.abs(np.sin(angle)))
    
    print(f"\nθ = {angle:.2f}:")
    print(f"  Autograd: {theta.grad.item():+.6f}")
    print(f"  Finite diff: {fd_deriv.item():+.6f}")
    print(f"  Theoretical: {theoretical:+.6f}")
    print(f"  Match? {'YES' if abs(theta.grad.item() - fd_deriv.item()) < 1e-4 else 'NO'}")