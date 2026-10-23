"""Manually exercise the historical optimizer from a poor initialization."""

import torch
from ideal_poly_volume_toolkit.geometry import (
    delaunay_triangulation_indices,
    triangle_volume_from_points_torch,
)


def build_Z(thetas: torch.Tensor) -> torch.Tensor:
    Z = torch.empty(thetas.numel() + 2, dtype=torch.complex128, device=thetas.device)
    Z[0] = 1 + 0j
    Z[1] = 0 + 0j
    Z[2:] = torch.exp(1j * thetas.to(torch.complex128))
    return Z


def torch_sum_volume(Z_t: torch.Tensor, idx, series_terms: int) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64, device=Z_t.device)
    for i, j, k in idx:
        total = total + triangle_volume_from_points_torch(
            Z_t[i], Z_t[j], Z_t[k], series_terms=series_terms
        )
    return total


# Start with points very close together (bad configuration)
thetas = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
print(f"Initial thetas: {thetas}")

# Setup LBFGS
opt = torch.optim.LBFGS([thetas], lr=1.0, max_iter=20, line_search_fn="strong_wolfe")

for it in range(1, 21):
    with torch.no_grad():
        Z_np = build_Z(thetas).detach().cpu().numpy()
        idx = delaunay_triangulation_indices(Z_np)

    def closure():
        opt.zero_grad(set_to_none=True)
        Z_t = build_Z(thetas)
        total = torch_sum_volume(Z_t, idx, 96)
        loss = -total
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        Z_post = build_Z(thetas)
        val_post = torch_sum_volume(Z_post, idx, 96)
        print(f"[{it:02d}] volume = {val_post.item():.8f}, thetas = {thetas.numpy()}")
