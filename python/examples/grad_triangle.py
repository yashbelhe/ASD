"""
Finite-difference gradient test for a filled triangle indicator shader.

We compare the boundary-loss gradient with a finite-difference estimate of the
rendered area when nudging one vertex coordinate.
"""

from pathlib import Path
import torch
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.helpers import BoundaryLossConfig, boundary_loss_slang, points_on_grid as grid_points  # noqa: E402
from python.integrands import TriangleIntegrandSlang  # noqa: E402


def evaluate_mean(integrand, pts):
    with torch.no_grad():
        vals = integrand(pts)
    return vals.mean().item()


def main():
    src = repo_root / "slang" / "triangle.slang"
    dst = repo_root / "slang" / "__gen__triangle.slang"
    compile_if_needed(src, dst)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    integrand = TriangleIntegrandSlang().to(device)
    cfg = BoundaryLossConfig(grid_size=2048, kde_k=11, num_subdivision=20)
    sample_points = grid_points(2048, jitter=False).to(device)

    param_idx = 0  # v0.x
    eps = 5e-4

    integrand.zero_grad()
    loss = boundary_loss_slang(integrand, cfg)
    loss.backward()
    boundary_grad = integrand.p.grad[param_idx].item()

    with torch.no_grad():
        base = integrand.p[param_idx].item()
        integrand.p.data[param_idx] = base + eps
        mean_plus = evaluate_mean(integrand, sample_points)
        integrand.p.data[param_idx] = base - eps
        mean_minus = evaluate_mean(integrand, sample_points)
        integrand.p.data[param_idx] = base

    fd_grad = (mean_plus - mean_minus) / (2.0 * eps)

    print(f"boundary-loss grad dL/dv0x = {boundary_grad:.6f}")
    print(f"finite-diff grad  dI/dv0x = {fd_grad:.6f}")


if __name__ == "__main__":
    main()
