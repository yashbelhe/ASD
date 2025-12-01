"""
Finite-difference gradient test for a half-plane indicator shader.

We integrate the shader output over [0, 1]^2, approximate the derivative of
that integral w.r.t the threshold parameter using central finite differences,
and compare it against the analytic gradient.
"""

from pathlib import Path
import torch
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.helpers import BoundaryLossConfig, boundary_loss_slang, points_on_grid as grid_points  # noqa: E402
from python.integrands import HalfPlaneIntegrandSlang  # noqa: E402


def evaluate_mean(integrand, pts):
    with torch.no_grad():
        vals = integrand(pts)
    return vals.mean().item()


def main():
    src = repo_root / "slang" / "half_plane.slang"
    dst = repo_root / "slang" / "__gen__half_plane.slang"
    compile_if_needed(src, dst)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    integrand = HalfPlaneIntegrandSlang(threshold=0.5).to(device)
    cfg = BoundaryLossConfig(grid_size=4096, kde_k=15, num_subdivision=20)
    sample_points = grid_points(4096, jitter=False).to(device)

    eps = 1e-3

    integrand.zero_grad()
    loss = boundary_loss_slang(integrand, cfg)
    loss.backward()
    boundary_grad = integrand.p.grad[0].item()

    with torch.no_grad():
        base = integrand.p[0].item()
        integrand.p.data[0] = base + eps
        mean_plus = evaluate_mean(integrand, sample_points)
        integrand.p.data[0] = base - eps
        mean_minus = evaluate_mean(integrand, sample_points)
        integrand.p.data[0] = base

    fd_grad = (mean_plus - mean_minus) / (2.0 * eps)

    print(f"boundary-loss grad dL/dt = {boundary_grad:.6f}")
    print(f"finite-diff grad  dI/dt = {fd_grad:.6f}")


if __name__ == "__main__":
    main()
