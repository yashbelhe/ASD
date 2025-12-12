"""
Finite-difference gradient test for a quadratic Bezier stroke.

We compare boundary-loss gradients against finite differences of the rendered
indicator area when nudging the curve width.
"""

from pathlib import Path
import torch
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.utils.boundary import BoundaryLossConfig, boundary_loss  # noqa: E402
from python.utils.segments import points_on_grid as grid_points  # noqa: E402
from python.integrands import QuadraticBezierIntegrandSlang  # noqa: E402


def evaluate_mean(integrand, pts):
    with torch.no_grad():
        vals = integrand(pts)
    return vals.mean().item()


def main():
    src = repo_root / "slang" / "bezier_curve.slang"
    dst = repo_root / "slang" / "__gen__bezier_curve.slang"
    compile_if_needed(src, dst)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    integrand = QuadraticBezierIntegrandSlang().to(device)
    cfg = BoundaryLossConfig(grid_size=3072, kde_k=13, num_subdivision=25)
    sample_points = grid_points(2048, jitter=False).to(device)

    eps = 5e-4

    integrand.zero_grad()
    loss = boundary_loss(integrand, cfg)
    loss.backward()
    boundary_grads = integrand.p.grad.detach().cpu().numpy()

    fd_grads = []
    with torch.no_grad():
        for idx in range(integrand.p.shape[0]):
            base = integrand.p[idx].item()
            integrand.p.data[idx] = base + eps
            mean_plus = evaluate_mean(integrand, sample_points)
            integrand.p.data[idx] = base - eps
            mean_minus = evaluate_mean(integrand, sample_points)
            integrand.p.data[idx] = base
            fd_grads.append((mean_plus - mean_minus) / (2.0 * eps))

    for idx, (b_grad, fd_grad) in enumerate(zip(boundary_grads, fd_grads)):
        print(f"param {idx}: boundary-loss = {b_grad:.6f}, finite-diff = {fd_grad:.6f}")


if __name__ == "__main__":
    main()
