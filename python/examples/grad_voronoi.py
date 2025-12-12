"""Gradient check for 3x3 Voronoi shader (site positions only)."""

from pathlib import Path
import torch
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.utils.boundary import BoundaryLossConfig, boundary_loss  # noqa: E402
from python.utils.segments import points_on_grid as grid_points  # noqa: E402
from python.integrands import VoronoiGridIntegrandSlang  # noqa: E402


def area_loss_tensor(integrand, pts):
    vals = integrand(pts)
    return vals.sum(dim=-1).mean()


def total_loss_value(integrand, pts, cfg):
    with torch.no_grad():
        area = integrand(pts).sum(dim=-1).mean().item()
        boundary = boundary_loss(integrand, cfg).item()
    return area + boundary


def main():
    src = repo_root / "slang" / "voronoi_simple.slang"
    dst = repo_root / "slang" / "__gen__voronoi_simple.slang"
    compile_if_needed(src, dst)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    integrand = VoronoiGridIntegrandSlang().to(device)
    cfg = BoundaryLossConfig(grid_size=4096, kde_k=11, num_subdivision=20)
    sample_points = grid_points(4096, jitter=False).to(device)

    eps = 1e-4

    integrand.zero_grad()
    area_loss = area_loss_tensor(integrand, sample_points)
    boundary_term = boundary_loss(integrand, cfg)
    total_loss = area_loss + boundary_term
    total_loss.backward()
    total_grads = integrand.p.grad.detach().cpu().numpy()

    fd_grads = []
    with torch.no_grad():
        for site_idx in range(9):
            for coord_offset in range(2):
                idx = site_idx * 5 + coord_offset
                base = integrand.p[idx].item()
                integrand.p.data[idx] = base + eps
                loss_plus = total_loss_value(integrand, sample_points, cfg)
                integrand.p.data[idx] = base - eps
                loss_minus = total_loss_value(integrand, sample_points, cfg)
                integrand.p.data[idx] = base
                fd_grads.append((loss_plus - loss_minus) / (2.0 * eps))

    coord_grads = total_grads.reshape(-1, 5)[:, :2].reshape(-1)
    for idx, (our_grad, fd_grad) in enumerate(zip(coord_grads, fd_grads)):
        print(f"param {idx}: total-grad = {our_grad:.6f}, finite-diff = {fd_grad:.6f}")


if __name__ == "__main__":
    main()
