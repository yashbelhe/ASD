"""
Run a simple gradient check on the circles shader.

Approximate integral over [0,1]^2 of the shader output and compare the
autograd gradient w.r.t radius against the analytic derivative.
"""

import math
from pathlib import Path
import torch
import sys

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.integrands import CirclesIntegrandSlang  # noqa: E402
from python.helpers import boundary_loss_slang, BoundaryLossConfig, points_on_grid as grid_points  # noqa: E402


def main():
    repo_root = Path(__file__).resolve().parents[1].parent
    src = repo_root / "slang" / "circle_single.slang"
    dst = repo_root / "slang" / "__gen__circle_single.slang"
    compile_if_needed(src, dst)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    integrand = CirclesIntegrandSlang().to(device)

    # Single circle parameters (cx, cy, r)
    cx, cy, r = 0.5, 0.5, 0.4
    integrand.p.data = torch.tensor([cx, cy, r], device=device, requires_grad=True)

    # Boundary-driven loss using edge sampling
    # Make sure grad buffers are clear
    integrand.zero_grad()

    # Run boundary loss and backprop using a config object
    cfg = BoundaryLossConfig(grid_size=2**13, kde_k=13, num_subdivision=20)
    b_loss = boundary_loss_slang(integrand, cfg)
    b_loss.backward()
    got = integrand.p.grad[2].item()
    ref = -2.0 * math.pi * r  # analytic boundary derivative for area outside a circle
    print(f"boundary grad dI/dr ≈ {got:.6f}, reference (analytic) ≈ {ref:.6f}")

    return got, ref


if __name__ == "__main__":
    main()
