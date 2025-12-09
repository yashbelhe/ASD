"""Gradient check for multi-circle shader using boundary loss and finite differences."""

import math
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(repo_root))

from compiler.compile_shader import compile_if_needed  # noqa: E402
from python.integrands import MultiCirclesIntegrandSlang  # noqa: E402
from python.helpers import BoundaryLossConfig, boundary_loss  # noqa: E402


def main():
    src = repo_root / "slang" / "circles.slang"
    dst = repo_root / "slang" / "__gen__circles.slang"
    compile_if_needed(src, dst)

    # Build a small stack of circles: (cx, cy, radius, opacity)
    circles = [
        (0.3, 0.3, 0.15, 0.8),
        (0.6, 0.5, 0.20, 0.5),
        (0.4, 0.7, 0.10, 0.9),
    ]
    integrand = MultiCirclesIntegrandSlang(circles).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Choose a parameter index to differentiate (e.g., radius of the second circle)
    param_idx = 2 + 4  # radius of circle at index 1
    eps = 1e-4

    cfg = BoundaryLossConfig(grid_size=512, kde_k=11, num_subdivision=20)

    integrand.zero_grad()
    loss = boundary_loss(integrand, cfg)
    loss.backward()
    auto_grad = integrand.p.grad[param_idx].item()

    with torch.no_grad():
        base = integrand.p[param_idx].item()
        integrand.p[param_idx] = base + eps
        loss_plus = boundary_loss(integrand, cfg).item()
        integrand.p[param_idx] = base - eps
        loss_minus = boundary_loss(integrand, cfg).item()
        integrand.p[param_idx] = base

    fd_grad = (loss_plus - loss_minus) / (2.0 * eps)

    print(f"Auto grad (param {param_idx}) = {auto_grad:.6f}")
    print(f"Finite diff grad          = {fd_grad:.6f}")
    print(f"Absolute error            = {abs(auto_grad - fd_grad):.6e}")


if __name__ == "__main__":
    main()
